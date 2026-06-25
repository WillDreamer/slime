# Workspace-Bench RL on Greenland (AECE) — Qwen3.5-4B-Base

Migration of the `examples/tau-bench` RL framework to
[Workspace-Bench](https://github.com/OpenDataBox/Workspace-Bench): a benchmark
where an agent is given a large **workspace of files** + a task instruction,
**edits files**, and produces output files. Reward is a **rubric LLM-judge**
(Claude on Bedrock), not tau's deterministic DB-hash.

Two fundamental differences from tau-bench drove the design:

1. **No user simulator** — Workspace-Bench is single-shot/agentic, not
   conversational. So the entire GLM/Bedrock user-sim + multi-model
   `--sglang-config` machinery is **deleted**. The episode ends when the agent
   calls the `finish` terminate tool.
2. **Reward = rubric judge** — each task ships an array of free-text `rubrics`; a
   Claude judge marks each pass/fail and the reward is the **fraction passed**
   ∈ [0, 1] (dense). We call Bedrock directly (reusing
   `greenland_utils/api_usage.py`'s `invoke_model` pattern) — NOT Workspace-Bench's
   upstream ClaudeCode.js judge, so there is no node / external-endpoint dependency.

## What changed vs. tau-bench

| Piece | tau-bench (`examples/tau-bench`) | Workspace-Bench (this dir) |
|---|---|---|
| Agent loop | `trainable_agents.py` (multi-turn tool-use) | **reused ~verbatim** (imports → `ws_bench`, `TAU_*`→`WS_*`, file-agent tool-instruction text, judge time-attribution) |
| Tools | retail/airline API tools (mutate in-memory `data`) | **file tools** `list_dir/read_file/grep/write_file/edit_file/finish/think` on a real workspace (`ws_bench/envs/tools.py`) |
| Env | `tau_bench` (DB state, user sim) | **`WorkspaceEnv`** (`ws_bench/envs/base.py`): reset materializes a per-rollout workspace; finish → rubric judge |
| Reward | DB-state hash + output-substring match | **Bedrock rubric judge** (`judge.py`), reward = fraction of rubrics passed |
| User simulator | GLM-4.7-Flash (multi-model YAML) or Bedrock Claude | **none** (deleted) |
| Tool-call parser | `qwen3_coder` | `qwen3_coder` (unchanged — matches Qwen3.5 template) |
| Data | `tau1_mock.py` → `retail_*_tasks.jsonl` | **`prepare_ws_data.py`** (HF download → jsonl + workspaces → S3) |
| Launcher | `run_qwen35_4b_tau_mns_async.sh` | **`run_qwen35_4b_ws_mns_async.sh`** (same scaffold, single-model EAGLE, judge egress self-test) |

The hard-won token-alignment / loss-mask / `<think>` machinery in
`trainable_agents.py` is reused unchanged — see the `tau-*` / `slime-think-*`
memories for why each piece exists. The env-interface contract it relies on
(`tools_info`, `wiki`, `reset→EnvResetResponse`, `step→EnvResponse`,
`terminate_tools`) is satisfied by `WorkspaceEnv`.

## Files

- `run_qwen35_4b_ws_mns_async.sh` — the Greenland launcher (submit this).
- `generate_with_ws.py` — slime custom rollout entry (`--custom-generate-function-path generate_with_ws.generate`). Builds the env, runs `agent.asolve`, converts to `Sample` (keeps tau's degenerate/empty/truncated guards verbatim).
- `trainable_agents.py` — multi-turn tool-use agent + training-tensor reconstruction (reused from tau).
- `async_env.py` — async facade over the sync `WorkspaceEnv` (offloads file I/O + the blocking judge call to a thread pool).
- `judge.py` — **Bedrock Claude rubric judge** (the reward). Singleton, retry, conservative parse.
- `openai_tool_adapter.py`, `sglang_tool_parser.py` — tool-call parsing (verbatim; imports → `ws_bench`).
- `ws_bench/` — vendored env package: `types.py`, `envs/{base,tools,tool,wiki,__init__}.py`, `agents/{base,tool_calling_agent}.py`.
- `prepare_ws_data.py` — download Workspace-Bench from HF, normalize, write jsonl + index map, stage to S3.

## Reward: the Bedrock rubric judge

`judge.py::RubricJudge` (selected by `WS_JUDGE_MODEL_ID`) mirrors `api_usage.py`:
`invoke_model` with the Anthropic body + automatic `temperature` drop for
opus-4-7/4-8. It uses the **default boto3 credential chain** — in the Greenland
container the bootstrap writes `/root/.aws/config` (`credential_source=EcsContainer`
→ assume `greenland-dev-role` @ 339712697413, which has Bedrock access); on a dev
box set `WS_JUDGE_PROFILE=greenland-dev`.

The judge inlines the agent's produced output-file contents into the prompt
(Bedrock can't read the filesystem), demands strict per-rubric JSON
(`{"rubrics":[{"index","passed","confidence","evidence"}]}`), and defaults any
missing/garbled verdict to `passed=false` (never reward unverifiable work).
`reward = num_passed / num_rubrics`. The tau format penalty (-0.1 on format-bad
success) and truncation reward (-0.2) layer on top, unchanged.

Configured by env (set in the run script, propagated to all rollout actors via the
Ray job runtime-env):

| Env var | Default | Meaning |
|---|---|---|
| `WS_JUDGE_MODEL_ID` | `us.anthropic.claude-sonnet-4-6` | Bedrock judge model |
| `WS_BEDROCK_REGION` | `us-east-1` | Bedrock region |
| `WS_JUDGE_FILE_CHARS` | `12000` | per-output-file char budget inlined into the prompt |
| `WS_ENV_THREAD_WORKERS` | `256` | thread pool for concurrent judge calls |
| `WS_ENABLE_THINKING` / `WS_STRIP_HISTORICAL_THINK` | `1` / `1` | think machinery (watch `rollout/frac_trained` early) |

> The model id list (callable on `greenland-dev` in `us-east-1`) lives in
> `greenland_utils/api_usage.py::AVAILABLE_MODELS`. Sonnet 4.6 accepts
> `temperature=0`; `us.anthropic.claude-opus-4-5-20251101-v1:0` is a
> higher-accuracy override.

## Workspace isolation (software copy-on-write)

The 5 persona workspaces (`chanpin_raw/kaifa_raw/research_raw/yunying_raw/houqin_raw`,
up to ~20 GB total) are staged ONCE per node, read-only and shared. Each rollout
gets a private, near-empty overlay (`WS_ROLLOUT_ROOT/<task_id>/<group_index>/`):
reads check the overlay first then fall back to the shared base; writes/edits
copy-up into the overlay. So `n_samples_per_prompt=16` parallel rollouts of one
task cost ~(touched files) each, NOT 16 × 20 GB. The overlay is torn down after
the reward is computed.

## Data

`prepare_ws_data.py` downloads the task dataset + the `filesys_en_workdirs.zip`
workspaces from HF (`Workspace-Bench/Workspace-Bench`,
`Workspace-Bench/Workspace-Bench-Workspaces`), normalizes each task's
`file_system` (localized label/persona) to its `*_raw` workspace dir, and writes:

- `ws_<split>_train_tasks.jsonl` / `ws_<split>_dev_tasks.jsonl` — rows `{"index": i}` (slime `--input-key index`, 90/10 split).
- `task_index_map.json` — `{"train": [...], "dev": [...]}` mapping index → task folder.
- `tasks/<task_id>/metadata.json` (+ data_manifest files), `workspaces/<persona>_raw/`.

Two paths, both producing the above under `WS_DATA_DIR`:

1. **Pre-staged on S3 (primary)** — upload to `s3://whx-agent/data/workspace-bench/`
   and pull with `--stage-data workspace-bench/` at submit. The run script then
   **skips** download (idempotent check on the train jsonl).
   ```bash
   cd /home/whx/AECE/slime/examples/workspace-bench
   python3 prepare_ws_data.py --split full        # downloads, stages, uploads to S3
   ```
2. **In-script generation (fallback)** — if the data isn't present in `WS_DATA_DIR`,
   the run script runs `prepare_ws_data.py --skip-upload` on the head node.

The env locates a task at rollout time: `int(sample.prompt)` →
`task_index_map.json[<split>][i]` → `tasks/<task_id>/metadata.json` →
`file_system` → `workspaces/<persona>_raw/`.

## Submit

Disaggregated, async (required), no user-sim node — e.g. 3 nodes (1 train + 2 rollout):
```bash
cd /home/whx/AECE
export WANDB_API_KEY="<key>"
python3 greenland_cli_mns_async.py obx \
  --script examples/workspace-bench/run_qwen35_4b_ws_mns_async.sh \
  --num-nodes 3 --rollout-nodes 2 --user-sim-nodes 0 \
  --stage-model Qwen3.5/Qwen3.5-4B-Base/ \
  --stage-model Qwen3.5/Qwen3.5-4B-Base_torch_dist/ \
  --stage-data workspace-bench/
```

Smoke (single rollout node): `--num-nodes 2 --rollout-nodes 1 --user-sim-nodes 0`,
and override `--env WS_SPLIT=lite` plus a small `--num-rollout`/`ROLLOUT_BATCH_SIZE`
to validate the pipeline cheaply first.

## ⚠️ Open risks

1. **Bedrock egress from P5EN** — the job runs in **ap-south-1** but the judge
   calls Bedrock in **us-east-1** (cross-region, public endpoint). The run script
   runs a **loud judge self-test** at startup (`[ws judge self-test] …`); check it
   in the OBX CloudWatch logs first. If it fails, every rollout scores reward=0.
   (Same risk tau-bench's Bedrock user-sim path carried.)
2. **HF dataset layout** — `prepare_ws_data.py` assumes per-task `metadata.json`
   folders. If the HF repo ships a CSV/parquet instead, extend `stage_tasks()`
   (it raises a clear error rather than silently producing empty data).
3. **Workspace footprint** — measure the unzipped `workspaces/` size on NVMe
   before trusting the S3 staging / per-node disk budget; start on `lite` if tight.
4. **think-ON alignment** — the same Qwen3.5 think machinery as tau. Watch
   `rollout/frac_trained` and `rollout/align_fail_turns` in the FIRST few steps;
   if `frac_trained` collapses, set `WS_ENABLE_THINKING=0` (memory
   `slime-think-token-loss-mask`).
