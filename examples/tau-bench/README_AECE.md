# tau-bench RL on Greenland (AECE) — Qwen3.5-4B-Base

Migration of `examples/tau-bench` (multi-turn tool-use GRPO) to the AECE
Greenland multi-node flow, with the external OpenAI/Gemini user simulator
**replaced by Bedrock Claude** (`greenland_utils/api_usage.py`-style invoke).

## What changed vs. the upstream example

| Piece | Upstream (`/home/whx/slime`) | AECE (this dir) |
|---|---|---|
| Base model | Qwen3-4B / Qwen3-8B / 30B | **Qwen3.5-4B-Base** (`scripts/models/qwen3.5-4B.sh`) |
| User simulator | external OpenAI/Gemini/GLM API via `litellm` | **Bedrock Claude** (`UserStrategy.CLAUDE`, boto3, no external API) |
| tau-bench install | `git clone … && pip install` at runtime | **vendored** under `./tau_bench/` (synced via S3, importable on `PYTHONPATH`) |
| Tool-call parser | `qwen25` (JSON args) | **`qwen3_coder`** (XML: `<tool_call><function=..><parameter=..>`; matches Qwen3.5's template) |
| Tool guidance injection | `.replace("You may call one or more functions…")` | anchor-aware: Qwen3.5's XML template has no such phrase, so inject before the `If you choose to call a function…` format spec instead (old replace was a silent no-op) |
| Env driving | `await env.{reset,step}` on an async tau-bench | **`async_env.AsyncTauEnv`** wraps the sync vendored env (offloads blocking Bedrock calls to a thread pool) |
| Launcher | `run_qwen3_4B.sh` (bare `python3 train.py`, colocate) | **`run_qwen35_4b_tau_mns.sh`** (Ray `job submit`, colocate/disaggregated, EFA — cloned from the validated `math_reasoning/run_qwen35_4b_base_mns.sh`) |

## Files

- `run_qwen35_4b_tau_mns.sh` — the Greenland launcher (submit this).
- `generate_with_tau.py` — slime custom rollout entry (`--custom-generate-function-path`).
- `trainable_agents.py` — multi-turn tool-use agent + training-tensor reconstruction.
- `async_env.py` — async facade over the sync tau-bench env (NEW).
- `openai_tool_adapter.py`, `sglang_tool_parser.py` — tool-call parsing (verbatim).
- `tau_bench/` — vendored tau-bench (from `JD-ETH/tau-bench@feature/litellm-retry`),
  patched: lazy `litellm` import + `BedrockClaudeUserSimulationEnv` in `envs/user.py`.
- `tau1_mock.py` — regenerate full task metadata in-container (optional).

## User simulator: Bedrock Claude

`tau_bench/envs/user.py::BedrockClaudeUserSimulationEnv` (selected by
`user_strategy="claude"`) replaces the litellm path. It uses the **default boto3
credential chain**: in the Greenland container the bootstrap writes
`/root/.aws/config` (`credential_source=EcsContainer` → assume
`greenland-dev-role` @ 339712697413), which has Bedrock access; on a dev box use
`AWS_PROFILE=greenland-dev`. The invoke path mirrors `api_usage.py`:
`invoke_model` + automatic `temperature` drop for opus-4-7/4-8.

Configured entirely by env (set in the run script, propagated to all rollout
actors via the Ray job runtime-env):

| Env var | Default | Meaning |
|---|---|---|
| `TAU_USER_MODEL_ID` | `us.anthropic.claude-opus-4-7` | Bedrock model id for the user sim |
| `TAU_BEDROCK_REGION` | `us-east-1` | Bedrock region (model ids enabled here) |
| `TAU_USER_STRATEGY` | `claude` | user sim backend |
| `TAU_ENV` / `TAU_TASK_SPLIT` | `retail` / `train` | tau-bench env + split |
| `TAU_TOOL_PARSER` | `qwen3_coder` | sglang tool-call parser (must match the model template) |
| `TAU_ENV_THREAD_WORKERS` | `256` | thread pool size for concurrent user-sim calls |

> **Verified (2026-06-19):** `us.anthropic.claude-opus-4-7` answers multi-turn
> tau-style user-sim prompts via `greenland-dev` in `us-east-1`. The model id
> list lives in `greenland_utils/api_usage.py::AVAILABLE_MODELS`.

## Data

The rollout consumes only the task `index` (`--input-key index`). Two paths,
both producing `retail_{train,dev,test}_tasks.jsonl` (500 / 20 / 115):

1. **Pre-staged on S3 (primary)** — already uploaded to
   `s3://whx-agent/data/tau-bench/`. Pull it with `--stage-data tau-bench/` at
   submit time; the run script then **skips** generation (idempotent check on
   the train split).
2. **In-script generation (fallback)** — if the data isn't present in
   `$DATA_ROOT/tau-bench`, the run script runs the original example's
   `tau1_mock.py` (per `README.md`) on the main node, writing full task metadata
   (`{"index": i, "metadata": {...}}`) via the SAME vendored `tau_bench`. Needs
   no LLM/boto3 (`user_strategy=human`). This is wired into
   `run_qwen35_4b_tau_mns.sh` directly — no manual pre-step required.

## Submit

Colocate, 1 node (8 GPU):
```bash
cd /home/whx/AECE
python3 greenland_cli_mns.py obx \
  --script examples/tau-bench/run_qwen35_4b_tau_mns.sh --num-nodes 1 \
  --stage-model Qwen3.5/Qwen3.5-4B-Base/ \
  --stage-model Qwen3.5/Qwen3.5-4B-Base_torch_dist/ \
  --stage-data tau-bench/
```

Disaggregated, 3 nodes (1 train + 2 rollout):
```bash
python3 greenland_cli_mns.py obx \
  --script examples/tau-bench/run_qwen35_4b_tau_mns.sh --num-nodes 3 --rollout-nodes 2 \
  --stage-model Qwen3.5/Qwen3.5-4B-Base/ \
  --stage-model Qwen3.5/Qwen3.5-4B-Base_torch_dist/ \
  --stage-data tau-bench/
```

(`--stage-model`/`--stage-data` override the CLI's math defaults so only the
Qwen3.5 checkpoint + tau data are pulled to NVMe.)

## ⚠️ Open risk: Bedrock egress from the P5EN nodes

The job runs in **ap-south-1** but Bedrock is called in **us-east-1** (a
cross-region, public-endpoint call). S3 egress is known to work in-container,
but cross-region Bedrock egress on these nodes is **not pre-verified**. The run
script runs a **loud one-shot self-test** at startup
(`[tau egress self-test] …`); check it in the OBX CloudWatch logs first. If it
fails, every rollout's user-sim call will abort. Fallbacks:
1. set `TAU_BEDROCK_REGION` to a Bedrock region reachable from ap-south-1 with
   the model enabled, or
2. switch to the self-play user sim (`examples/tau-bench-async`, user sim through
   the same SGLang engine — no external API at all).
