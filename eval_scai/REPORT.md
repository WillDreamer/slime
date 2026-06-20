# Unified Benchmark Harness — Report

**Location:** `/xuanwu-tank/north/xw27/multi/eval`
**Purpose:** evaluate a given HF checkpoint on 8 benchmarks with one command:
**GPQA, MMLU, AIME, Browse-Comp-Plus, Search, Tau (tau2), IFBench, IFEval.**

```bash
# all 8 benchmarks
CKPT=/path/to/hf_checkpoint bash run_all.sh

# a subset, with overrides
CKPT=... EVAL_GPUS=2,3 TAU2_LIMIT=5 bash run_all.sh gpqa aime tau2
```

Results land in `results/<ckpt_name>/`, logs in `logs/<ckpt_name>/`.
inspect-based results are browsable with
`inspect view --log-dir results/<ckpt_name>/inspect_logs`.

---

## 1. Architecture

One sglang server is launched for the checkpoint (`serve_model.sh`, OpenAI-
compatible, default port **8400**, GPUs `EVAL_GPUS`, TP = #GPUs, with
`--reasoning-parser qwen3 --tool-call-parser qwen` so thinking models and the
tool-calling benchmarks work). Every benchmark then talks to that one server:

| Benchmark | Harness | How it reaches the model |
|---|---|---|
| GPQA, MMLU, AIME, IFEval, Tau2 | inspect-ai + inspect_evals | inspect `sglang/` provider via `SGLANG_BASE_URL` |
| IFBench | custom (`eval_ifbench.py`) | `/v1/chat/completions` |
| Search | custom (`eval_search.py`) | `/generate` (raw, multi-turn) + retrieval server :8500 |
| Browse-Comp-Plus | upstream repo clients | `qwen_client.py --model-server` + BM25 MCP server :8600 |

Port map: eval model **8400**, search retriever **8500**, BCP MCP **8600**
(chosen to avoid 8000/8098/8265/13141/30000/30001/30011/30012 already in use).

## 2. File inventory

```
multi/eval/
├── run_all.sh                    # master entry point (this is "the script")
├── env.sh                        # every knob, env-overridable
├── serve_model.sh                # sglang server for the checkpoint + health wait
├── .venv_inspect/                # dedicated venv: inspect-ai 0.3.200 + inspect_evals
├── benchmarks/
│   ├── inspect/run_inspect.sh    # gpqa | mmlu | aime | ifeval | tau2
│   ├── ifbench/eval_ifbench.py   # queries model, scores w/ slime's official scorer
│   ├── ifbench/run_ifbench.sh
│   ├── search/eval_search.py     # Search-R1 multi-turn loop + EM scoring
│   ├── search/qa_em_format.py    # copied from slime/examples/search-r1 (EM scorer)
│   ├── search/retrieval_server.py# copied from slime (patched: --host/--port args)
│   ├── search/launch_retriever.sh
│   ├── search/run_search.sh
│   └── browsecomp_plus/
│       ├── BrowseComp-Plus/      # git clone github.com/texttron/BrowseComp-Plus
│       └── run_browsecomp.sh     # MCP server -> agent -> LLM judge
├── results/<ckpt>/               # ifbench.json, search_*.json, inspect_logs/, bcp evals
└── logs/<ckpt>/                  # per-benchmark logs, server logs, pids
```

## 3. Per-benchmark: source, config, scoring

### GPQA (`run_all.sh gpqa`)
- **Source:** `inspect_evals/gpqa_diamond` — UK AISI inspect_evals (local checkout
  `slime/eval/inspect_evals`, pkg 0.3.104), dataset Idavidrein/gpqa GPQA-Diamond,
  198 multiple-choice science questions.
- **Config:** CoT (`-T cot=true`), temperature 0, max-tokens 32000,
  max-connections 35, 1 epoch — identical to the previous runs in
  `slime/eval/script/run.sh`.
- **Metric:** choice accuracy.

### MMLU (`run_all.sh mmlu`)
- **Source:** `inspect_evals/mmlu_5_shot` (hendrycks MMLU via HF cais/mmlu),
  14,042 questions, 5-shot.
- **Config:** CoT, temperature **0.7** (matches previous runs), max-tokens 32000.
- **Metric:** accuracy.

### AIME (`run_all.sh aime`)
- **Source:** `inspect_evals/aime2025` (HF math-ai/aime25), 30 problems.
- **Config:** temperature 0, max-tokens 32000 — matches previous runs.
- **Metric:** exact-answer accuracy.

### IFEval (`run_all.sh ifeval`)
- **Source:** `inspect_evals/ifeval` (google/IFEval, 541 prompts; official
  google-research checkers via the `instruction_following_eval` package, installed
  in `.venv_inspect`).
- **Config:** temperature 0, max-tokens 32000.
- **Metric:** prompt/instruction-level strict + loose accuracy (inspect reports all 4).

### Tau / Tau2 (`run_all.sh tau2`)
- **Source:** `inspect_evals/tau2_{retail,airline,telecom}` — inspect-native port
  of sierra-research/tau2-bench with vendored data (no external tau2 package).
  Same harness used for the previous tau2 runs in `slime/eval/logs/`.
- **Config:** temperature 0, max-tokens 32000, `message_limit=50` per sample
  (`TAU2_MESSAGE_LIMIT`); the evaluated model also plays the user simulator
  (self-play) — pass `-T user_model=...` for a fixed simulator. `TAU2_LIMIT=5`
  for smoke tests. **Warning:** token-hungry (tens of M tokens for a full run).
- **Metric:** per-domain task pass rate.

### IFBench (`run_all.sh ifbench`)
- **Source:** the *same eval set and scorer used as eval reward during RL
  training*: `slime/examples/ifbench/IFBench_eval.jsonl` (300 prompts,
  allenai/IFBench held-out instruction set) scored by
  `slime/rollout/rm_hub/ifbench.py::compute_ifbench_reward`, which wraps the
  official IFBench checkers (vendored at `/xuanwu-tank/north/xw27/multi/IFBench`).
  The scorer module is loaded by file path to avoid slime's `ray` import chain.
- **Config:** temperature 0, max_tokens 8192 (matches `--eval-max-response-len`
  in the RL scripts), concurrency 32. Empty-`content` responses fall back to
  `reasoning_content` (matters for thinking models behind a reasoning parser).
- **Metric:** strict prompt-level accuracy (all instructions followed) →
  `results/<ckpt>/ifbench.json`.

### Search (`run_all.sh search`)
- **Source:** Search-R1 (PeterGriffinJin/Search-R1) test sets — the canonical
  7-dataset QA suite from `scripts/data_process/qa_search_test_merge.py`
  (`RUC-NLPIR/FlashRAG_datasets`):
  - **`SEARCH_DATASET=full` (default):** `/data1/whx/Search-R1/data/nq_hotpotqa_train/test.parquet`
    — nq, triviaqa, popqa (single-hop) + hotpotqa, 2wikimultihopqa, musique,
    bamboogle (multi-hop); 51,713 q total, subsampled `SEARCH_N_PER_DS=500`/dataset
    (~3,125 q). Reports **per-dataset EM** in the output JSON's `per_dataset`.
  - `SEARCH_DATASET=nq`: `/xuanwu-tank/center/whx/Search_data/test.parquet`
    — 3,610 NQ-only, the exact eval signal from whx's search RL training.
  - The full suite is driven for all checkpoints by `rerun_search_all.sh`
    (`MODELS="base math search tau" GPU=<id>`); `run_all.sh` uses it by default.
- **Rollout:** `eval_search.py` faithfully replicates
  `slime/examples/search-r1/generate_with_search.py`: `max_turns=2` generation
  rounds; `<search>q</search>` → dense retrieval (top-3) → `<information>` block;
  `<answer>` terminates; invalid actions get the standard retry message.
- **Retriever:** `retrieval_server.py` (copied from slime; **three patches**:
  (1) `--host/--port` args, default :8500; (2) CPU fallback — upstream
  unconditionally called `model.cuda()` for the e5 query encoder and crashed
  without a GPU; (3) retrieval serialized behind a lock — concurrent
  `/retrieve` calls segfaulted via a torch/faiss OpenMP clash in FastAPI's
  threadpool; verified fixed with a 6-way concurrent stress test).
  Serves **e5-base-v2 + faiss Flat over wiki-18** (21M passages) from
  `/xuanwu-tank/center/whx/Search_data/` (61 GB index, ~65 GB RAM in CPU
  mode; `RETRIEVER_GPU=<id>` for faiss-gpu).
- **Metric:** Search-R1 exact match (`qa_em_format.compute_score_em`,
  `format_score=0` → pure EM on the `<answer>` content), overall + per dataset →
  `results/<ckpt>/search_<dataset>.json`.

### Browse-Comp-Plus (`run_all.sh browsecomp`)
- **Source:** `git clone https://github.com/texttron/BrowseComp-Plus`
  (commit pinned in the clone; arXiv 2508.06600). Queries: 830, decrypted from
  HF `Tevatron/browsecomp-plus` → `data/browsecomp_plus_decrypted.jsonl`.
  Retrieval corpus: fixed ~100K-doc corpus, **BM25 prebuilt index** downloaded
  from `Tevatron/browsecomp-plus-indexes`.
- **Environment:** own `uv`-managed venv (`BrowseComp-Plus/.venv`, python 3.10)
  per upstream `uv.lock`; pyserini/BM25 needs a JDK — a portable **Temurin
  JDK 21** was installed at `/xuanwu-tank/north/xw27/jdk/jdk-21.0.11+10`
  (exported as `JAVA_HOME` in `env.sh`; system had JRE only).
- **Pipeline** (per upstream `docs/qwen.md`, vllm swapped for our sglang server):
  1. `searcher/mcp_server.py --searcher-type bm25` (MCP at `:8600/mcp`);
  2. `search_agent/qwen_client.py` agent loop (tool-calling; auto-resume per query);
  3. judge: `scripts_evaluation/evaluate_with_openai.py` via `OPENAI_BASE_URL`.
- **Config:** `BCP_MAX_TOKENS=10000` (upstream default), `BCP_LIMIT` for smoke
  tests. **Judge note:** the leaderboard judge is Qwen3-32B served via vllm
  (`evaluate_run.py`); for self-containment we default the judge to the
  evaluated model's endpoint — set `BCP_JUDGE_BASE_URL/_MODEL` to a Qwen3-32B
  server for leaderboard-comparable numbers.
- **Metric:** judge accuracy + citation precision/recall + recall of evidence
  docs → `results/<ckpt>/browsecomp_plus_evals/` and `evaluation_summary.json`
  in the run dir.

## 3b. Repeated runs (3×) + trajectories

For variance estimates, each model is evaluated **3 times** into
`results/<model>/run{1,2,3}/`, driven by `run_model_3runs.sh` (one process per
model, pinned to its GPU/server; idempotent — skips any benchmark whose output
already exists, so it fills gaps and resumes safely). `aggregate.py` prints a
per-benchmark **mean ± std** table across the runs.

- **GPU pinning:** Math→GPU0:30011, Search→GPU1:30012, Tau→GPU3:8400 (reusing
  the live servers via `run_inspect_api.py`, which sends
  `separate_reasoning=false` so a server launched with `--reasoning-parser`
  doesn't blind the graders). Base is run separately when a GPU frees.
- **MMLU policy:** `cot=true` always (never change without instruction —
  `memory/mmlu-cot-true.md`). Run **3×** for Base/Math (they terminate fast) but
  **1×** for Search/Tau (they ramble to the 16k cap → ~55h/run); MMLU is
  scheduled **last** so it doesn't block the cheap benchmarks.
- **Full datasets:** search = all 51,713 (no subsample, `SEARCH_N_PER_DS=0`);
  every other benchmark already runs its full set.
- **Token cap = 30,000** for gpqa/aime/ifeval/ifbench (raised from 16,384). The
  RL'd checkpoints (Search/Tau) don't terminate and were truncating at 16k on
  these benchmarks (AIME 90–93%, Tau IFEval 94%), suppressing scores. 30k (not a
  literal 32,768) leaves ~2.7k headroom under the model's 32,768 context — a
  literal 32k cap would overflow every short-prompt sample and score it
  model_length. tau2 stays at 16k/turn (per-turn cap; a 30k turn-cap overflows
  the growing multi-turn context) and MMLU is unchanged (its cot=true rambling
  is a separate issue).
- **Trajectories:** IFBench and Search now persist full prompt+response (search
  keeps the multi-turn `<search>/<information>/<answer>` string) to
  `results/<model>/run*/{ifbench,search_full}_trajectories.jsonl`. inspect
  tasks (`*.eval`) and BrowseComp (`runs/bm25/<model>_run*/run_*.json`) already
  stored full trajectories.

## 4. Environment notes / decisions

- **Dedicated inspect venv** (`.venv_inspect`): inspect's sglang provider needs
  `openai>=2.26.0`, but `sglang_env` hard-pins `openai==2.6.1` (sglang 0.5.10).
  An upgrade attempt was rolled back; inspect now runs isolated with
  inspect-ai 0.3.200 + the local inspect_evals checkout (editable, with
  `[ifeval,math]` extras).
- **sglang_env additions:** `faiss-cpu` (retrieval server) and the IFBench
  checker deps (`emoji syllapy langdetect immutabledict nltk absl-py spacy
  unicodedata2` per `/xuanwu-tank/north/xw27/multi/IFBench/requirements.txt`).
- **GPU defaults:** `EVAL_GPUS=2,3` (the GPUs we control on this box). On a free
  node use `EVAL_GPUS=0,1,2,3` etc.; TP follows the GPU count.

## 5. Verification performed (smoke tests, all passed)

| Component | Test | Result |
|---|---|---|
| IFBench evaluator | 2 prompts vs live Qwen3-30B-A3B-Base server | scored end-to-end, JSON written |
| inspect + sglang provider | `gpqa_diamond --limit 1` vs live server | completed, .eval log written |
| BCP setup | uv sync, BM25 index (15 files), decrypt 830/830 | OK |
| BCP MCP server | BM25 + Temurin JDK, `GET /mcp` | HTTP 200 |
| BCP agent | 1 query through qwen_client vs live server | run JSON saved |
| BCP judge | evaluate_with_openai vs live server as judge | summary + CSV written |
| Search retriever | 61G index load, /retrieve query, 6-way concurrency stress | all 200, server stable |
| Search eval loop | 3 NQ questions end-to-end (rollout → retrieval → EM scoring) | completed; retrieval confirmed exercised |
| All shell/python files | `bash -n` / `py_compile` | clean |

Caveats observed during smoke tests (informational, not failures):
- A **base** (non-instruct) checkpoint scores ~0 on tool-dependent benchmarks
  (tau2, BCP tool use) because it doesn't emit `<tool_call>` format — that's a
  model property, not a harness issue.
- For thinking models that never emit `</think>`, chat-endpoint `content` can be
  empty; the IFBench evaluator falls back to `reasoning_content`, and inspect
  handles reasoning content natively.

## 6. Quick reference

```bash
cd /xuanwu-tank/north/xw27/multi/eval

# Full sweep on the 8B RL checkpoint (example)
CKPT=/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420 \
EVAL_GPUS=2,3 bash run_all.sh

# Smoke-test everything cheaply
CKPT=... TAU2_LIMIT=2 BCP_LIMIT=3 SEARCH_NQ_LIMIT=20 bash run_all.sh

# Individual benchmark
CKPT=... bash run_all.sh ifbench
```
