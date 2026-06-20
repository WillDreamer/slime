# `eval_scai` — unified evaluation harness

Evaluate one HF checkpoint across **8 benchmarks** against an sglang server:

> **GPQA · MMLU · AIME · IFEval · Tau2 · IFBench · Search (Search-R1) · BrowseComp-Plus**

This is an adaptation of `/xuanwu-tank/north/xw27/multi/eval` for the
`slime_scai` tree. Only the harness **code/scripts** were copied; the large
assets (BrowseComp-Plus corpus/index ~9.5G, the `.venv_inspect`, prior
`results/`) are **reused from the original `eval/` dir**, not duplicated — see
[§6](#6-what-was-adapted--reused).

---

## 1. Quick start

```bash
cd /xuanwu-tank/north/xw27/multi/slime_scai/eval_scai

# all 8 benchmarks on a checkpoint:
CKPT=/path/to/hf_checkpoint bash run_all.sh

# a subset:
CKPT=/path/to/hf_checkpoint bash run_all.sh gpqa aime ifbench

# common overrides:
CKPT=... EVAL_GPUS=2,3 EVAL_PORT=8400 bash run_all.sh
CKPT=... TAU2_LIMIT=5 BCP_LIMIT=20 bash run_all.sh tau2 browsecomp   # smoke test
```

`run_all.sh` (1) serves the checkpoint with sglang, then (2) runs each requested
benchmark, continuing past individual failures, and (3) prints a status summary.
Results land in `results/<ckpt_name>/`, logs in `logs/<ckpt_name>/`.

`CKPT` is **required** (an HF-format checkpoint dir). `CKPT_NAME` defaults to its
basename and is used as the served-model name and the results subdir.

---

## 2. How it fits together

```
run_all.sh                       ← orchestrator (CKPT=… bash run_all.sh [benchmarks])
 ├─ env.sh                       ← all config/knobs (sourced by every script)
 ├─ serve_model.sh               ← launches the sglang server for CKPT, waits for /health
 └─ benchmarks/
     ├─ inspect/run_inspect.sh   ← gpqa | mmlu | aime | ifeval | tau2  (via inspect-ai)
     ├─ ifbench/run_ifbench.sh   ← IFBench (custom; uses slime_scai's checkers)
     ├─ search/run_search.sh     ← Search-R1 EM (needs a retriever server)
     └─ browsecomp_plus/run_browsecomp.sh  ← BrowseComp-Plus (BM25 MCP + agent + judge)
aggregate.py                     ← run1/run2/run3 → mean ± std table
```

All benchmarks talk to the **one** sglang server `serve_model.sh` starts
(OpenAI-compatible at `http://EVAL_HOST:EVAL_PORT/v1`, served model = `CKPT_NAME`).

---

## 3. Prerequisites (external — not in this dir)

| Need | Default location | Used by |
|---|---|---|
| **sglang python** (`PYBIN`) | `/xuanwu-tank/north/xw27/envs/sglang_env/bin/python` | serve, ifbench, search |
| **inspect venv** (`INSPECT_BIN`) | `…/multi/eval/.venv_inspect/bin/inspect` *(reused)* | gpqa/mmlu/aime/ifeval/tau2 |
| **JDK 21** (`JAVA_HOME`) | `…/xw27/jdk/jdk-21.0.11+10` | BrowseComp-Plus BM25 (pyserini) |
| **BrowseComp-Plus repo** (`BCP_ROOT`) | `…/multi/eval/benchmarks/browsecomp_plus/BrowseComp-Plus` *(reused, ~9.5G)* | browsecomp |
| **Retriever index/corpus** | `…/center/whx/Search_data/{e5_Flat.index, wiki-18.jsonl}` | search |
| **Search eval data** | `…/center/whx/Search_data/test.parquet` (nq) · `/data1/whx/Search-R1/.../test.parquet` (full) | search |
| **IFBench eval data** | `…/multi/slime_scai/examples/ifbench/IFBench_eval.jsonl` *(in slime_scai)* | ifbench |
| **GPUs** | `EVAL_GPUS` (default `2,3`); TP = #GPUs | serve |

> Every path above is an `env.sh` variable and can be overridden by exporting it
> before the run.

---

## 4. Running benchmarks individually

First serve the model once (subsequent runs reuse it if healthy):
```bash
CKPT=/path/to/hf_checkpoint bash serve_model.sh
```

Then any of:
```bash
# inspect-ai benchmarks
CKPT=… bash benchmarks/inspect/run_inspect.sh gpqa
CKPT=… bash benchmarks/inspect/run_inspect.sh mmlu
CKPT=… bash benchmarks/inspect/run_inspect.sh tau2     # uses TAU2_DOMAINS

# IFBench (rule-based instruction following)
CKPT=… bash benchmarks/ifbench/run_ifbench.sh

# Search-R1 (auto-starts the retriever via launch_retriever.sh)
CKPT=… SEARCH_DATASET=nq bash benchmarks/search/run_search.sh     # 3,610 NQ q
CKPT=… SEARCH_DATASET=full bash benchmarks/search/run_search.sh   # 7-dataset suite

# BrowseComp-Plus (BM25 MCP server → agent loop → LLM judge)
CKPT=… BCP_LIMIT=20 bash benchmarks/browsecomp_plus/run_browsecomp.sh
```

Notes:
- **inspect** reaches the model through inspect's sglang provider pointed at the
  already-running server (`SGLANG_BASE_URL`), and writes `.eval` logs to
  `results/<ckpt>/inspect_logs/` (view with `inspect view --log-dir …`).
- **ifbench** scores with the strict IFBench checker imported from slime_scai
  (`--slime-root` → `slime_scai`), data from `slime_scai/examples/ifbench/`.
- **search** needs the retriever on port `RETRIEVER_PORT` (8500); set
  `RETRIEVER_GPU=<id>` to use the GPU FAISS index.
- **browsecomp** requires `BCP_ROOT` to have a built BM25 index + decrypted
  dataset + its `.venv` (present in the reused original checkout).

---

## 5. Multi-run aggregation (mean ± std)

`aggregate.py` expects `results/<model>/run1/`, `run2/`, `run3/` subdirs:
```bash
# run the model 3× into run1/run2/run3 (see run_model_3runs.sh for the pattern)
CKPT=… bash run_model_3runs.sh

# aggregate into a per-benchmark mean ± std table
/xuanwu-tank/north/xw27/multi/eval/.venv_inspect/bin/python aggregate.py results
```
`aggregate.py` reads inspect `.eval` logs + `ifbench.json` + `search_full.json` +
BrowseComp `evaluation_summary.json`.

The other `run_*.sh` at the top level are operator convenience wrappers for
specific past scenarios (cheap subsets, specific ports, tau-then-base chains);
their paths have been rewritten to `eval_scai`, but `run_all.sh` is the canonical
entry point.

---

## 6. What was adapted / reused

Copied from `…/multi/eval` (code/scripts only, ~338K):
- `env.sh`, `serve_model.sh`, `run_all.sh`, `aggregate.py`, all `run_*.sh`
- `benchmarks/{inspect,ifbench,search}/` (full)
- `benchmarks/browsecomp_plus/run_browsecomp.sh` (wrapper only)

**Path rewrites applied** (`env.sh` + convenience scripts):
- `EVAL_ROOT` → this dir (`…/slime_scai/eval_scai`)
- `SLIME_ROOT` → `…/slime_scai` (so IFBench data/checkers come from slime_scai)
- per-script hardcoded `/multi/eval`, `/multi/slime` refs → the `slime_scai` equivalents

**Deliberately NOT copied (large) — reused from the original `eval/` via absolute paths:**
- `.venv_inspect/` → `INSPECT_BIN` points at `…/multi/eval/.venv_inspect`
- `benchmarks/browsecomp_plus/BrowseComp-Plus/` (9.5G corpus/index/venv) →
  `BCP_ROOT` points at `…/multi/eval/benchmarks/browsecomp_plus/BrowseComp-Plus`
- prior `results/`, `logs/` (fresh ones are created under `eval_scai/`)

To make `eval_scai` fully standalone, recreate `.venv_inspect` here and clone +
index BrowseComp-Plus under this dir, then override `INSPECT_BIN` / `BCP_ROOT`.

`REPORT.md` is the historical results report from the original run (its paths
point at the old location, kept as-is for reference).

---

## 7. Key config knobs (`env.sh`)

| Var | Default | Meaning |
|---|---|---|
| `CKPT` | *(required)* | HF checkpoint dir to evaluate |
| `EVAL_GPUS` / `EVAL_PORT` | `2,3` / `8400` | served-model GPUs (TP=#GPUs) and port |
| `EVAL_MEM_FRACTION` | `0.85` | sglang static mem fraction |
| `REASONING_PARSER` | *(off)* | leave off — these ckpts don't emit `</think>`; enabling routes all tokens to reasoning and graders see empty content |
| `INSPECT_MAX_TOKENS` | `16384` | must fit the 32768 context with the prompt |
| `MMLU_TEMPERATURE` | `0.7` | MMLU always `cot=true` |
| `TAU2_LIMIT` / `TAU2_DOMAINS` | – / `retail airline telecom` | tau2 smoke cap / domains |
| `SEARCH_DATASET` | `full` | `nq` (3,610) or `full` (7-dataset) |
| `RETRIEVER_GPU` | *(cpu)* | GPU id for FAISS, else CPU |
| `BCP_LIMIT` / `BCP_JUDGE_*` | all 830 / served model | browsecomp query cap / judge endpoint |

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `Set CKPT=…` error | `CKPT` is required — export the HF checkpoint path |
| server never healthy | check `logs/<ckpt>/eval_server.log`; weight load from `/xuanwu-tank` can take minutes; lower `EVAL_MEM_FRACTION` if OOM |
| inspect benchmark 0 / empty answers | don't set `REASONING_PARSER`; keep `INSPECT_MAX_TOKENS ≤ 16384` |
| search fails | retriever not up — check `RETRIEVER_PORT` 8500 / index+corpus paths |
| browsecomp: "BCP venv missing" / "index missing" | `BCP_ROOT` must be a fully set-up BrowseComp-Plus checkout (the reused original is) |
| `inspect`/aggregate import errors | use the `.venv_inspect` python/inspect, not `PYBIN` |
