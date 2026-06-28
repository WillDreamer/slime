# tau3 evaluation (τ³-bench) — adapted into the eval_scai pipeline

## TL;DR: what "tau3" actually is

There is **no separate `tau3-bench` repository.** "τ³-bench" ("tau three") is
**release `v1.0.0` of `github.com/sierra-research/tau2-bench`** — the same repo,
same `tau2` CLI. The checkout already on disk
(`/mnt/old-data3/home/ec2-user/tau2-bench`, version `1.0.0`) *is* tau3; the
earlier `eval_scai/run_tau2_*.sh` runs simply never exercised the tau3-specific
parts.

## tau1 → tau2 → tau3 (what changed)

| | τ-bench (tau1) | τ²-bench (tau2) | τ³-bench (tau3, v1.0.0) |
|---|---|---|---|
| Repo | `JD-ETH/tau-bench` fork (`import tau_bench`) | `sierra-research/tau2-bench` | **same repo**, tagged v1.0.0 |
| Domains | retail, airline | + telecom, mock | **+ `banking_knowledge`** (RAG/knowledge) |
| User sim | scripted/LLM | LLM user simulator + **user tools**, dual-control | same |
| Eval modes | text | text | text **+ voice full-duplex (audio-native)** |
| Tasks | original | reworked | **+75 task fixes (SABER)**; explicit **task splits** `base`/`train`/`test` |
| Metric | pass^k | pass^k | pass^k (unchanged) |
| Install / py | pip / 3.10 | pip / 3.10 | **`uv` / py 3.12–3.13** |

The only differences that matter for our **text + tool** eval of the 4 ckpts:

1. **`banking_knowledge` domain** — knowledge-retrieval customer service. Run it
   with a `--retrieval-config`. We use **`bm25`** (fully **offline**: a single
   `KB_search` tool, no API key, no sandbox, no extra server). Other offline
   options: `no_knowledge`, `full_kb`, `golden_retrieval`, `grep_only`.
   (Online configs — `openai_embeddings`, `qwen_embeddings`, `alltools`,
   `*_reranker` — need `OPENAI_API_KEY`/`OPENROUTER_API_KEY` and/or a sandbox;
   we deliberately avoid them so no extra serving is required.)
2. **`--task-split-name base`** — the 75+ task fixes ship in the task data; `base`
   is the full task set to use for *evaluation* (`train`/`test` are for RL).
   Counts: airline 50, retail 114, telecom 114, banking_knowledge per its set.
3. Voice (`--audio-native`) is **out of scope** here (needs realtime providers).

## Files

| File | Purpose |
|---|---|
| `serve_tau3_fleet.sh` | 4 agent servers (7000-7003) + 4 GLM-4.7-Flash user-sim replicas (7006-7009) |
| `run_tau3.sh` | 4 lanes × {retail, airline, telecom, mock, banking_knowledge}, GLM user-sim, `base` split, auto-resume |
| `aggregate_tau3.py` | pass^1 mean ± std across runs + coverage from `tau3rp_*` sims |

## Serving topology (8× H200, 1 model/GPU)

| GPU | Port | Serves | Parsers |
|---|---|---|---|
| 0 | 7000 | qwen-8b-base | tool-call `qwen` (**no reasoning parser**) |
| 1 | 7001 | Qwen3-8B-Base-Math | `qwen` |
| 2 | 7002 | …SeaSFT-Search | `qwen` |
| 3 | 7003 | …SeaSFT-Search-TauSFT-Tau | `qwen` |
| 4–7 | 7006–7009 | GLM-4.7-Flash (user sim) | tool-call `glm47` + reasoning `glm45` |

Lane→user-sim pairing: base↔7006, math↔7007, search↔7008, tau↔7009.

**Critical serving fixes** (without them every lane scores ~0 — see
`../TAU2_SEARCH_SETUP.md`): agents run **without** a reasoning parser; stop token
`<|im_end|>` (search lane also `</tool_call>` + `TAU2_FIRST_TOOL_CALL=1`);
`TAU2_LENIENT_TOOL_ARGS=1`; `max_tokens=8192`.

## Run

```bash
# inside the `slime` container (host net, 8 GPUs, hf_cache mounted)
bash /mnt/old-data3/home/ec2-user/slime/aws/_setup_tau_deps.sh   # install+patch tau2 CLI
bash serve_tau3_fleet.sh                                          # start 8 servers, wait for /health
bash run_tau3.sh                                                  # 4 lanes in parallel, resumable
python3 aggregate_tau3.py                                         # results

# knobs: RUNS="1 2 3"  NUM_TRIALS=4  DOMAINS="retail airline telecom mock banking_knowledge"
#        SPLIT=base  KB_RETRIEVAL=bm25  CONC=16  MAX_STEPS=200  MAX_TOKENS=8192
```

All outputs land under `/mnt/old-data3`:
sims in `tau2-bench/data/simulations/tau3rp_*` (copied to
`eval_scai/tau3/tau3_rp_results/`), logs in `eval_scai/tau3/logs/`.
