#!/bin/bash
# ============================================================================
# Shared configuration for the unified eval harness.
# Sourced by run_all.sh and the per-benchmark run scripts.
# Every knob can be overridden via environment variables before invocation.
# ============================================================================

EVAL_ROOT="/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai"
SLIME_ROOT="/xuanwu-tank/north/xw27/multi/slime_scai"
PYBIN="${PYBIN:-/xuanwu-tank/north/xw27/envs/sglang_env/bin/python}"
# inspect runs from its own venv: inspect's sglang provider needs openai>=2.26
# while sglang_env must keep openai==2.6.1 (sglang 0.5.10 hard pin).
# NOTE: this large venv is NOT copied into eval_scai; we reuse the one in the
# original eval/ dir. Recreate it here and override INSPECT_BIN to go standalone.
INSPECT_BIN="${INSPECT_BIN:-/xuanwu-tank/north/xw27/multi/eval/.venv_inspect/bin/inspect}"
# Portable JDK 21 (Temurin) for pyserini/BM25 in BrowseComp-Plus.
export JAVA_HOME="${JAVA_HOME:-/xuanwu-tank/north/xw27/jdk/jdk-21.0.11+10}"
export PATH="${JAVA_HOME}/bin:${PATH}"

# --- Checkpoint under evaluation (HF format) --------------------------------
# REQUIRED: set CKPT to the HF checkpoint directory of the model to evaluate.
CKPT="${CKPT:?Set CKPT=<path to HF checkpoint>}"
CKPT_NAME="${CKPT_NAME:-$(basename "${CKPT}")}"

# --- Eval-target model server (sglang, OpenAI-compatible) -------------------
EVAL_GPUS="${EVAL_GPUS:-2,3}"               # GPUs for the served model
EVAL_PORT="${EVAL_PORT:-8400}"
EVAL_HOST="${EVAL_HOST:-127.0.0.1}"
EVAL_BASE_URL="http://${EVAL_HOST}:${EVAL_PORT}/v1"
EVAL_MEM_FRACTION="${EVAL_MEM_FRACTION:-0.85}"
# Empty = let sglang derive from the model config (e.g. Qwen3-8B-Base caps at
# 32768; forcing higher makes the server refuse to start). Set explicitly only
# for models with rope-scaling headroom.
EVAL_CONTEXT_LEN="${EVAL_CONTEXT_LEN:-}"
# Tool parser needed for tau2/BCP tool calls (inert otherwise).
# Reasoning parser default OFF: these checkpoints never emit </think>, so the
# qwen3 parser routes ALL tokens to reasoning_content and graders see empty
# content (verified: gpqa scored 0/198, below the 25% guess floor). Without
# the parser, <think> text stays in content and answer extraction works.
REASONING_PARSER="${REASONING_PARSER:-}"
TOOL_PARSER="${TOOL_PARSER:-qwen}"
# TP defaults to the number of GPUs in EVAL_GPUS.
_n_gpus=$(echo "${EVAL_GPUS}" | tr ',' '\n' | wc -l)
EVAL_TP="${EVAL_TP:-${_n_gpus}}"

# --- Results / logs ----------------------------------------------------------
RESULTS_DIR="${RESULTS_DIR:-${EVAL_ROOT}/results/${CKPT_NAME}}"
LOG_DIR="${LOG_DIR:-${EVAL_ROOT}/logs/${CKPT_NAME}}"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# --- inspect-ai benchmarks (GPQA / MMLU / AIME / IFEval / Tau2) --------------
INSPECT_MAX_CONNECTIONS="${INSPECT_MAX_CONNECTIONS:-16}"  # shared servers: be gentle
# 16384, NOT 32000: prompt + completion must fit the model's 32768 context.
# At 32000, mmlu's ~1k-token 5-shot prompts overflowed -> requests rejected
# with model_length -> scored incorrect (mmlu read 26.9% when true acc ~85%).
INSPECT_MAX_TOKENS="${INSPECT_MAX_TOKENS:-16384}"
INSPECT_TIMEOUT="${INSPECT_TIMEOUT:-3600}"   # 32k-token CoT on a busy server can exceed default client timeout
INSPECT_TEMPERATURE="${INSPECT_TEMPERATURE:-0.0}"
MMLU_TEMPERATURE="${MMLU_TEMPERATURE:-0.7}"     # matches previous runs
TAU2_LIMIT="${TAU2_LIMIT:-}"                    # e.g. 5 for smoke tests; empty = full
TAU2_MESSAGE_LIMIT="${TAU2_MESSAGE_LIMIT:-50}"  # cap agent/user turns per sample
TAU2_DOMAINS="${TAU2_DOMAINS:-retail airline telecom}"

# --- IFBench -----------------------------------------------------------------
IFBENCH_DATA="${IFBENCH_DATA:-${SLIME_ROOT}/examples/ifbench/IFBench_eval.jsonl}"
IFBENCH_MAX_TOKENS="${IFBENCH_MAX_TOKENS:-8192}"   # matches training eval len
IFBENCH_TEMPERATURE="${IFBENCH_TEMPERATURE:-0.0}"
IFBENCH_CONCURRENCY="${IFBENCH_CONCURRENCY:-32}"

# --- Search (Search-R1) ------------------------------------------------------
RETRIEVER_PORT="${RETRIEVER_PORT:-8500}"
RETRIEVER_URL="http://127.0.0.1:${RETRIEVER_PORT}/retrieve"
RETRIEVER_INDEX="${RETRIEVER_INDEX:-/xuanwu-tank/center/whx/Search_data/e5_Flat.index}"
RETRIEVER_CORPUS="${RETRIEVER_CORPUS:-/xuanwu-tank/center/whx/Search_data/wiki-18.jsonl}"
RETRIEVER_GPU="${RETRIEVER_GPU:-}"          # set to a GPU id to use --faiss_gpu
# full  : 7-dataset Search-R1 suite (default) — nq, triviaqa, popqa, hotpotqa,
#         2wikimultihopqa, musique, bamboogle (51,713 q), subsampled per
#         SEARCH_N_PER_DS; reports per-dataset EM. This is the canonical
#         Search-R1 evaluation (scripts/data_process/qa_search_test_merge.py).
# nq    : whx's NQ-only set (3,610 q; matches the RL-training eval signal)
SEARCH_DATASET="${SEARCH_DATASET:-full}"
SEARCH_DATA_NQ="${SEARCH_DATA_NQ:-/xuanwu-tank/center/whx/Search_data/test.parquet}"
SEARCH_DATA_FULL="${SEARCH_DATA_FULL:-/data1/whx/Search-R1/data/nq_hotpotqa_train/test.parquet}"
SEARCH_N_PER_DS="${SEARCH_N_PER_DS:-0}"     # 0 = full 51,713 (default); >0 subsamples per dataset
SEARCH_NQ_LIMIT="${SEARCH_NQ_LIMIT:-0}"     # cap for 'nq' set (0 = all 3,610)
SEARCH_MAX_TURNS="${SEARCH_MAX_TURNS:-5}"   # generation rounds, matches training rollout SEARCH_R1_CONFIGS['max_turns']=5
SEARCH_TOPK="${SEARCH_TOPK:-3}"
SEARCH_MAX_NEW_TOKENS="${SEARCH_MAX_NEW_TOKENS:-2048}"
SEARCH_CONCURRENCY="${SEARCH_CONCURRENCY:-64}"

# --- BrowseComp-Plus ----------------------------------------------------------
# The BrowseComp-Plus repo (~9.5G: data/, indexes/, .venv) is NOT copied into
# eval_scai. We point at the original eval/ checkout so the corpus, BM25 index
# and venv are reused. To go standalone, clone+index BCP under eval_scai and
# override BCP_ROOT.
BCP_ROOT="${BCP_ROOT:-/xuanwu-tank/north/xw27/multi/eval/benchmarks/browsecomp_plus/BrowseComp-Plus}"
BCP_VENV="${BCP_ROOT}/.venv"
BCP_MCP_PORT="${BCP_MCP_PORT:-8600}"
BCP_LIMIT="${BCP_LIMIT:-}"                  # empty = all 830 queries
BCP_MAX_TOKENS="${BCP_MAX_TOKENS:-10000}"
# Judge endpoint for evaluate_with_openai.py (defaults to the served model;
# the official leaderboard judge is Qwen3-32B — point these at one if you
# need leaderboard-comparable numbers).
BCP_JUDGE_BASE_URL="${BCP_JUDGE_BASE_URL:-${EVAL_BASE_URL}}"
BCP_JUDGE_MODEL="${BCP_JUDGE_MODEL:-${CKPT_NAME}}"
BCP_JUDGE_API_KEY="${BCP_JUDGE_API_KEY:-dummy}"
