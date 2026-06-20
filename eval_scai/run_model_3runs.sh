#!/bin/bash
# ============================================================================
# Drive N complete benchmark runs for ONE model on ONE GPU/server.
# Idempotent: each benchmark is skipped if its output for that run already
# exists, so this can be re-launched safely and fills gaps.
#
# Layout: results/<NAME>/run{1,2,3}/{inspect_logs/*.eval, ifbench.json,
#         search_full.json, *_trajectories.jsonl, browsecomp_plus_evals/}
#
# MMLU policy: run in runs 1..MMLU_RUNS, and scheduled LAST (after every run's
# fast benchmarks) so its ~55h crawl on the RL'd checkpoints (search/tau) does
# not block the cheap benchmarks. MMLU is always cot=true (user requirement).
#
# All benchmarks use the existing server at :PORT (reused, never restarted here)
# via run_inspect_api.py (sends separate_reasoning=false so parser-on servers
# don't blind the graders). Full datasets (search = all 51,713).
#
# Required env: KEY (base|math|search|tau)  GPU  PORT  SERVED  [RUNS=3] [MMLU_RUNS]
# ============================================================================
set -u
EVAL_ROOT="/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai"
PYBIN="/xuanwu-tank/north/xw27/envs/sglang_env/bin/python"
IPY="${EVAL_ROOT}/.venv_inspect/bin/python"
RETR="http://127.0.0.1:8500/retrieve"
SEARCH_DATA="/data1/whx/Search-R1/data/nq_hotpotqa_train/test.parquet"

KEY="${KEY:?KEY=base|math|search|tau}"
GPU="${GPU:?GPU=}"; PORT="${PORT:?PORT=}"; SERVED="${SERVED:?SERVED=served-model-name}"
RUNS="${RUNS:-3}"
case "${KEY}" in
  base)   NAME="Qwen3-8B-Base"; CKPT="/xuanwu-tank/center/whx/Qwen3-8B-Base"; MMLU_RUNS="${MMLU_RUNS:-3}";;
  math)   NAME="Qwen3-8B-Base-Math"; CKPT="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math/hf_iter_0000300"; MMLU_RUNS="${MMLU_RUNS:-3}";;
  search) NAME="Qwen3-8B-Base-Math-SeaSFT-Search"; CKPT="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420"; MMLU_RUNS="${MMLU_RUNS:-1}";;
  tau)    NAME="Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau"; CKPT="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau/hf_iter_0000300"; MMLU_RUNS="${MMLU_RUNS:-1}";;
  *) echo "bad KEY ${KEY}" >&2; exit 1;;
esac
BASE_URL="http://127.0.0.1:${PORT}/v1"
GEN_URL="http://127.0.0.1:${PORT}"
LOGD="${EVAL_ROOT}/logs/${NAME}"; mkdir -p "${LOGD}"
log() { echo "[$(date +%F' '%T)] [${NAME}] $*"; }

# --- benchmark steps (each idempotent: skip if output present) ---------------
inspect_task() {  # task_name, run_dir, label, extra args...
    local task="$1" rd="$2" label="$3"; shift 3
    if ls "${rd}/inspect_logs/"*"${label}"*.eval >/dev/null 2>&1; then
        # only skip if a SUCCESS log exists
        if "${IPY}" - "$rd" "$label" <<'PY' 2>/dev/null; then return; fi
import sys, glob
from inspect_ai.log import read_eval_log
rd, lab = sys.argv[1], sys.argv[2]
ok = any(read_eval_log(f, header_only=True).status == "success"
         for f in glob.glob(f"{rd}/inspect_logs/*{lab}*.eval"))
sys.exit(0 if ok else 1)
PY
    fi
    log "inspect ${task} -> ${rd}"
    "${IPY}" "${EVAL_ROOT}/benchmarks/inspect/run_inspect_api.py" \
        --task "${task}" --model-name "${SERVED}" --base-url "${BASE_URL}" \
        --log-dir "${rd}/inspect_logs" "$@" 2>&1 | tee -a "${LOGD}/$(basename "${rd}")_${label}.log"
}

ifbench_step() {  # run_dir
    local rd="$1"; [ -f "${rd}/ifbench.json" ] && return
    log "ifbench -> ${rd}"
    "${PYBIN}" "${EVAL_ROOT}/benchmarks/ifbench/eval_ifbench.py" \
        --base-url "${BASE_URL}" --model "${SERVED}" \
        --data /xuanwu-tank/north/xw27/multi/slime_scai/examples/ifbench/IFBench_eval.jsonl \
        --output "${rd}/ifbench.json" --trajectory-output "${rd}/ifbench_trajectories.jsonl" \
        --slime-root /xuanwu-tank/north/xw27/multi/slime_scai \
        --max-tokens 30000 --temperature 0.0 --concurrency 32 2>&1 | tee -a "${LOGD}/$(basename "${rd}")_ifbench.log"
}

search_step() {  # run_dir
    local rd="$1"; [ -f "${rd}/search_full.json" ] && return
    # The CPU faiss retriever (~3.7 retrievals/s, serialized internally) melts
    # down if multiple drivers search at once. Global flock => one model's
    # full-51k search at a time across all GPUs.
    log "search(full 51k) [waiting on global retriever lock] -> ${rd}"
    exec 9>"${EVAL_ROOT}/logs/.search.lock"
    flock 9
    log "search(full 51k) [lock acquired] -> ${rd}"
    "${PYBIN}" "${EVAL_ROOT}/benchmarks/search/eval_search.py" \
        --base-url "${GEN_URL}" --retriever-url "${RETR}" --tokenizer "${CKPT}" \
        --data "${SEARCH_DATA}" --output "${rd}/search_full.json" \
        --trajectory-output "${rd}/search_full_trajectories.jsonl" \
        --n-per-dataset 0 --max-turns 2 --topk 3 --max-new-tokens 2048 --concurrency 32 \
        2>&1 | tee -a "${LOGD}/$(basename "${rd}")_search.log"
    flock -u 9; exec 9>&-
}

browsecomp_step() {  # run_dir
    local rd="$1"
    [ -f "${rd}/browsecomp_plus_evals/bm25/${NAME}_$(basename "${rd}")/evaluation_summary.json" ] && return
    log "browsecomp -> ${rd}"
    CKPT="${CKPT}" CKPT_NAME="${NAME}_$(basename "${rd}")" RESULTS_DIR="${rd}" \
        EVAL_PORT="${PORT}" EVAL_HOST="127.0.0.1" LOG_DIR="${LOGD}" \
        bash "${EVAL_ROOT}/benchmarks/browsecomp_plus/run_browsecomp.sh" \
        2>&1 | tee -a "${LOGD}/$(basename "${rd}")_browsecomp.log"
}

# --- Phase A: fast benchmarks for every run ----------------------------------
for i in $(seq 1 "${RUNS}"); do
    RD="${EVAL_ROOT}/results/${NAME}/run${i}"; mkdir -p "${RD}/inspect_logs"
    log "=== run ${i}/${RUNS}: fast benchmarks ==="
    # 30000-token cap (not 32768): generation + prompt must fit the model's 32768
    # context, so we leave ~2.7k headroom — a literal 32768 cap would overflow
    # every short-prompt sample and score it model_length.
    inspect_task inspect_evals/gpqa_diamond "${RD}" gpqa-diamond --max-tokens 30000 --temperature 0 --task-arg cot=true
    inspect_task inspect_evals/aime2024     "${RD}" aime2024     --max-tokens 30000 --temperature 0
    inspect_task inspect_evals/aime2025     "${RD}" aime2025     --max-tokens 30000 --temperature 0
    inspect_task inspect_evals/ifeval       "${RD}" ifeval       --max-tokens 30000 --temperature 0
    ifbench_step "${RD}"
    search_step  "${RD}"
    browsecomp_step "${RD}"
    inspect_task inspect_evals/tau2_retail  "${RD}" tau2-retail  --temperature 0 --message-limit 50
    inspect_task inspect_evals/tau2_airline "${RD}" tau2-airline --temperature 0 --message-limit 50
    inspect_task inspect_evals/tau2_telecom "${RD}" tau2-telecom --temperature 0 --message-limit 50
done

# --- Phase B: MMLU (cot=true), runs 1..MMLU_RUNS, scheduled last --------------
for i in $(seq 1 "${MMLU_RUNS}"); do
    RD="${EVAL_ROOT}/results/${NAME}/run${i}"; mkdir -p "${RD}/inspect_logs"
    log "=== run ${i}: MMLU (cot=true) ==="
    inspect_task inspect_evals/mmlu_5_shot "${RD}" mmlu-5-shot --temperature 0.7 --task-arg cot=true
done

log "############ ${NAME}: ${RUNS} runs COMPLETE (mmlu x${MMLU_RUNS}) ############"
