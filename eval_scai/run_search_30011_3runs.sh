#!/bin/bash
# ============================================================================
# Search-ONLY, 3 fresh runs for the model served on :30011 (Qwen3-8B-Base-Math).
# Per user request 2026-06-18: keep the existing runs untouched and write these
# fresh full-51k Search-R1 EM runs under a SEPARATE results name.
# Main server :30011, retriever :8500.
#
# Mirrors run_model_3runs.sh search_step exactly (full 51k, max-turns 2,
# topk 3, max-new-tokens 2048, concurrency 32, global retriever flock).
# Idempotent: skips a run whose search_full.json already exists in OUT_NAME.
# ============================================================================
set -u
EVAL_ROOT="/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai"
PYBIN="/xuanwu-tank/north/xw27/envs/sglang_env/bin/python"
MODEL="Qwen3-8B-Base-Math"
OUT_NAME="${OUT_NAME:-Qwen3-8B-Base-Math-search_rerun_20260618}"   # separate from existing runs
CKPT="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math/hf_iter_0000300"
PORT="${PORT:-30011}"
GEN_URL="http://127.0.0.1:${PORT}"
RETR="http://127.0.0.1:8500/retrieve"
SEARCH_DATA="/data1/whx/Search-R1/data/nq_hotpotqa_train/test.parquet"
RUNS="${RUNS:-3}"
LOGD="${EVAL_ROOT}/logs/${OUT_NAME}"; mkdir -p "${LOGD}"
log() { echo "[$(date +%F' '%T)] [${OUT_NAME}/search3] $*"; }

# preflight: main server + retriever must be live
if ! curl -s -m 5 "${GEN_URL}/v1/models" | grep -q '"id"'; then
    log "ERROR: main server ${GEN_URL} not responding"; exit 1
fi
if ! curl -s -m 8 "${RETR}" -H "Content-Type: application/json" \
       -d '{"queries":["ping"],"topk":1,"return_scores":true}' | grep -q result; then
    log "ERROR: retriever ${RETR} not responding"; exit 1
fi

for i in $(seq 1 "${RUNS}"); do
    RD="${EVAL_ROOT}/results/${OUT_NAME}/run${i}"; mkdir -p "${RD}"
    if [ -f "${RD}/search_full.json" ]; then
        log "run${i}: output already exists, skipping -> ${RD}/search_full.json"; continue
    fi
    log "run${i}/${RUNS}: search(full 51k) [waiting on global retriever lock]"
    (
      exec 9>"${EVAL_ROOT}/logs/.search.lock"
      flock 9
      log "run${i}/${RUNS}: search(full 51k) [lock acquired]"
      "${PYBIN}" "${EVAL_ROOT}/benchmarks/search/eval_search.py" \
          --base-url "${GEN_URL}" --retriever-url "${RETR}" --tokenizer "${CKPT}" \
          --data "${SEARCH_DATA}" --output "${RD}/search_full.json" \
          --trajectory-output "${RD}/search_full_trajectories.jsonl" \
          --n-per-dataset 0 --max-turns 2 --topk 3 --max-new-tokens 2048 --concurrency 32 \
          2>&1 | tee -a "${LOGD}/run${i}_search.log"
    )
    if [ -f "${RD}/search_full.json" ]; then
        em=$("${PYBIN}" -c "import json;print(json.load(open('${RD}/search_full.json'))['summary']['em_overall'])" 2>/dev/null || echo '?')
        log "run${i}/${RUNS}: DONE em_overall=${em}"
    else
        log "run${i}/${RUNS}: FAILED (no output written)"
    fi
done
log "############ ${OUT_NAME}: search-only ${RUNS} fresh runs COMPLETE ############"
