#!/bin/bash
# ============================================================================
# Full 7-dataset Search-R1 EM evaluation across all checkpoints.
# Part of the master harness — the canonical search benchmark (nq, triviaqa,
# popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle), reporting per-dataset
# EM, superseding the earlier NQ-only search numbers.
#
# Usage:
#   MODELS="base math search tau" GPU=3 bash rerun_search_all.sh
#   MODELS=search bash rerun_search_all.sh        # reuse :30012, no GPU needed
#
# For each model: reuse an existing server if mapped (search -> :30012), else
# serve fresh on $GPU:$PORT, run eval_search over the 7-dataset parquet, tear
# the server down. Results -> results/<name>/search_full.json.
# ============================================================================
set -u
EVAL_ROOT="/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai"
PYBIN="/xuanwu-tank/north/xw27/envs/sglang_env/bin/python"
GPU="${GPU:-3}"
PORT="${PORT:-8410}"
NPDS="${SEARCH_N_PER_DS:-500}"      # per-dataset subsample (0 = all 51,713)
CONC="${SEARCH_CONCURRENCY:-48}"
DATA="/data1/whx/Search-R1/data/nq_hotpotqa_train/test.parquet"
RETR="http://127.0.0.1:8500/retrieve"
MODELS="${MODELS:-base math search tau}"

declare -A CKPTS=(
  [base]="/xuanwu-tank/center/whx/Qwen3-8B-Base"
  [math]="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math/hf_iter_0000300"
  [search]="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420"
  [tau]="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau/hf_iter_0000300"
)
declare -A NAMES=(
  [base]="Qwen3-8B-Base"
  [math]="Qwen3-8B-Base-Math"
  [search]="Qwen3-8B-Base-Math-SeaSFT-Search"
  [tau]="Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau"
)
declare -A REUSE=(
  [search]="http://127.0.0.1:30012"   # whx's already-serving search-model server
)

# Retriever must be up (61G index; launch separately if not).
if ! curl -s -m 5 "${RETR}" -H "Content-Type: application/json" \
       -d '{"queries":["ping"],"topk":1,"return_scores":true}' 2>/dev/null | grep -q result; then
    echo "[rerun_search] ERROR: retriever ${RETR} not responding; start benchmarks/search/launch_retriever.sh" >&2
    exit 1
fi

for m in ${MODELS}; do
    ckpt="${CKPTS[$m]}"; name="${NAMES[$m]}"; reuse="${REUSE[$m]:-}"
    mkdir -p "${EVAL_ROOT}/results/${name}" "${EVAL_ROOT}/logs/${name}"
    served=0
    if [ -n "${reuse}" ]; then
        base="${reuse}"
        echo "[rerun_search] ${name}: reusing ${base}"
    else
        echo "[rerun_search] ${name}: serving on GPU${GPU}:${PORT}"
        if ! CKPT="${ckpt}" CKPT_NAME="${name}" EVAL_GPUS="${GPU}" EVAL_PORT="${PORT}" \
             EVAL_MEM_FRACTION=0.85 bash "${EVAL_ROOT}/serve_model.sh"; then
            echo "[rerun_search] ${name}: serve failed, skipping" >&2; continue
        fi
        base="http://127.0.0.1:${PORT}"; served=1
    fi

    echo ">>> [$(date +%H:%M:%S)] search-full ${name}"
    "${PYBIN}" "${EVAL_ROOT}/benchmarks/search/eval_search.py" \
        --base-url "${base}" --retriever-url "${RETR}" --tokenizer "${ckpt}" \
        --data "${DATA}" --output "${EVAL_ROOT}/results/${name}/search_full.json" \
        --trajectory-output "${EVAL_ROOT}/results/${name}/search_full_trajectories.jsonl" \
        --n-per-dataset "${NPDS}" --max-turns 2 --topk 3 \
        --max-new-tokens 2048 --concurrency "${CONC}" \
        2>&1 | tee "${EVAL_ROOT}/logs/${name}/search_full.log"

    if [ "${served}" -eq 1 ]; then
        spid=$(cat "${EVAL_ROOT}/logs/${name}/eval_server.pid" 2>/dev/null || true)
        [ -n "${spid}" ] && kill -9 "${spid}" 2>/dev/null
        sleep 8
    fi
done
echo "############ [$(date +%F' '%T)] search-full re-run COMPLETE for: ${MODELS} ############"
