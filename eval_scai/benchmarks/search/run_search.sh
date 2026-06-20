#!/bin/bash
# Search benchmark wrapper: ensure retriever is up, then run the EM eval.
# Usage: CKPT=<hf ckpt> bash run_search.sh
#   SEARCH_DATASET=nq   -> whx's NQ eval set (3,610 q; matches RL training eval)
#   SEARCH_DATASET=full -> 7-dataset Search-R1 suite, SEARCH_N_PER_DS per dataset
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../env.sh"

bash "${SCRIPT_DIR}/launch_retriever.sh"

if [ "${SEARCH_DATASET}" = "full" ]; then
    DATA="${SEARCH_DATA_FULL}"
    NPDS="${SEARCH_N_PER_DS}"
    LIMIT=0
else
    DATA="${SEARCH_DATA_NQ}"
    NPDS=0
    LIMIT="${SEARCH_NQ_LIMIT}"
fi

"${PYBIN}" "${SCRIPT_DIR}/eval_search.py" \
    --base-url "http://127.0.0.1:${EVAL_PORT}" \
    --retriever-url "${RETRIEVER_URL}" \
    --tokenizer "${CKPT}" \
    --data "${DATA}" \
    --output "${RESULTS_DIR}/search_${SEARCH_DATASET}.json" \
    --trajectory-output "${RESULTS_DIR}/search_${SEARCH_DATASET}_trajectories.jsonl" \
    --max-turns "${SEARCH_MAX_TURNS}" \
    --topk "${SEARCH_TOPK}" \
    --max-new-tokens "${SEARCH_MAX_NEW_TOKENS}" \
    --concurrency "${SEARCH_CONCURRENCY}" \
    --n-per-dataset "${NPDS}" \
    --limit "${LIMIT}" \
    2>&1 | tee "${LOG_DIR}/search.log"
