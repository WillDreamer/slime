#!/bin/bash
# Launch the local dense retrieval server (e5 + faiss over wiki-18).
# Index/corpus: /xuanwu-tank/center/whx/Search_data (61G Flat index, 21M passages).
# NOTE: loading the Flat index needs ~65 GB RAM (CPU mode, default) or one GPU
# with --faiss_gpu (set RETRIEVER_GPU=<id>). First run downloads intfloat/e5-base-v2.
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../env.sh"

if curl -s -m 5 "http://127.0.0.1:${RETRIEVER_PORT}/retrieve" -H "Content-Type: application/json" \
     -d '{"queries":["ping"],"topk":1,"return_scores":true}' 2>/dev/null | grep -q result; then
    echo "[retriever] already serving on :${RETRIEVER_PORT}, reusing."
    exit 0
fi

EXTRA=()
ENVS=()
if [ -n "${RETRIEVER_GPU}" ]; then
    EXTRA+=(--faiss_gpu)
    ENVS=(CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU}")
fi

echo "[retriever] launching on :${RETRIEVER_PORT} (gpu='${RETRIEVER_GPU:-cpu}')..."
env "${ENVS[@]}" nohup "${PYBIN}" "${SCRIPT_DIR}/retrieval_server.py" \
    --index_path "${RETRIEVER_INDEX}" \
    --corpus_path "${RETRIEVER_CORPUS}" \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --topk "${SEARCH_TOPK}" \
    --port "${RETRIEVER_PORT}" \
    "${EXTRA[@]}" \
    > "${LOG_DIR}/retriever.log" 2>&1 &
echo $! > "${LOG_DIR}/retriever.pid"

# Index load takes a while (61G from network storage).
for i in $(seq 1 240); do
    if curl -s -m 5 "http://127.0.0.1:${RETRIEVER_PORT}/retrieve" -H "Content-Type: application/json" \
         -d '{"queries":["ping"],"topk":1,"return_scores":true}' 2>/dev/null | grep -q result; then
        echo "[retriever] ready after ~$((i*15))s"
        exit 0
    fi
    sleep 15
done
echo "[retriever] ERROR: not ready; see ${LOG_DIR}/retriever.log" >&2
exit 1
