#!/bin/bash
# Launch the sglang server for the checkpoint under evaluation and wait for it.
# Usage:  CKPT=<hf ckpt> bash serve_model.sh
# The server is OpenAI-compatible at ${EVAL_BASE_URL}; served model name is
# ${CKPT_NAME} so benchmark harnesses can reference it by name.

set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

# Already healthy? Reuse it.
if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${EVAL_PORT}/health" 2>/dev/null | grep -q 200; then
    echo "[serve_model] server already healthy on :${EVAL_PORT}, reusing."
    exit 0
fi

echo "[serve_model] launching ${CKPT} on GPUs ${EVAL_GPUS} (tp=${EVAL_TP}) port ${EVAL_PORT}"
CTX_ARGS=()
[ -n "${EVAL_CONTEXT_LEN}" ] && CTX_ARGS=(--context-length "${EVAL_CONTEXT_LEN}")
[ -n "${REASONING_PARSER}" ] && CTX_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" nohup "${PYBIN}" -m sglang.launch_server \
    --model-path "${CKPT}" \
    --served-model-name "${CKPT_NAME}" \
    --host 0.0.0.0 --port "${EVAL_PORT}" \
    --tp "${EVAL_TP}" \
    --mem-fraction-static "${EVAL_MEM_FRACTION}" \
    "${CTX_ARGS[@]}" \
    --tool-call-parser "${TOOL_PARSER}" \
    > "${LOG_DIR}/eval_server.log" 2>&1 &
echo $! > "${LOG_DIR}/eval_server.pid"

# Wait until healthy (weight load can take several minutes from /xuanwu-tank).
for i in $(seq 1 120); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${EVAL_PORT}/health" 2>/dev/null | grep -q 200; then
        echo "[serve_model] ready after ~$((i*15))s"
        exit 0
    fi
    sleep 15
done
echo "[serve_model] ERROR: server did not become healthy; see ${LOG_DIR}/eval_server.log" >&2
exit 1
