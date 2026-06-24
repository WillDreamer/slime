#!/bin/bash
# Launch a fleet of sglang servers (one model per GPU), following serve_model.sh.
# Run INSIDE the `slime` docker container. Tool-call parser enabled (qwen).
# Ports 7000-7004, GPUs 3-7, single-GPU each (tp=1).
set -u

LOG_DIR="/home/ec2-user/slime/eval_scai/logs/fleet"
mkdir -p "${LOG_DIR}"

MEM_FRACTION="0.85"
TOOL_PARSER="qwen"

# gpu  port  model-path                                              served-name
FLEET=(
  "3 7000 Qwen/Qwen3-8B-Base qwen-8b-base"
  "4 7001 willhx/Qwen3-8B-Base-Math Qwen3-8B-Base-Math"
  "5 7002 willhx/Qwen3-8B-Base-Math-SeaSFT-Search Qwen3-8B-Base-Math-SeaSFT-Search"
  "6 7003 willhx/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau"
  "7 7004 willhx/Qwen3-8B-Base-Math-SeaSFT-Search Qwen3-8B-Base-Math-SeaSFT-Search"
)

for spec in "${FLEET[@]}"; do
  read -r GPU PORT MODEL NAME <<< "${spec}"
  if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q 200; then
    echo "[fleet] port ${PORT} already healthy, skipping ${NAME}"
    continue
  fi
  echo "[fleet] launching ${MODEL} on GPU ${GPU} port ${PORT} (served as ${NAME})"
  CUDA_VISIBLE_DEVICES="${GPU}" nohup python -m sglang.launch_server \
      --model-path "${MODEL}" \
      --served-model-name "${NAME}" \
      --host 0.0.0.0 --port "${PORT}" \
      --tp 1 \
      --mem-fraction-static "${MEM_FRACTION}" \
      --tool-call-parser "${TOOL_PARSER}" \
      > "${LOG_DIR}/server_${PORT}.log" 2>&1 &
  echo $! > "${LOG_DIR}/server_${PORT}.pid"
done

echo "[fleet] all launch commands issued; logs in ${LOG_DIR}"
