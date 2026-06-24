#!/bin/bash
# Swap gpu7 (:7004) from the 8B agent model to the BrowseComp-Plus judge model
# (Qwen3-32B, the official leaderboard judge). Run this AFTER predictions are
# written (run_gpu7_browsecomp.sh predict), then run the judge phase.
# A single H200 (143 GB) holds Qwen3-32B bf16 comfortably at tp=1.
set -e
PORT="${JUDGE_PORT:-7004}"
GPU="${JUDGE_GPU:-7}"
MODEL="${JUDGE_MODEL_PATH:-Qwen/Qwen3-32B}"
NAME="${JUDGE_MODEL_NAME:-Qwen3-32B}"
LOG=/home/ec2-user/slime/eval_scai/logs/fleet/judge_${PORT}.log

echo "[judge] freeing :${PORT} on gpu${GPU}"
docker exec slime bash -lc "fuser -k ${PORT}/tcp 2>/dev/null || true"
echo "[judge] launching ${MODEL} on gpu${GPU} :${PORT}"
docker exec -d slime bash -lc "CUDA_VISIBLE_DEVICES=${GPU} nohup python -m sglang.launch_server \
  --model-path ${MODEL} --served-model-name ${NAME} \
  --host 0.0.0.0 --port ${PORT} --tp 1 --mem-fraction-static 0.85 \
  > ${LOG} 2>&1 &"
echo "[judge] waiting for :${PORT}/health (weights download on first run)..."
docker exec slime bash -lc "until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${PORT}/health 2>/dev/null | grep -q 200; do sleep 10; done; echo READY"
echo "[judge] Qwen3-32B serving on :${PORT}. Now run: bash run_gpu7_browsecomp.sh judge"
