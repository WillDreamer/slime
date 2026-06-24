#!/bin/bash
# One-command BrowseComp-Plus on the gpu7 model, inside the `slime` container.
# gpu7 = port 7004 = Qwen3-8B-Base-Math-SeaSFT-Search (tool-call trained).
#
# Two-phase flow (the judge model replaces the agent model on gpu7):
#   1.  bash run_gpu7_browsecomp.sh predict      # 8B generates predictions on :7004
#   2.  bash serve_judge_gpu7.sh                 # swap gpu7 -> Qwen3-32B on :7004
#   3.  bash run_gpu7_browsecomp.sh judge        # score with the judge on :7004
# Smoke test first:  BCP_LIMIT=3 bash run_gpu7_browsecomp.sh predict
set -e
PHASE="${1:-all}"
BCP="/home/ec2-user/slime/eval_scai/benchmarks/browsecomp_plus"
ROOT="${BCP}/BrowseComp-Plus"
JDK="$(ls -d ${ROOT}/jdk/jdk-21* 2>/dev/null | head -1)"

docker exec \
  -e EVAL_ROOT=/home/ec2-user/slime/eval_scai \
  -e SLIME_ROOT=/home/ec2-user/slime \
  -e RESULTS_DIR=/home/ec2-user/slime/eval_scai/results/Qwen3-8B-Base-Math-SeaSFT-Search \
  -e LOG_DIR=/home/ec2-user/slime/eval_scai/logs/Qwen3-8B-Base-Math-SeaSFT-Search \
  -e PYBIN=python \
  -e CKPT=willhx/Qwen3-8B-Base-Math-SeaSFT-Search \
  -e CKPT_NAME=Qwen3-8B-Base-Math-SeaSFT-Search \
  -e EVAL_PORT=7004 \
  -e BCP_ROOT="${ROOT}" \
  -e BCP_VENV="${ROOT}/.venv" \
  -e JAVA_HOME="${JDK}" \
  -e BCP_MODEL_SERVER=http://127.0.0.1:7004 \
  -e BCP_PHASE="${PHASE}" \
  -e BCP_JUDGE_BASE_URL="${BCP_JUDGE_BASE_URL:-http://127.0.0.1:7004/v1}" \
  -e BCP_JUDGE_MODEL="${BCP_JUDGE_MODEL:-Qwen3-32B}" \
  -e BCP_JUDGE_API_KEY=EMPTY \
  -e BCP_LIMIT="${BCP_LIMIT:-}" \
  slime bash "${BCP}/run_browsecomp_toolcall.sh"
