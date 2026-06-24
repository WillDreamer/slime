#!/bin/bash
# Official tau2-bench (sierra-research) eval, leaderboard config, GLM user-sim.
#   - agent temp 0, user (GLM-4.7-Flash) temp 0
#   - max_steps 200, max_errors 10 (official defaults)
#   - num_trials=4 (official pass^k) + 3 outer runs (variance)
#   - max_concurrency 16
#   - 128k-context sglang agents (7000/7001/7002)
#
# Resumable: tau2 auto-resumes an existing --save-to. Run in the slime container
# (tau2-bench installed there); servers live in slime-srv (host networking).
#   Usage:  [ONLY=<name-substr>] RUNS=3 NUM_TRIALS=4 bash run_tau2_official.sh
set -u
TAU2=/home/ec2-user/tau2-bench
LOGD=/home/ec2-user/slime/eval_scai/logs/tau2; mkdir -p "$LOGD"
RUNS="${RUNS:-3}"
NUM_TRIALS="${NUM_TRIALS:-4}"
CONC="${CONC:-16}"               # 32k KV (native) sustains higher concurrency
MAX_STEPS="${MAX_STEPS:-200}"     # official leaderboard value
MAX_TOKENS="${MAX_TOKENS:-8192}" # raised from 2048: long-CoT model was truncating mid-reasoning before emitting <tool_call> (see TAU2_FORMAT_ISSUE.md)
DOMAINS="${DOMAINS:-retail airline telecom}"
ONLY="${ONLY:-}"
log(){ echo "[$(date +%F' '%T)] $*"; }

# served-name  agent_port  glm_user_port
SPECS=(
  "qwen-8b-base 7000 7006"
  "Qwen3-8B-Base-Math 7001 7007"
  "Qwen3-8B-Base-Math-SeaSFT-Search 7002 7008"
)
export OPENAI_API_KEY=dummy
cd "$TAU2" || exit 1

for spec in "${SPECS[@]}"; do
  read -r NAME APORT GLM <<< "$spec"
  [ -n "${ONLY}" ] && [ "${NAME}" != "${ONLY}" ] && continue
  AARGS="{\"temperature\":0,\"max_tokens\":${MAX_TOKENS},\"api_base\":\"http://127.0.0.1:${APORT}/v1\"}"
  UARGS="{\"temperature\":0,\"max_tokens\":${MAX_TOKENS},\"api_base\":\"http://127.0.0.1:${GLM}/v1\"}"
  for i in $(seq 1 "${RUNS}"); do
    for d in ${DOMAINS}; do
      SAVE="tau2off_${NAME}_${d}_run${i}"
      log "tau2 ${NAME} ${d} run${i} (trials=${NUM_TRIALS}, conc=${CONC}) -> ${SAVE}"
      timeout 432000 tau2 run \
        --domain "${d}" \
        --agent-llm "openai/${NAME}" --agent-llm-args "${AARGS}" \
        --user-llm "openai/GLM-4.7-Flash" --user-llm-args "${UARGS}" \
        --num-trials "${NUM_TRIALS}" --max-concurrency "${CONC}" --max-steps "${MAX_STEPS}" \
        --save-to "${SAVE}" \
        > "${LOGD}/${SAVE}.log" 2>&1
      log "tau2 ${NAME} ${d} run${i} rc=$?"
    done
  done
done
log "############ official tau2 (trials=${NUM_TRIALS} x ${RUNS} runs) COMPLETE ############"
