#!/bin/bash
# Official tau2-bench for the TAU model (final tool-trained checkpoint) only.
# Separate from run_tau2_official.sh so it can run concurrently without editing
# the in-flight matrix. agent=:7003, user-sim=GLM :7008. Same leaderboard-ish
# config as the main matrix (temp0, num_trials=4, 3 runs, conc8, max_steps50,
# max_tokens1024). Resumable. Run in slime.
set -u
TAU2=/home/ec2-user/tau2-bench
LOGD=/home/ec2-user/slime/eval_scai/logs/tau2; mkdir -p "$LOGD"
RUNS="${RUNS:-3}"; NUM_TRIALS="${NUM_TRIALS:-4}"; CONC="${CONC:-16}"
MAX_STEPS="${MAX_STEPS:-200}"; MAX_TOKENS="${MAX_TOKENS:-2048}"
DOMAINS="${DOMAINS:-retail airline telecom}"
NAME=Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau
APORT=7003; GLM=7006
log(){ echo "[$(date +%F' '%T)] [tau2tau] $*"; }
export OPENAI_API_KEY=dummy
cd "$TAU2" || exit 1
AARGS="{\"temperature\":0,\"max_tokens\":${MAX_TOKENS},\"api_base\":\"http://127.0.0.1:${APORT}/v1\"}"
UARGS="{\"temperature\":0,\"max_tokens\":${MAX_TOKENS},\"api_base\":\"http://127.0.0.1:${GLM}/v1\"}"
for i in $(seq 1 "${RUNS}"); do
  for d in ${DOMAINS}; do
    SAVE="tau2off_${NAME}_${d}_run${i}"
    log "tau2 ${NAME} ${d} run${i} (trials=${NUM_TRIALS}, conc=${CONC}) -> ${SAVE}"
    timeout 432000 tau2 run --domain "${d}" \
      --agent-llm "openai/${NAME}" --agent-llm-args "${AARGS}" \
      --user-llm "openai/GLM-4.7-Flash" --user-llm-args "${UARGS}" \
      --num-trials "${NUM_TRIALS}" --max-concurrency "${CONC}" --max-steps "${MAX_STEPS}" \
      --save-to "${SAVE}" > "${LOGD}/${SAVE}.log" 2>&1
    log "tau2 ${NAME} ${d} run${i} rc=$?"
  done
done
log "############ tau2 TAU model (trials=${NUM_TRIALS} x ${RUNS}) COMPLETE ############"
