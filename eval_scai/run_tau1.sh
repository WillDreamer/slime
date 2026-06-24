#!/bin/bash
# Training-aligned tau1 eval at temperature 0 (requirement: temp0 for all tau).
# base + math, retail + airline, RUNS runs. Uses the 128k sglang servers
# (7000/7001) as agents and GLM (7006/7007) as user-sim (temp 0). Run in slime.
set -u
ROOT=/home/ec2-user/slime/eval_scai
SLIME=/home/ec2-user/slime
PY=python
RUNS="${RUNS:-3}"
TEMP=0
DOMAINS="${DOMAINS:-retail airline}"
CONC="${CONC:-16}"
LOGD="${ROOT}/logs/tau1"; mkdir -p "${LOGD}"
log(){ echo "[$(date +%F' '%T)] $*"; }

# optional $1 = run only the model whose served-name contains this string
ONLY="${1:-}"
# name agent_port hf_ckpt glm_port
SPECS=(
  "qwen-8b-base 7000 Qwen/Qwen3-8B-Base 7006"
  "Qwen3-8B-Base-Math 7001 willhx/Qwen3-8B-Base-Math 7007"
  "Qwen3-8B-Base-Math-SeaSFT-Search 7002 willhx/Qwen3-8B-Base-Math-SeaSFT-Search 7008"
  "Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau 7003 willhx/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau 7006"
)
for spec in "${SPECS[@]}"; do
  read -r NAME APORT HF GLM <<< "$spec"
  [ -n "${ONLY}" ] && [ "${NAME}" != "${ONLY}" ] && continue
  for i in $(seq 1 "${RUNS}"); do
    RD="${ROOT}/results/${NAME}/run${i}"; mkdir -p "${RD}"
    for env in ${DOMAINS}; do
      OUT="${RD}/tau1_${env}.json"
      if [ -f "${OUT}" ] && "${PY}" -c "import json;d=json.load(open('${OUT}'));assert d['summary']['temperature']==0;d['summary']['accuracy']" 2>/dev/null; then
        log "skip tau1 ${NAME} ${env} run${i} (done @temp0)"; continue; fi
      rm -f "${OUT}"   # discard any stale (temp!=0) result
      log "tau1 ${NAME} ${env} run${i} (temp0)"
      OPENAI_API_BASE="http://127.0.0.1:${GLM}/v1" OPENAI_API_KEY=dummy \
      PYTHONPATH="${SLIME}/examples/tau-bench:${SLIME}" \
      "${PY}" "${ROOT}/benchmarks/tau/run_tau_eval.py" \
        --hf-checkpoint "${HF}" --router-ip 127.0.0.1 --router-port "${APORT}" \
        --env "${env}" --task-split test --temperature "${TEMP}" \
        --concurrency "${CONC}" --user-model GLM-4.7-Flash --user-provider openai \
        --output "${OUT}" \
        > "${LOGD}/tau1_${NAME}_${env}_run${i}.log" 2>&1
      log "tau1 ${NAME} ${env} run${i} rc=$?"
    done
  done
done
log "############ tau1 (temp0) x${RUNS} COMPLETE ############"
