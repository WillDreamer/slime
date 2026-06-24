#!/bin/bash
# ============================================================================
# Run tau1 (training-aligned) + tau2 (inspect) on ONE model, RUNS times each,
# against the already-running servers. Self-contained: does NOT source env.sh
# (that points at a different /xuanwu-tank copy without the tau1 code).
#
#   Usage:  RUNS=3 bash run_tau_both.sh <base|math>
#
#   base -> agent :7000 (qwen-8b-base),        user-sim GLM :7006
#   math -> agent :7001 (Qwen3-8B-Base-Math),  user-sim GLM :7007
#
# Run INSIDE the slime container. tau1 uses the container python (slime +
# tau_bench); tau2 uses the local .venv_inspect (inspect_ai + inspect_evals).
# ============================================================================
set -u
TAG="${1:?usage: run_tau_both.sh <base|math>}"
ROOT=/home/ec2-user/slime/eval_scai
SLIME=/home/ec2-user/slime
VPY="${ROOT}/.venv_inspect/bin/python"     # inspect venv (tau2)
PY=python                                   # container python (tau1)
RUNS="${RUNS:-3}"
TAU1_DOMAINS="${TAU1_DOMAINS:-retail airline}"
TAU2_DOMAINS="${TAU2_DOMAINS:-retail airline telecom}"
TEMP="${TEMP:-0.7}"
MSG_LIMIT="${MSG_LIMIT:-50}"
# tau2-specific: these checkpoints have 32k context, so the standard
# message_limit=50 overflows mid-conversation. Cap turns + per-turn tokens to
# stay in budget (validated: 24 msgs @ 1024 tok completes & scores).
TAU2_MSG_LIMIT="${TAU2_MSG_LIMIT:-24}"
TAU2_MAX_TOKENS="${TAU2_MAX_TOKENS:-1024}"
TAU2_CONC="${TAU2_CONC:-8}"
TAU1_CONC="${TAU1_CONC:-16}"
LIMIT="${LIMIT:-}"                          # optional smoke cap

case "${TAG}" in
  base) APORT=7000; HF=Qwen/Qwen3-8B-Base;          NAME=qwen-8b-base;      GLM=7006 ;;
  math) APORT=7001; HF=willhx/Qwen3-8B-Base-Math;   NAME=Qwen3-8B-Base-Math; GLM=7007 ;;
  *) echo "unknown tag ${TAG}"; exit 2 ;;
esac
GLM_URL="http://127.0.0.1:${GLM}/v1"
LOGD="${ROOT}/logs/tau_both"; mkdir -p "${LOGD}"
log(){ echo "[$(date +%F' '%T)] [${NAME}] $*"; }

for i in $(seq 1 "${RUNS}"); do
  RD="${ROOT}/results/${NAME}/run${i}"; mkdir -p "${RD}/inspect_logs"

  # -------- tau1 (retail, airline) --------
  for env in ${TAU1_DOMAINS}; do
    OUT="${RD}/tau1_${env}.json"
    if [ -f "${OUT}" ] && "${PY}" -c "import json;json.load(open('${OUT}'))['summary']['accuracy']" 2>/dev/null; then
      log "skip tau1 ${env} run${i} (done)"; continue; fi
    EXTRA=(); [ -n "${LIMIT}" ] && EXTRA+=(--limit "${LIMIT}")
    log "tau1 ${env} run${i} -> ${OUT}"
    OPENAI_API_BASE="${GLM_URL}" OPENAI_API_KEY=dummy \
    PYTHONPATH="${SLIME}/examples/tau-bench:${SLIME}" \
    "${PY}" "${ROOT}/benchmarks/tau/run_tau_eval.py" \
      --hf-checkpoint "${HF}" --router-ip 127.0.0.1 --router-port "${APORT}" \
      --env "${env}" --task-split test --temperature "${TEMP}" \
      --concurrency "${TAU1_CONC}" --user-model GLM-4.7-Flash --user-provider openai \
      --output "${OUT}" "${EXTRA[@]}" \
      > "${LOGD}/tau1_${NAME}_${env}_run${i}.log" 2>&1
    log "tau1 ${env} run${i} rc=$?"
  done

  # -------- tau2 (retail, airline, telecom) --------
  for env in ${TAU2_DOMAINS}; do
    if ls "${RD}/inspect_logs/"*tau2*"${env}"*.eval >/dev/null 2>&1 && \
       "${VPY}" -c "import glob,sys
from inspect_ai.log import read_eval_log
ok=any(read_eval_log(f,header_only=True).status=='success' for f in glob.glob('${RD}/inspect_logs/*tau2*${env}*.eval'))
sys.exit(0 if ok else 1)" 2>/dev/null; then
      log "skip tau2 ${env} run${i} (done)"; continue; fi
    EXTRA=(); [ -n "${LIMIT}" ] && EXTRA+=(--limit "${LIMIT}")
    log "tau2 ${env} run${i}"
    "${VPY}" "${ROOT}/benchmarks/inspect/run_tau2_openai.py" \
      --task "inspect_evals/tau2_${env}" \
      --model-name "${NAME}" --base-url "http://127.0.0.1:${APORT}/v1" \
      --user-model-name GLM-4.7-Flash --user-base-url "${GLM_URL}" \
      --log-dir "${RD}/inspect_logs" --temperature "${TEMP}" \
      --message-limit "${TAU2_MSG_LIMIT}" --max-tokens "${TAU2_MAX_TOKENS}" \
      --max-connections "${TAU2_CONC}" \
      "${EXTRA[@]}" \
      > "${LOGD}/tau2_${NAME}_${env}_run${i}.log" 2>&1
    log "tau2 ${env} run${i} rc=$?"
  done
done
log "############ ${NAME}: tau1+tau2 x${RUNS} COMPLETE ############"
