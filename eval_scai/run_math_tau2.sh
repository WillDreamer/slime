#!/bin/bash
# ============================================================================
# Math model (Qwen3-8B-Base-Math) — tau-bench x3 with GLM-4.7-Flash as the
# user simulator (user request 2026-06-16). Runs BOTH:
#   * tau2 = tau2-bench via inspect-ai (retail/airline/telecom)
#   * tau1 = training-aligned tau-bench v1 (retail/airline), reusing
#            generate_with_tau.generate so the number matches training rollout.
#
#   Agent  = Qwen3-8B-Base-Math @ :30011  (whx server, tool-call-parser qwen)
#   User   = GLM-4.7-Flash      @ :8401   (our server on GPU 3, role="user")
#
# Self-play is broken when agent==user (echo loop). A distinct capable user
# simulator (GLM) makes the eval meaningful. Idempotent: skip completed runs.
# ============================================================================
set -u
EVAL_ROOT="/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai"
SLIME_ROOT="$(cd -- "${EVAL_ROOT}/.." &>/dev/null && pwd)"
IPY="${EVAL_ROOT}/.venv_inspect/bin/python"
NAME="Qwen3-8B-Base-Math"
SERVED="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math/hf_iter_0000300"
PORT=30011
BASE_URL="http://127.0.0.1:${PORT}/v1"
USER_MODEL="GLM-4.7-Flash"
USER_URL="http://127.0.0.1:8401/v1"
RUNS="${RUNS:-3}"
MSG_LIMIT="${MSG_LIMIT:-50}"
# tau2 agent temperature, aligned with tau1's eval temperature (0.7).
TAU2_TEMPERATURE="${TAU2_TEMPERATURE:-0.7}"
LOGD="${EVAL_ROOT}/logs/${NAME}"; mkdir -p "${LOGD}"
log() { echo "[$(date +%F' '%T)] [${NAME}/tau] $*"; }

# --- tau1 (training-aligned) config ----------------------------------------
# Needs a python with slime + tau_bench (NOT the inspect venv). Override
# TAU1_PYBIN to that interpreter. The user sim is the GLM server above, reached
# via the OpenAI provider; the agent server (:30011) is hit on /generate (raw).
TAU1_PYBIN="${TAU1_PYBIN:-python}"
TAU1_EXAMPLE_DIR="${TAU1_EXAMPLE_DIR:-${SLIME_ROOT}/examples/tau-bench}"
TAU1_DOMAINS="${TAU1_DOMAINS:-retail airline}"
TAU1_USER_MODEL="${TAU1_USER_MODEL:-zai-org/GLM-4.7-Flash}"

# Wait for the GLM user-simulator server to be healthy before starting.
log "waiting for GLM user-sim at ${USER_URL} ..."
for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:8401/health" >/dev/null 2>&1 && { log "GLM user-sim ready"; break; }
    sleep 15
    [ "$i" -eq 240 ] && { log "ERROR: GLM user-sim never became healthy"; exit 1; }
done

tau2_task() {  # task, run_dir, label
    local task="$1" rd="$2" label="$3"
    if ls "${rd}/inspect_logs/"*"${label}"*.eval >/dev/null 2>&1; then
        if "${IPY}" - "$rd" "$label" <<'PY' 2>/dev/null; then log "skip ${label} (success in $(basename "$rd"))"; return; fi
import sys, glob
from inspect_ai.log import read_eval_log
rd, lab = sys.argv[1], sys.argv[2]
ok = any(read_eval_log(f, header_only=True).status == "success"
         for f in glob.glob(f"{rd}/inspect_logs/*{lab}*.eval"))
sys.exit(0 if ok else 1)
PY
    fi
    log "tau2 ${task} -> $(basename "${rd}")  (agent=${NAME}, user=${USER_MODEL}, temp=${TAU2_TEMPERATURE})"
    "${IPY}" "${EVAL_ROOT}/benchmarks/inspect/run_inspect_api.py" \
        --task "${task}" --model-name "${SERVED}" --base-url "${BASE_URL}" \
        --user-model-name "${USER_MODEL}" --user-base-url "${USER_URL}" \
        --log-dir "${rd}/inspect_logs" --temperature "${TAU2_TEMPERATURE}" --message-limit "${MSG_LIMIT}" \
        2>&1 | tee -a "${LOGD}/$(basename "${rd}")_${label}.log"
}

tau1_task() {  # env, run_dir
    local env="$1" rd="$2" out="${2}/tau1_${1}.json"
    if [ -f "${out}" ] && "${TAU1_PYBIN}" -c "import json; json.load(open('${out}'))['summary']['accuracy']" >/dev/null 2>&1; then
        log "skip tau1 ${env} (done in $(basename "${rd}"))"; return
    fi
    log "tau1 ${env} -> $(basename "${rd}")  (agent=${NAME}@:${PORT}, user=${TAU1_USER_MODEL})"
    OPENAI_API_BASE="${USER_URL}" OPENAI_API_KEY=dummy \
    PYTHONPATH="${TAU1_EXAMPLE_DIR}:${SLIME_ROOT}:${PYTHONPATH:-}" \
    "${TAU1_PYBIN}" "${EVAL_ROOT}/benchmarks/tau/run_tau_eval.py" \
        --hf-checkpoint "${SERVED}" --router-ip 127.0.0.1 --router-port "${PORT}" \
        --env "${env}" --task-split test --user-model "${TAU1_USER_MODEL}" \
        --user-provider openai --tau-example-dir "${TAU1_EXAMPLE_DIR}" \
        --output "${out}" \
        2>&1 | tee -a "${LOGD}/$(basename "${rd}")_tau1-${env}.log"
}

for i in $(seq 1 "${RUNS}"); do
    RD="${EVAL_ROOT}/results/${NAME}/run${i}"; mkdir -p "${RD}/inspect_logs"
    log "=== run ${i}/${RUNS}: tau2 retail/airline/telecom + tau1 ${TAU1_DOMAINS} ==="
    tau2_task inspect_evals/tau2_retail  "${RD}" tau2-retail
    tau2_task inspect_evals/tau2_airline "${RD}" tau2-airline
    tau2_task inspect_evals/tau2_telecom "${RD}" tau2-telecom
    for env in ${TAU1_DOMAINS}; do tau1_task "${env}" "${RD}"; done
done
log "############ ${NAME}: tau1+tau2 x${RUNS} with GLM user-sim COMPLETE ############"
