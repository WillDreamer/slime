#!/bin/bash
# ============================================================================
# tau-bench (v1) eval — training-aligned, reuses generate_with_tau.generate.
#
# Runs the SAME rollout function slime trains/evals on, against the eval server,
# over the held-out test split. This is the tau1 counterpart to the inspect-ai
# tau2 eval (benchmarks/inspect, BENCH=tau2); run_all.sh runs BOTH.
#
# Output: results/<ckpt>/run<i>/tau1_<env>.json  (read by aggregate.py)
#
# Requires the slime/training env (tau_bench + slime importable). Set TAU1_PYBIN
# to that python. The user simulator is reached via the OpenAI-compatible
# provider, so OPENAI_API_BASE/KEY must point at the GLM user-sim server
# (TAU1_USER_BASE_URL / TAU1_USER_API_KEY below).
# ============================================================================
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../env.sh"

RUNS="${TAU1_RUNS:-3}"
log() { echo "[$(date +%F' '%T)] [${CKPT_NAME}/tau1] $*"; }

# tau_bench reaches the user simulator through the OpenAI provider; point it at
# the GLM user-sim server (distinct from the agent server — self-play loops).
export OPENAI_API_BASE="${TAU1_USER_BASE_URL}"
export OPENAI_API_KEY="${TAU1_USER_API_KEY}"
# training modules (generate_with_tau, trainable_agents, openai_tool_adapter)
# import each other relative to cwd; make them importable.
export PYTHONPATH="${TAU1_EXAMPLE_DIR}:${SLIME_ROOT}:${PYTHONPATH:-}"

# Optionally wait for the GLM user-sim server to be healthy first.
if [ -n "${TAU1_USER_HEALTH_URL}" ]; then
    log "waiting for GLM user-sim at ${TAU1_USER_HEALTH_URL} ..."
    for i in $(seq 1 240); do
        curl -sf "${TAU1_USER_HEALTH_URL}" >/dev/null 2>&1 && { log "user-sim ready"; break; }
        sleep 15
        [ "$i" -eq 240 ] && { log "ERROR: user-sim never became healthy"; exit 1; }
    done
fi

rc_all=0
for i in $(seq 1 "${RUNS}"); do
    RD="${RESULTS_DIR}/run${i}"; mkdir -p "${RD}"
    for env in ${TAU1_DOMAINS}; do
        OUT="${RD}/tau1_${env}.json"
        # Idempotent: skip a run that already produced a summary with accuracy.
        if [ -f "${OUT}" ] && "${TAU1_PYBIN}" -c "import json,sys; json.load(open('${OUT}'))['summary']['accuracy']" >/dev/null 2>&1; then
            log "skip ${env} run${i} (already done: ${OUT})"
            continue
        fi
        log "tau1 ${env} run${i} -> ${OUT} (agent=${CKPT_NAME}@${EVAL_HOST}:${EVAL_PORT}, user=${TAU1_USER_MODEL})"
        EXTRA=()
        [ -n "${TAU1_LIMIT}" ] && EXTRA+=(--limit "${TAU1_LIMIT}")
        # Per-env test-task file (optional): TAU1_TASKS_<ENV> e.g. TAU1_TASKS_RETAIL.
        env_uc="$(echo "${env}" | tr '[:lower:]' '[:upper:]')"
        tasks_var="TAU1_TASKS_${env_uc}"; tasks_file="${!tasks_var:-}"
        [ -n "${tasks_file}" ] && EXTRA+=(--tasks-file "${tasks_file}")
        "${TAU1_PYBIN}" "${SCRIPT_DIR}/run_tau_eval.py" \
            --hf-checkpoint "${CKPT}" \
            --router-ip "${EVAL_HOST}" --router-port "${EVAL_PORT}" \
            --env "${env}" --task-split "${TAU1_SPLIT}" \
            --temperature "${TAU1_TEMPERATURE}" --max-new-tokens "${TAU1_MAX_NEW_TOKENS}" \
            --concurrency "${TAU1_CONCURRENCY}" \
            --user-model "${TAU1_USER_MODEL}" --user-provider "${TAU1_USER_PROVIDER}" \
            --tau-example-dir "${TAU1_EXAMPLE_DIR}" \
            --output "${OUT}" "${EXTRA[@]}" \
            2>&1 | tee -a "${LOG_DIR}/tau1_${env}_run${i}.log"
        rc=${PIPESTATUS[0]}; [ "${rc}" -ne 0 ] && rc_all=${rc}
    done
done
log "############ ${CKPT_NAME}: tau1 x${RUNS} (${TAU1_DOMAINS}) DONE (rc=${rc_all}) ############"
exit "${rc_all}"
