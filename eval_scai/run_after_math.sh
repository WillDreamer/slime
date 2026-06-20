#!/bin/bash
# GPU3 follow-on (runs after the math model's suite finishes):
#   1. wait for math's run_all (orphan pid passed as $1) to exit
#   2. kill math's :8400 server
#   3. Tau checkpoint full 8-benchmark suite on GPU3        (user-requested)
#   4. search model's deferred browsecomp on GPU3           (parser-free server)
set -u
EVAL_ROOT="/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai"
MATH_PID="${1:?usage: run_after_math.sh <math_run_all_pid>}"

echo "[after_math] waiting for math run_all (pid ${MATH_PID})..."
while kill -0 "${MATH_PID}" 2>/dev/null; do sleep 60; done
echo "[after_math] math finished at $(date +%F' '%T)"

SPID=$(cat "${EVAL_ROOT}/logs/Qwen3-8B-Base-Math/eval_server.pid" 2>/dev/null || true)
[ -n "${SPID}" ] && kill -9 "${SPID}" 2>/dev/null
sleep 10

# --- 3. Tau checkpoint: full suite ------------------------------------------
export EVAL_GPUS=3 EVAL_PORT=8400 EVAL_MEM_FRACTION=0.85
export INSPECT_MAX_CONNECTIONS=24 INSPECT_TIMEOUT=7200
TAU_CKPT="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau/hf_iter_0000300"
TAU_NAME="Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau"
echo "############ [$(date +%F' '%T)] ${TAU_NAME}: full suite ############"
CKPT="${TAU_CKPT}" CKPT_NAME="${TAU_NAME}" \
    bash "${EVAL_ROOT}/run_all.sh" gpqa aime ifeval ifbench search mmlu browsecomp tau2
SPID=$(cat "${EVAL_ROOT}/logs/${TAU_NAME}/eval_server.pid" 2>/dev/null || true)
[ -n "${SPID}" ] && kill -9 "${SPID}" 2>/dev/null
sleep 10

# --- 4. search model: deferred browsecomp (needs parser-free server) ---------
S_CKPT="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420"
S_NAME="Qwen3-8B-Base-Math-SeaSFT-Search"
echo "############ [$(date +%F' '%T)] ${S_NAME}: deferred browsecomp ############"
export CKPT="${S_CKPT}" CKPT_NAME="${S_NAME}"
bash "${EVAL_ROOT}/serve_model.sh" && \
    bash "${EVAL_ROOT}/benchmarks/browsecomp_plus/run_browsecomp.sh"
SPID=$(cat "${EVAL_ROOT}/logs/${S_NAME}/eval_server.pid" 2>/dev/null || true)
[ -n "${SPID}" ] && kill -9 "${SPID}" 2>/dev/null

echo "############ [$(date +%F' '%T)] GPU3 PIPELINE COMPLETE ############"
