#!/bin/bash
# ============================================================================
# Chain: kill GLM user-sim (free GPU3) -> serve TAU model -> run cheap suite
# (gpqa, aime2025, ifeval, ifbench) x3 runs (idempotent: tau run1 already done,
# fills run2 gaps + run3) -> kill TAU server -> serve BASE model -> same cheap
# suite x3 runs -> kill BASE server.
#
# One model on GPU3:8400 at a time. No reasoning parser (env.sh validated
# default: these checkpoints never emit </think>; run_inspect_api sends
# separate_reasoning=false anyway). Idempotent per-benchmark skip-on-success.
# ============================================================================
set -u
EVAL_ROOT="/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai"
IPY="${EVAL_ROOT}/.venv_inspect/bin/python"
PYBIN="/xuanwu-tank/north/xw27/envs/sglang_env/bin/python"
GPU=3; PORT=8400
BASE_URL="http://127.0.0.1:${PORT}/v1"
RUNS="${RUNS:-3}"
DRIVER_LOG="${EVAL_ROOT}/logs/tau_then_base_cheap.log"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "${DRIVER_LOG}"; }

# --- idempotent benchmark steps (mirror run_search_cheap.sh) -----------------
inspect_task() {  # task run_dir label extra...
    local task="$1" rd="$2" label="$3" served="$4"; shift 4
    if ls "${rd}/inspect_logs/"*"${label}"*.eval >/dev/null 2>&1; then
        if "${IPY}" - "$rd" "$label" <<'PY' 2>/dev/null; then log "  skip ${label} (success exists)"; return; fi
import sys, glob
from inspect_ai.log import read_eval_log
rd, lab = sys.argv[1], sys.argv[2]
ok = any(read_eval_log(f, header_only=True).status == "success"
         for f in glob.glob(f"{rd}/inspect_logs/*{lab}*.eval"))
sys.exit(0 if ok else 1)
PY
    fi
    log "  inspect ${task} -> $(basename "${rd}")"
    "${IPY}" "${EVAL_ROOT}/benchmarks/inspect/run_inspect_api.py" \
        --task "${task}" --model-name "${served}" --base-url "${BASE_URL}" \
        --log-dir "${rd}/inspect_logs" "$@" 2>&1 | tee -a "${LOGD}/$(basename "${rd}")_${label}.log"
}

ifbench_step() {  # run_dir served
    local rd="$1" served="$2"; [ -f "${rd}/ifbench.json" ] && { log "  skip ifbench"; return; }
    log "  ifbench -> $(basename "${rd}")"
    "${PYBIN}" "${EVAL_ROOT}/benchmarks/ifbench/eval_ifbench.py" \
        --base-url "${BASE_URL}" --model "${served}" \
        --data /xuanwu-tank/north/xw27/multi/slime_scai/examples/ifbench/IFBench_eval.jsonl \
        --output "${rd}/ifbench.json" --trajectory-output "${rd}/ifbench_trajectories.jsonl" \
        --slime-root /xuanwu-tank/north/xw27/multi/slime_scai \
        --max-tokens 30000 --temperature 0.0 --concurrency 32 2>&1 | tee -a "${LOGD}/$(basename "${rd}")_ifbench.log"
}

# --- serve / stop a model on GPU3:8400 ---------------------------------------
serve() {  # ckpt_path  -> echoes server pid
    local ckpt="$1"
    CUDA_VISIBLE_DEVICES="${GPU}" nohup "${PYBIN}" -m sglang.launch_server \
        --model-path "${ckpt}" --served-model-name "${ckpt}" \
        --host 0.0.0.0 --port "${PORT}" --tp 1 --mem-fraction-static 0.85 \
        --tool-call-parser qwen --trust-remote-code \
        > "${LOGD}/eval_server.log" 2>&1 &
    echo $!
}
wait_healthy() {  # returns 0 when /health is 200, fails after ~30min
    for _ in $(seq 1 120); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:${PORT}/health 2>/dev/null)" = "200" ] && return 0
        sleep 15
    done
    return 1
}
stop_server() {  # pid
    local pid="$1"
    kill "${pid}" 2>/dev/null; sleep 8
    kill -9 "${pid}" 2>/dev/null
    # reap leftover sglang children on this port if any
    sleep 3
}

run_model() {  # name  ckpt
    local NAME="$1" CKPT="$2"
    LOGD="${EVAL_ROOT}/logs/${NAME}"; mkdir -p "${LOGD}"
    log "=== SERVE ${NAME} on GPU${GPU}:${PORT} ==="
    local SPID; SPID=$(serve "${CKPT}")
    if ! wait_healthy; then log "ERROR: ${NAME} server did not become healthy; see ${LOGD}/eval_server.log"; stop_server "${SPID}"; return 1; fi
    log "${NAME} server ready (pid ${SPID})"
    for i in $(seq 1 "${RUNS}"); do
        local RD="${EVAL_ROOT}/results/${NAME}/run${i}"; mkdir -p "${RD}/inspect_logs"
        log "--- ${NAME} run ${i}/${RUNS}: gpqa, aime, ifeval, ifbench ---"
        inspect_task inspect_evals/gpqa_diamond "${RD}" gpqa-diamond "${CKPT}" --max-tokens 30000 --temperature 0 --task-arg cot=true
        inspect_task inspect_evals/aime2025     "${RD}" aime2025     "${CKPT}" --max-tokens 30000 --temperature 0
        inspect_task inspect_evals/ifeval       "${RD}" ifeval       "${CKPT}" --max-tokens 30000 --temperature 0
        ifbench_step "${RD}" "${CKPT}"
    done
    log "=== ${NAME}: cheap suite COMPLETE (${RUNS} runs); stopping server ==="
    stop_server "${SPID}"
}

# ---- 1) kill GLM user-sim to free GPU3 --------------------------------------
# GLM_PID passed explicitly by launcher (kill-by-PID; avoids pgrep self-match).
# Verify it really is the GLM sglang server before killing.
if [ -n "${GLM_PID:-}" ] && ps -o cmd= -p "${GLM_PID}" 2>/dev/null | grep -q 'GLM-4.7-Flash'; then
    log "killing GLM user-sim pid ${GLM_PID} (frees GPU3)"; kill "${GLM_PID}" 2>/dev/null; sleep 8; kill -9 "${GLM_PID}" 2>/dev/null
else
    log "WARN: GLM_PID='${GLM_PID:-}' not a GLM server; skipping kill (GPU3 may be occupied)"
fi
# wait for GPU3 to actually free (>=40GB free) before serving
for _ in $(seq 1 40); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU} 2>/dev/null)
    [ "${free:-0}" -ge 40000 ] && break; sleep 5
done
log "GPU${GPU} free MiB before serve: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i ${GPU} 2>/dev/null)"

# ---- 2) TAU model, then 3) BASE model ---------------------------------------
run_model "Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau" \
          "/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau/hf_iter_0000300"
run_model "Qwen3-8B-Base" \
          "/xuanwu-tank/center/whx/Qwen3-8B-Base"

log "############ CHAIN COMPLETE: tau + base cheap suites done ############"
