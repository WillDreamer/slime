#!/bin/bash
# ============================================================================
# Search model (Qwen3-8B-Base-Math-SeaSFT-Search) — CHEAP benchmarks only.
# Per user request 2026-06-16: pause search / tau2 / browsecomp for this model
# and first run the remaining benchmarks: GPQA, AIME, IFBench, IFEval.
#
# Idempotent: each step skips if its output already exists (success .eval for
# inspect tasks, ifbench.json for ifbench). Reuses whx's :30012 server.
# ============================================================================
set -u
EVAL_ROOT="/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai"
IPY="${EVAL_ROOT}/.venv_inspect/bin/python"
PYBIN="/xuanwu-tank/north/xw27/envs/sglang_env/bin/python"
NAME="Qwen3-8B-Base-Math-SeaSFT-Search"
SERVED="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420"
PORT=30012
BASE_URL="http://127.0.0.1:${PORT}/v1"
RUNS="${RUNS:-3}"
LOGD="${EVAL_ROOT}/logs/${NAME}"; mkdir -p "${LOGD}"
log() { echo "[$(date +%F' '%T)] [${NAME}/cheap] $*"; }

inspect_task() {  # task, run_dir, label, extra...
    local task="$1" rd="$2" label="$3"; shift 3
    if ls "${rd}/inspect_logs/"*"${label}"*.eval >/dev/null 2>&1; then
        if "${IPY}" - "$rd" "$label" <<'PY' 2>/dev/null; then log "skip ${label} (success exists in $(basename "$rd"))"; return; fi
import sys, glob
from inspect_ai.log import read_eval_log
rd, lab = sys.argv[1], sys.argv[2]
ok = any(read_eval_log(f, header_only=True).status == "success"
         for f in glob.glob(f"{rd}/inspect_logs/*{lab}*.eval"))
sys.exit(0 if ok else 1)
PY
    fi
    log "inspect ${task} -> $(basename "${rd}")"
    "${IPY}" "${EVAL_ROOT}/benchmarks/inspect/run_inspect_api.py" \
        --task "${task}" --model-name "${SERVED}" --base-url "${BASE_URL}" \
        --log-dir "${rd}/inspect_logs" "$@" 2>&1 | tee -a "${LOGD}/$(basename "${rd}")_${label}.log"
}

ifbench_step() {  # run_dir
    local rd="$1"; [ -f "${rd}/ifbench.json" ] && { log "skip ifbench ($(basename "$rd"))"; return; }
    log "ifbench -> $(basename "${rd}")"
    "${PYBIN}" "${EVAL_ROOT}/benchmarks/ifbench/eval_ifbench.py" \
        --base-url "${BASE_URL}" --model "${SERVED}" \
        --data /xuanwu-tank/north/xw27/multi/slime_scai/examples/ifbench/IFBench_eval.jsonl \
        --output "${rd}/ifbench.json" --trajectory-output "${rd}/ifbench_trajectories.jsonl" \
        --slime-root /xuanwu-tank/north/xw27/multi/slime_scai \
        --max-tokens 30000 --temperature 0.0 --concurrency 32 2>&1 | tee -a "${LOGD}/$(basename "${rd}")_ifbench.log"
}

for i in $(seq 1 "${RUNS}"); do
    RD="${EVAL_ROOT}/results/${NAME}/run${i}"; mkdir -p "${RD}/inspect_logs"
    log "=== run ${i}/${RUNS}: gpqa, aime24, aime25, ifeval, ifbench ==="
    inspect_task inspect_evals/gpqa_diamond "${RD}" gpqa-diamond --max-tokens 30000 --temperature 0 --task-arg cot=true
    inspect_task inspect_evals/aime2024     "${RD}" aime2024     --max-tokens 30000 --temperature 0
    inspect_task inspect_evals/aime2025     "${RD}" aime2025     --max-tokens 30000 --temperature 0
    inspect_task inspect_evals/ifeval       "${RD}" ifeval       --max-tokens 30000 --temperature 0
    ifbench_step "${RD}"
done
log "############ ${NAME}: cheap benchmarks COMPLETE (${RUNS} runs) ############"
