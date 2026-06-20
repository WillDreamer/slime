#!/bin/bash
# ============================================================================
# Unified benchmark runner: evaluate one HF checkpoint on all 8 benchmarks.
#
#   GPQA | MMLU | AIME | BrowseComp-Plus | Search | Tau2 | IFBench | IFEval
#
# Usage:
#   CKPT=/path/to/hf_checkpoint bash run_all.sh                  # all 8
#   CKPT=... bash run_all.sh gpqa aime ifbench                   # a subset
#
# Common overrides (see env.sh for the full list):
#   EVAL_GPUS=2,3      GPUs for the served model (TP = #GPUs)
#   EVAL_PORT=8400     server port
#   SEARCH_DATASET=nq|full     RETRIEVER_GPU=<id>
#   TAU2_LIMIT=5       smoke-test tau2
#   BCP_LIMIT=20       smoke-test browsecomp-plus
#
# Results land in results/<ckpt_name>/ ; logs in logs/<ckpt_name>/.
# inspect benchmarks additionally write .eval logs readable with
# `inspect view --log-dir results/<ckpt_name>/inspect_logs`.
# ============================================================================
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

ALL_BENCHMARKS=(gpqa mmlu aime ifeval tau2 ifbench search browsecomp)
BENCHMARKS=("${@}")
[ ${#BENCHMARKS[@]} -eq 0 ] && BENCHMARKS=("${ALL_BENCHMARKS[@]}")

echo "============================================================"
echo " Checkpoint : ${CKPT}"
echo " Name       : ${CKPT_NAME}"
echo " Server     : ${EVAL_BASE_URL} (GPUs ${EVAL_GPUS}, TP ${EVAL_TP})"
echo " Benchmarks : ${BENCHMARKS[*]}"
echo " Results    : ${RESULTS_DIR}"
echo "============================================================"

# 1. Serve the checkpoint (reused if already up).
bash "${SCRIPT_DIR}/serve_model.sh"

# 2. Run each requested benchmark; keep going on individual failures.
declare -A STATUS
for b in "${BENCHMARKS[@]}"; do
    echo; echo ">>> [$(date +%H:%M:%S)] running ${b}"
    set +e
    case "${b}" in
      gpqa|mmlu|aime|ifeval|tau2)
        bash "${SCRIPT_DIR}/benchmarks/inspect/run_inspect.sh" "${b}" \
            2>&1 | tee "${LOG_DIR}/${b}.log"
        rc=${PIPESTATUS[0]}
        ;;
      ifbench)
        bash "${SCRIPT_DIR}/benchmarks/ifbench/run_ifbench.sh"; rc=$?
        ;;
      search)
        bash "${SCRIPT_DIR}/benchmarks/search/run_search.sh"; rc=$?
        ;;
      browsecomp)
        bash "${SCRIPT_DIR}/benchmarks/browsecomp_plus/run_browsecomp.sh"; rc=$?
        ;;
      *)
        echo "Unknown benchmark '${b}'" >&2; rc=1
        ;;
    esac
    set -e
    STATUS[${b}]=$([ "${rc}" -eq 0 ] && echo OK || echo "FAIL(rc=${rc})")
done

# 3. Summary.
echo; echo "================== RUN SUMMARY (${CKPT_NAME}) =================="
for b in "${BENCHMARKS[@]}"; do printf "  %-12s %s\n" "${b}" "${STATUS[${b}]}"; done
echo "Results dir: ${RESULTS_DIR}"
echo "inspect logs: inspect view --log-dir ${RESULTS_DIR}/inspect_logs"
