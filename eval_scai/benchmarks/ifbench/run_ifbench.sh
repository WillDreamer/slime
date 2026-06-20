#!/bin/bash
# IFBench eval wrapper. Usage: CKPT=<hf ckpt> bash run_ifbench.sh
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../env.sh"

"${PYBIN}" "${SCRIPT_DIR}/eval_ifbench.py" \
    --base-url "${EVAL_BASE_URL}" \
    --model "${CKPT_NAME}" \
    --data "${IFBENCH_DATA}" \
    --output "${RESULTS_DIR}/ifbench.json" \
    --trajectory-output "${RESULTS_DIR}/ifbench_trajectories.jsonl" \
    --slime-root "${SLIME_ROOT}" \
    --max-tokens "${IFBENCH_MAX_TOKENS}" \
    --temperature "${IFBENCH_TEMPERATURE}" \
    --concurrency "${IFBENCH_CONCURRENCY}" \
    2>&1 | tee "${LOG_DIR}/ifbench.log"
