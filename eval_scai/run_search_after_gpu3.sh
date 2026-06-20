#!/bin/bash
# Wait for the GPU3 pipeline (watcher pid $1) to finish, then run full 7-dataset
# search for base/math/tau on GPU3.
set -u
WPID="${1:?usage: run_search_after_gpu3.sh <gpu3_watcher_pid>}"
echo "[search_gpu3] waiting for GPU3 pipeline (pid ${WPID})..."
while kill -0 "${WPID}" 2>/dev/null; do sleep 120; done
echo "[search_gpu3] GPU3 free at $(date +%F' '%T); starting base/math/tau full-search"
sleep 15
SEARCH_N_PER_DS=500 SEARCH_CONCURRENCY=64 GPU=3 PORT=8410 MODELS="base math tau" \
    bash /xuanwu-tank/north/xw27/multi/slime_scai/eval_scai/rerun_search_all.sh
