#!/bin/bash

# List the PIDs you want to wait for
PIDS=(2493478 2495767)

echo "Waiting for PIDs to terminate: ${PIDS[@]}"

while true; do
    alive=()

    # Check which PIDs are still running
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
        fi
    done

    # If none left → break
    if [ ${#alive[@]} -eq 0 ]; then
        echo "All PIDs have terminated."
        break
    fi

    echo "Still running: ${alive[@]} ... checking again in 5s"
    sleep 50
done

# # ----------------------------
# # START NEXT COMMAND HERE
# # ----------------------------
# echo "Starting next job..."
# bash /data1/xw27/agent/ARLArena/examples/math_trainer/train_grpo_s_cispo.sh

bash /data1/whx/slime/examples/tau-bench/run_qwen3_30B.sh