#!/bin/bash

# List the PIDs you want to wait for
# 等待8张卡显存超过5分钟都低于10%就运行

GPU_NUM=8
THRESHOLD_MB=144092  # 10% of 80GB card (A100), adjust if using different GPUs
TIME_LIMIT_SECS=300  # 5分钟=300秒
SLEEP_INTERVAL=100  # 每10秒检查一次

echo "Checking ${GPU_NUM} GPUs for >5min usage all below 10%..."

start_time=$(date +%s)

while true; do
    low_usage=true
    usage_list=()

    for idx in $(seq 0 $((GPU_NUM-1))); do
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $idx 2>/dev/null)
        # fallback to 0 if cannot get information for the GPU
        mem_used=${mem_used:-0}
        usage_list+=($mem_used)
        if [ "$mem_used" -gt "$THRESHOLD_MB" ]; then
            low_usage=false
        fi
    done

    curr_time=$(date +%s)
    elapsed=$((curr_time - start_time))

    echo "Elapsed: ${elapsed}s | GPU mem used: ${usage_list[*]} MB"

    if $low_usage; then
        if [ "$elapsed" -ge "$TIME_LIMIT_SECS" ]; then
            echo "All ${GPU_NUM} GPUs have memory usage below 10% for over 5 minutes. Proceeding..."
            break
        fi
    else
        # 有任何一张卡显存不低于10%了，重新计时
        start_time=$(date +%s)
    fi

    sleep $SLEEP_INTERVAL
done


bash /data1/whx/slime/examples/search-r1/run_qwen3_30b_a3b_seq_resume.sh