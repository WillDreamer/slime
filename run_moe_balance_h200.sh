#!/bin/bash
# =============================================================
# MoE Balance Static Measurement - H200 Server (hhzhang)
# =============================================================
# Measures expert load balance of a pretrained MoE model WITHOUT
# any RL training (forward-only mode, no weight updates).
# Compares routing decisions between SGLang (inference) and
# Megatron (training forward pass) on the same data.
#
# Usage:
#   bash run_moe_balance_h200.sh <run_id> [gpu_devices] [num_rollout]
#
# Examples:
#   bash run_moe_balance_h200.sh 1                # run 1, default GPU 0,2
#   bash run_moe_balance_h200.sh 2 2,4            # run 2, GPU 2,4
#   bash run_moe_balance_h200.sh 3 0,2 10         # run 3, GPU 0,2, 10 rollouts
#
# Output is saved to: /data1/hhzhang/moe_balance/base_ifbench_output_<run_id>/
# =============================================================

set -ex

# ============================================================
# 1. Config
# ============================================================
RUN_ID=${1:?  "Usage: bash run_moe_balance_h200.sh <run_id> [gpu_devices] [num_rollout]"}
GPU_DEVICES=${2:-"0,2"}
NUM_ROLLOUT=${3:-38}              # 38 rollouts * 8 samples = 304, covers all 300 IFBench samples
NUM_GPUS=2                        # minimum 2 for TP=2, EP=2

REPO_DIR=$(cd "$(dirname "$0")" && pwd)
DOCKER_IMAGE=slimerl/slime:latest

HF_CKPT=/xuanwu-tank/north/hhzhang/hf_cache/models/Qwen3-30B-A3B
TORCH_DIST_CKPT=/data1/hhzhang/moe_balance/Qwen3-30B-A3B_torch_dist
SLIME_CKPT=/data1/hhzhang/moe_balance/Qwen3-30B-A3B_slime
IFBENCH_DATA=/xuanwu-tank/north/xw27/multi/slime/examples/ifbench/IFBench_eval.jsonl
BALANCE_OUTPUT=/data1/hhzhang/moe_balance/base_ifbench_output_${RUN_ID}

mkdir -p ${SLIME_CKPT} ${BALANCE_OUTPUT}

# ============================================================
# 2. Convert checkpoint if needed (single GPU)
# ============================================================
if [ ! -f "${TORCH_DIST_CKPT}/latest_checkpointed_iteration.txt" ]; then
    echo "=== Converting HF checkpoint to torch_dist ==="
    FIRST_GPU=$(echo ${GPU_DEVICES} | cut -d',' -f1)
    docker run --rm --gpus "\"device=${FIRST_GPU}\"" \
        --ipc=host --ulimit memlock=-1 \
        -v /xuanwu-tank:/xuanwu-tank \
        -v /data1/hhzhang/moe_balance:/data1/hhzhang/moe_balance \
        -v ${REPO_DIR}:${REPO_DIR} \
        -w ${REPO_DIR} \
        -e PYTHONPATH=/root/Megatron-LM/ \
        -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
        ${DOCKER_IMAGE} \
        bash -c "
            source scripts/models/qwen3-30B-A3B.sh
            python tools/convert_hf_to_torch_dist.py \
                \${MODEL_ARGS[@]} \
                --hf-checkpoint ${HF_CKPT} \
                --save ${TORCH_DIST_CKPT}
        "
    echo "=== Checkpoint conversion done ==="
fi

# ============================================================
# 3. Run forward-only MoE balance debug
# ============================================================
echo "=== Starting MoE balance forward-only debug ==="
echo "=== GPUs: ${GPU_DEVICES}, Rollouts: ${NUM_ROLLOUT} ==="

docker run --rm --gpus "\"device=${GPU_DEVICES}\"" \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /xuanwu-tank:/xuanwu-tank \
    -v /data1/hhzhang/moe_balance:/data1/hhzhang/moe_balance \
    -v ${REPO_DIR}:${REPO_DIR} \
    -w ${REPO_DIR} \
    -e PYTHONPATH=/root/Megatron-LM/ \
    ${DOCKER_IMAGE} \
    bash -c '
set -ex

source scripts/models/qwen3-30B-A3B.sh

NUM_GPUS='"${NUM_GPUS}"'

ray start --head --node-ip-address 127.0.0.1 --num-gpus ${NUM_GPUS} \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

sleep 3

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MOE_BALANCE_FORWARD_ONLY\": \"1\",
    \"MOE_BALANCE_TRACKING\": \"1\",
    \"MOE_BALANCE_DATA_DIR\": \"'"${BALANCE_OUTPUT}"'\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node ${NUM_GPUS} \
    --colocate \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint '"${HF_CKPT}"' \
    --ref-load '"${TORCH_DIST_CKPT}"' \
    --load '"${SLIME_CKPT}"'/ \
    --save '"${SLIME_CKPT}"'/ \
    --save-interval 9999 \
    --prompt-data '"${IFBENCH_DATA}"' \
    --input-key prompt \
    --metadata-key metadata \
    --apply-chat-template \
    --rm-type ifbench \
    --use-rollout-routing-replay \
    --num-rollout '"${NUM_ROLLOUT}"' \
    --rollout-batch-size 8 \
    --n-samples-per-prompt 1 \
    --rollout-max-response-len 2048 \
    --rollout-temperature 1 \
    --global-batch-size 8 \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 2 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 4096 \
    --rollout-num-gpus-per-engine 2 \
    --sglang-mem-fraction-static 0.5 \
    --advantage-estimator grpo \
    --eps-clip 0.2 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --use-precision-aware-optimizer \
    --optimizer-cpu-offload \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash
'

echo "=== Done! Run ${RUN_ID} output: ${BALANCE_OUTPUT} ==="
