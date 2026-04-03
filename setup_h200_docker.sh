#!/bin/bash
# =============================================================
# MoE Balance Debug - H200 Server Setup (Docker)
# =============================================================
# Prerequisites: Docker with GPU support (nvidia-docker), 2+ GPUs
# Model: Qwen3-30B-A3B base (NOT Instruct)
# Task: IFBench (local reward, no external API)
# =============================================================

set -ex

# ============================================================
# 1. Config - modify these paths for your server
# ============================================================
WORKDIR=/data/moe_balance          # change to your preferred path
REPO_DIR=$(cd "$(dirname "$0")" && pwd)  # slime repo root
DOCKER_IMAGE=slimerl/slime:latest
HF_MODEL_ID=Qwen/Qwen3-30B-A3B

HF_CKPT=${WORKDIR}/Qwen3-30B-A3B
TORCH_DIST_CKPT=${WORKDIR}/Qwen3-30B-A3B_torch_dist
SLIME_CKPT=${WORKDIR}/Qwen3-30B-A3B_slime
IFBENCH_DATA=${WORKDIR}/ifbench/IFBench_eval.jsonl
BALANCE_OUTPUT=${WORKDIR}/moe_balance_output
NUM_GPUS=2   # minimum 2 for TP=2, EP=2

mkdir -p ${WORKDIR} ${BALANCE_OUTPUT}

# ============================================================
# 2. Pull Docker image
# ============================================================
docker pull ${DOCKER_IMAGE}

# ============================================================
# 3. Download model (HuggingFace)
# ============================================================
if [ ! -d "${HF_CKPT}" ]; then
    echo "=== Downloading Qwen3-30B-A3B base checkpoint ==="
    # Option A: huggingface-cli (if available on host)
    # huggingface-cli download ${HF_MODEL_ID} --local-dir ${HF_CKPT}
    # Option B: inside container
    docker run --rm \
        -v ${WORKDIR}:${WORKDIR} \
        ${DOCKER_IMAGE} \
        huggingface-cli download ${HF_MODEL_ID} --local-dir ${HF_CKPT}
else
    echo "=== HF checkpoint already exists, skipping ==="
fi

# ============================================================
# 4. Download IFBench data
# ============================================================
if [ ! -f "${IFBENCH_DATA}" ]; then
    echo "=== Downloading IFBench dataset ==="
    mkdir -p ${WORKDIR}/ifbench
    docker run --rm \
        -v ${WORKDIR}:${WORKDIR} \
        ${DOCKER_IMAGE} \
        bash -c "pip install huggingface_hub && python -c \"
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Coldog2333/IFBench', filename='IFBench_eval.jsonl',
                repo_type='dataset', local_dir='${WORKDIR}/ifbench')
\""
else
    echo "=== IFBench data already exists, skipping ==="
fi

# ============================================================
# 5. Convert checkpoint: HF -> torch_dist (needs GPU)
# ============================================================
if [ ! -d "${TORCH_DIST_CKPT}" ]; then
    echo "=== Converting HF checkpoint to torch_dist ==="
    docker run --rm --gpus all \
        -v ${WORKDIR}:${WORKDIR} \
        -v ${REPO_DIR}:${REPO_DIR} \
        -w ${REPO_DIR} \
        ${DOCKER_IMAGE} \
        bash -c "
            source scripts/models/qwen3-30B-A3B.sh
            python tools/convert_hf_to_torch_dist.py \
                \${MODEL_ARGS[@]} \
                --hf-checkpoint ${HF_CKPT} \
                --save ${TORCH_DIST_CKPT}
        "
    echo "=== Checkpoint conversion done ==="
else
    echo "=== torch_dist checkpoint already exists, skipping ==="
fi

# ============================================================
# 6. Run forward-only MoE balance debug
# ============================================================
echo "=== Starting MoE balance forward-only debug ==="

docker run --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${WORKDIR}:${WORKDIR} \
    -v ${REPO_DIR}:${REPO_DIR} \
    -w ${REPO_DIR} \
    ${DOCKER_IMAGE} \
    bash -c '
set -ex

source scripts/models/qwen3-30B-A3B.sh

NUM_GPUS='"${NUM_GPUS}"'

# Start Ray head
ray start --head --node-ip-address 127.0.0.1 --num-gpus ${NUM_GPUS} \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

sleep 3

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MOE_BALANCE_FORWARD_ONLY\": \"1\",
    \"MOE_BALANCE_TRACKING\": \"1\",
    \"MOE_BALANCE_OUTPUT_DIR\": \"'"${BALANCE_OUTPUT}"'\"
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
    --num-rollout 2 \
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
    --sglang-mem-fraction-static 0.6 \
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

echo "=== Done! Balance output: ${BALANCE_OUTPUT} ==="
