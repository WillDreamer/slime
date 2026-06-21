#!/bin/bash

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Overridable via env (Greenland OBX entrypoint sets these to the container's
# local NVMe mirror; defaults keep dev-box behaviour unchanged).
ROOT_DIR=${ROOT_DIR:-/data2/whx}
MODEL_ROOT=${MODEL_ROOT:-/data2/whx/models}
DATA_ROOT=${DATA_ROOT:-/data2/whx/data}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-4B.sh"
WANDB_API_KEY="${WANDB_API_KEY}"

GPU_LIST=(0 1 2 3 4 5 6 7)
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# Automatically detect the number of n_gpus_per_node
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

ROLLOUT_BATCH_SIZE=256
GLOBAL_BATCH_SIZE=2048
WANDB_GROUP="reason_rl_Qwen35-4B_bs_${ROLLOUT_BATCH_SIZE}"


CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base/
   --ref-load ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base_torch_dist/
   --load ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base_torch_dist/
   --save ${MODEL_ROOT}/AECE/Qwen3.5-4B-Base-Math/
   --save-interval 10
)

# 将 rollout 数据全部保存到本地（每次 rollout 的 samples 会存为 .pt 文件）
# 路径中的 {rollout_id} 会被替换为实际 rollout 编号；评估数据会存为 eval_{rollout_id}.pt
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/AECE/reason_rl_math/rollout_debug"
# 二选一：
# 1) 只保存 rollout 数据：
#    --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
# 2) 使用 dump-details 会同时保存 rollout_data、train_data 以及 tokenizer 等，便于事后分析：
#    --dump-details "${ROLLOUT_DEBUG_DIR}"

ROLLOUT_ARGS=(
   --prompt-data "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 500
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 16
   --rollout-max-response-len 16384
   --rollout-temperature 1
   --over-sampling-batch-size 512
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data

   # wandb 记录 rollout 的 reward / pass@k（否则训练侧才有 rollout/raw_reward，且默认不记 passrate）
   --log-passrate

   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
   # 若还需要 train_data、tokenizer 等，可改用：--dump-details "${ROLLOUT_DEBUG_DIR}"

   #eval args
#    --eval-interval 10
#    --eval-prompt-data aime "${SCRIPT_DIR}/../../data/aime-2024.jsonl" aime25 "${SCRIPT_DIR}/../../data/aime-2025.jsonl"
#    --n-samples-per-eval-prompt 16
#    --eval-max-response-len 16384
#    --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   # colocate + 16k 长序列下,训练侧反向激活峰值过高导致 Megatron OOM
   # (TorchMemorySaver malloc 要 ~16.6GB 分不出)。15360→9216 对齐官方 run-qwen3-4B,
   # 降每卡 micro-batch token 数 → 降激活峰值。代价:训练 step 略多/略慢。
   --max-tokens-per-gpu 9216

   # logits-step OOM fix: chunk log-probs/entropy along the token dim so the peak
   # buffer is [chunk, V=248320] (~1GiB) instead of the full [T, V] clone (~15GiB at
   # 16k seq). Numerically identical; independent of batch/seq/max-tokens.
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   # --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project AECE
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   # colocate 下 SGLang 与训练权重共享显存;0.8 时 rollout 跑到峰值会 CUDA OOM
   # (engine 进程被 SIGQUIT,RolloutManager 连不上 router 而失败)。降到 0.7
   # 给 KV cache 动态增长 + 训练权重留更多余量。
   --sglang-mem-fraction-static 0.7
   --sglang-server-concurrency 1024
   # --sglang-ep-size ${NUM_GPUS}
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)

   # 自定义 all-reduce kernel 在此 H200/驱动上做 CUDA graph capture 时报
   # "custom_all_reduce.cuh: CUDA error: invalid argument"。关掉它(回退 NCCL
   # all-reduce),保留 cuda graph,推理性能几乎无损。
   --sglang-disable-custom-all-reduce
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(

   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# # launch the master node of ray in container
# export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# # Build the runtime environment JSON with proper variable substitution
# RUNTIME_ENV_JSON="{
#   \"env_vars\": {
#     \"PYTHONPATH\": \"/root/Megatron-LM/\",
#     \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
#     \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
#   }
# }"

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONPATH="${SLIME_DIR:-${ROOT_DIR}/slime}:${MEGATRON_DIR:-${ROOT_DIR}/Megatron-LM}:${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
# Dev-box 上需要前置系统库目录；但在 Greenland 镜像里前置会让 libcudnn_graph.so.9
# 解析到系统旧版、与 pip 版 libcudnn_cnn 符号错配。容器 bootstrap 设 SKIP_SYS_LDPATH=1 跳过。
if [ -z "${SKIP_SYS_LDPATH:-}" ]; then
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
fi
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir ${RAY_TEMP_DIR:-${ROOT_DIR}/ray_temp}

# 通过 ray job submit 提交(而非裸跑 python3 train.py):让 train.py 的 ray.init()
# 接到 head 的 GCS,而不是自己猜一个 Docker-bridge IP(172.17.x.x)导致连不上 GCS。
# runtime-env 把 PYTHONPATH 等传进 job 进程。
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}