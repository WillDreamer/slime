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

# Disaggregated vs colocate memory budgets. In disaggregated mode the actor owns
# its GPUs outright (no SGLang sharing), so we can raise the per-GPU token budget
# and SGLang's static memory fraction; in colocate they share and must stay low.
# ROLLOUT_NUM_GPUS is exported by the Greenland bootstrap (0 / unset = colocate).
if [ "${ROLLOUT_NUM_GPUS:-0}" -gt 0 ]; then
    # 15360 OOM'd在训练侧 logits 步(job 962645/890e96fb):
    # buf9 = empty_strided_cuda((s10,1,248320), fp32) 要 15.28GiB 分不出
    # (vocab=248320 巨大,fp32 logits 峰值 ∝ max-tokens)。降回 9216(colocate 验证过的值)。
    MAX_TOKENS_PER_GPU=9216           # 词表 248320 太大,fp32 logits 峰值高 -> 保持低
    SGLANG_MEM_FRACTION=0.85          # rollout GPUs are dedicated -> larger KV cache
else
    MAX_TOKENS_PER_GPU=9216           # colocate: shared mem, keep activation peak low
    SGLANG_MEM_FRACTION=0.7           # colocate: leave room for training weights
fi


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
   # (TorchMemorySaver malloc 要 ~16.6GB 分不出)。colocate 用 9216(对齐官方
   # run-qwen3-4B,降每卡 micro-batch token 数→降激活峰值);disaggregated 下训练
   # 卡独占,可回到 15360。由上面 MAX_TOKENS_PER_GPU 按模式自动选。
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}

   # logits-step OOM fix: slime computes log-probs/entropy on the FULL [T, V=248320]
   # fp32 logits at once when unset (default -1); compute_log_probs/_VocabParallelEntropy
   # each clone a full [T,V] (~15GiB at 16k seq). Chunk along the token dim -> peak
   # [chunk,V] (~1GiB), numerically identical, independent of batch/seq/max-tokens.
   # (Proven on the tau-bench 4B run; cheap insurance here too.)
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
   # (engine 进程被 SIGQUIT,RolloutManager 连不上 router 而失败)。colocate 用 0.7
   # 给 KV cache 动态增长 + 训练权重留余量;disaggregated 下推理卡独占,可上 0.85。
   # 由上面 SGLANG_MEM_FRACTION 按模式自动选。
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
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
   # 关闭 bias-dropout fusion:Megatron 的 bias_dropout_add_fused_train 带 @jit_fuser
   # (torch>=2.2 即 torch.compile)。dropout=0 时它融的是 no-op、零收益;而
   # --use-dynamic-batch-size 下每个 microbatch 形状/stride 不同,torch.compile 每步
   # 重编译,撞 recompile_limit(8) 后训练退化到 ~42s/microbatch 并卡死,被 Greenland
   # stuck-detector 杀(job 605065/025fc68d 即此)。关掉走纯 Python 路径,根除 recompile。
   --no-bias-dropout-fusion
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
# Multi-node: the Greenland bootstrap exports MASTER_ADDR (= head IP) and:
#   ACTOR_NUM_NODES   number of TRAINING nodes (colocate: = total nodes)
#   ROLLOUT_NUM_GPUS  disaggregated only: GPUs dedicated to rollout/SGLang.
#                     Unset/0 => colocate (rollout shares the actor GPUs).
# Single-node/dev-box: defaults to 1 node, 0 rollout-only GPUs, 127.0.0.1.
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-0}
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

# Multi-node: workers join the head asynchronously (Greenland brings child nodes
# up in random order). Wait until ALL GPUs (training + rollout) have registered
# before submitting, else slime's placement group can't be scheduled.
# Colocate: total = ACTOR_NUM_NODES*NUM_GPUS. Disaggregated: + ROLLOUT_NUM_GPUS,
# since those rollout-only nodes are separate Ray nodes in the same cluster.
# NOTE: query the GCS via ray.cluster_resources() (same source slime uses), NOT
# `ray status` text — right after `ray start --head` the monitor prints
# "No cluster status. It may take a few seconds..." and the text scrape then
# parses 0 forever, stalling this loop (observed on the live 6-node run).
EXPECTED_GPUS=$(( ACTOR_NUM_NODES * NUM_GPUS + ROLLOUT_NUM_GPUS ))
if [ "$EXPECTED_GPUS" -gt "$NUM_GPUS" ]; then
    echo "Waiting for ${EXPECTED_GPUS} GPUs (train $(( ACTOR_NUM_NODES * NUM_GPUS )) + rollout ${ROLLOUT_NUM_GPUS}) to register with Ray..."
    for i in $(seq 1 120); do
        GOT=$(python3 -c "import ray; ray.init(address='auto', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU',0))); ray.shutdown()" 2>/dev/null)
        GOT=${GOT:-0}
        echo "  [$i] ${GOT}/${EXPECTED_GPUS} GPUs registered"
        [ "$GOT" -ge "$EXPECTED_GPUS" ] && break
        sleep 5
    done
fi

# 通过 ray job submit 提交(而非裸跑 python3 train.py):让 train.py 的 ray.init()
# 接到 head 的 GCS,而不是自己猜一个 Docker-bridge IP(172.17.x.x)导致连不上 GCS。
# runtime-env 把 PYTHONPATH 等传进 job 进程。
# Multi-node: actors are scheduled onto worker nodes and inherit the JOB's
# runtime-env (not the worker shell), so the socket-iface pin must live here too,
# else cross-node NCCL/Gloo on those actors can still pick the link-local NIC.
#
# Also force EFA + make cross-node NCCL failures LOUD instead of a silent hang.
# A prior 6-node run hung forever on the first actor->rollout weight-sync NCCL
# collective (no NET/OFI line, just "ProcessGroupNCCL ... can cause a hang"),
# because Greenland placed nodes across two subnets (10.0.x / 10.1.x) and NCCL
# has no default collective timeout. These knobs (per Greenland docs):
#   FI_PROVIDER=efa            force libfabric to EFA; fail (not hang) if absent
#   FI_EFA_USE_DEVICE_RDMA=1   prefer the RDMA write path on p5en
#   NCCL_DEBUG=INFO + SUBSYS   surface "NET/OFI ... Selected provider is efa" vs Socket
#   TORCH_NCCL_BLOCKING_WAIT=1 + timeout   convert a stuck collective into an error
MULTINODE_ENV=""
if [ "${ACTOR_NUM_NODES:-1}" -gt 1 ]; then
    MULTINODE_ENV=",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-eth0}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME:-eth0}\",
    \"TP_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-eth0}\",
    \"FI_PROVIDER\": \"${FI_PROVIDER:-efa}\",
    \"FI_EFA_USE_DEVICE_RDMA\": \"1\",
    \"FI_EFA_FORK_SAFE\": \"1\",
    \"NCCL_DEBUG\": \"${NCCL_DEBUG:-INFO}\",
    \"NCCL_DEBUG_SUBSYS\": \"INIT,NET\",
    \"TORCH_NCCL_BLOCKING_WAIT\": \"1\",
    \"TORCH_NCCL_TIMEOUT_MS\": \"${TORCH_NCCL_TIMEOUT_MS:-600000}\",
    \"NCCL_ASYNC_ERROR_HANDLING\": \"1\",
    \"NCCL_NET_PLUGIN\": \"${NCCL_NET_PLUGIN:-/opt/amazon/ofi-nccl/lib/libnccl-net.so}\",
    \"LD_LIBRARY_PATH\": \"/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}\""
fi
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"${MULTINODE_ENV}
  }
}"

# Colocate vs disaggregated resource args.
#   colocate (ROLLOUT_NUM_GPUS=0): actor and SGLang time-share the same GPUs;
#     slime offloads one while the other runs. --colocate, no --rollout-num-gpus.
#   disaggregated (ROLLOUT_NUM_GPUS>0): actor occupies ACTOR_NUM_NODES*8 GPUs and
#     SGLang occupies a SEPARATE ROLLOUT_NUM_GPUS GPUs concurrently. Total placement
#     group = actor + rollout (= all nodes). No --colocate; pass --rollout-num-gpus.
RESOURCE_ARGS=(
   --actor-num-nodes ${ACTOR_NUM_NODES}
   --actor-num-gpus-per-node ${NUM_GPUS}
   --num-gpus-per-node ${NUM_GPUS}
)
if [ "${ROLLOUT_NUM_GPUS:-0}" -gt 0 ]; then
    RESOURCE_ARGS+=( --rollout-num-gpus ${ROLLOUT_NUM_GPUS} )
    echo "Disaggregated: actor=$(( ACTOR_NUM_NODES * NUM_GPUS )) GPU, rollout=${ROLLOUT_NUM_GPUS} GPU (no colocate)"
else
    RESOURCE_ARGS+=( --colocate )
    echo "Colocate: actor and rollout share $(( ACTOR_NUM_NODES * NUM_GPUS )) GPU"
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   ${RESOURCE_ARGS[@]} \
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