#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Qwen3.5-35B-A3B-Base math-reasoning GRPO RL on Greenland — multi-node (_mns)
#
# Derived from run_qwen35_4b_base_mns.sh (the VALIDATED Greenland multinode
# launcher). The Ray topology, colocate/disaggregated env-gating, GPU-registration
# wait loop, EFA/NCCL runtime-env, and --no-bias-dropout-fusion are kept
# byte-for-byte. What changed for 35B-A3B (a 256-expert MoE vs the dense 4B):
#   * source scripts/models/qwen3.5-35B-A3B.sh  (all MoE MODEL_ARGS + MOE_LAYER_FREQ)
#   * PERF_ARGS: expert parallelism EP=8/ETP=1 + TP=2 (was TP=1 DP-only on 4B).
#     EP=8 divides 256 experts (32/rank); TP=2 shards attention+shared-expert so
#     35B weights fit. CP=1 (math is 16k ctx, no need for the SWE script's CP=8).
#   * MISC_ARGS: flex dispatcher + DeepEP (verified present in image 392e9980:
#     `deep_ep` importable). This OVERRIDES the alltoall in the model config
#     (Megatron takes the last --moe-token-dispatcher-type).
#   * SGLANG_ARGS: per-engine=8 (one full node/engine, TP=8 NVLink) + ep/dp-attn
#     for serving the MoE (mirrors the 8-node SWE 35B-A3B reference, scaled).
#   * --use-distributed-optimizer (shard optimizer state — needed at 35B).
#   * max-tokens-per-gpu=6144 (vs 4B's 9216): 35B weights + MoE activation peak
#     leave less room; fp32 logits at 6144 ~= 6GB (the 4B OOM was the logits step).
#   * CKPT/output/stage paths -> Qwen3.5-35B-A3B-Base.
#
# Submit (disaggregated, 6 nodes = 2 train + 4 rollout):
#   python3 greenland_cli_mns.py obx \
#     --script examples/math_reasoning/run_qwen35_35b_a3b_base_mns.sh \
#     --num-nodes 6 --rollout-nodes 4 \
#     --stage-model Qwen3.5/Qwen3.5-35B-A3B-Base/ \
#     --stage-model Qwen3.5/Qwen3.5-35B-A3B-Base_torch_dist/
# ─────────────────────────────────────────────────────────────────────────────

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
# 35B-A3B model config: brings all MoE MODEL_ARGS (--num-experts 256,
# --moe-router-topk 8, --moe-shared-expert-gate, ...) + the MOE_LAYER_FREQ loop.
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-35B-A3B.sh"
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
WANDB_GROUP="reason_rl_Qwen35-35B-A3B_bs_${ROLLOUT_BATCH_SIZE}"

# ── Expert/model parallelism for the actor (training) side ──
# 35B-A3B on the disaggregated actor (2 nodes = 16 GPU):
#   TP=2 PP=1 CP=1 EP=8 ETP=1  ->  attention data-parallel = 16/(TP*PP*CP) = 8.
# EP=8 | 256 experts (32 experts/EP-rank). global-batch (2048) must be divisible
# by attention-DP (8): 2048/8=256, clean. Single-node colocate (8 GPU) still works:
# EP=8 spans the 8 GPUs, attention-DP=4. Overridable via env.
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
ETP_SIZE="${ETP_SIZE:-1}"

# Disaggregated vs colocate memory budgets. ROLLOUT_NUM_GPUS is exported by the
# Greenland bootstrap (0 / unset = colocate).
if [ "${ROLLOUT_NUM_GPUS:-0}" -gt 0 ]; then
    # 6144 (vs 4B's 9216): 35B weights + MoE expert activations raise the per-GPU
    # peak; fp32 logits (vocab 248320) at 6144 ~= 6GB. Keep headroom to avoid the
    # logits-step OOM that killed the 4B run at 15360. Env-overridable: with only
    # 1 training node (TP forced to 1, attention not sharded) lower this further.
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}"
    SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
else
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}"
    SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.7}"
fi

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3.5/Qwen3.5-35B-A3B-Base/
   --ref-load ${MODEL_ROOT}/Qwen3.5/Qwen3.5-35B-A3B-Base_torch_dist/
   --load ${MODEL_ROOT}/AECE/Qwen3.5-35B-A3B-Base-Math/
   --save ${MODEL_ROOT}/AECE/Qwen3.5-35B-A3B-Base-Math/
   # 每 2 个 step 存一次(崩溃可从最近恢复);但只永久保留 step 为 50 倍数的
   # checkpoint(step=rollout_id+1),其余非永久的只滚动保留最近 1 个。
   --save-interval 2
   --save-retain-interval 50
)

ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/AECE/reason_rl_math_35b/rollout_debug"

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

   --log-passrate

   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size ${ETP_SIZE}

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}

   # OOM fix (job 482838 died here: ppo_utils.py:698 compute_log_probs(logits.clone())
   # tried to alloc 15.39GiB with only 12.4GiB free — 35B-A3B weights+optim+activations
   # already held 102.8GiB). slime computes log-probs/entropy on the FULL [T, V=248320]
   # fp32 logits at once when this is unset (default -1), and compute_log_probs/
   # _VocabParallelEntropy each materialize another full [T,V] clone. With seq up to
   # 16384 that clone is ~15GiB. Chunking along the token dim bounds the peak to
   # [chunk, V] (~1GiB at 1024) — numerically identical, does NOT touch
   # batch-size / seq-len / max-tokens-per-gpu. (Same fix proven on the tau-bench 4B run.)
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

   # TIS (Truncated Importance Sampling): corrects the train/infer logprob
   # mismatch (SGLang rollout vs Megatron train forward) by reweighting the
   # policy-gradient with a truncated importance ratio. Especially important for
   # MoE — expert routing differs slightly between the two backends, so the
   # uncorrected PG is biased; TIS caps the ratio (mis.yaml: truncate to [0.5,2.0],
   # batch-normalized) to stabilize the gradient. Needs the CUSTOM_ARGS below +
   # per-token rollout_log_probs (slime collects these by default, return_logprob=True).
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # 35B: shard optimizer state across DP ranks (ZeRO-1-like) — required to fit
   # Adam moments for a model this size. CPU-offload left off for now (EFA
   # cross-node + offload can be slow); enable if optimizer-state OOM appears.
   --use-distributed-optimizer
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

# ── Rollout engine (SGLang) for the 35B-A3B MoE ──
# 4 rollout nodes (32 GPU): one engine per full node (TP=8 over intra-node NVLink)
# => 4 engines. EP=8 (| 256), DP-attention on, dense layers TP=1 — mirrors the
# 8-node SWE 35B-A3B reference, scaled to 32 GPU. rollout-num-gpus is supplied by
# the disaggregated RESOURCE_ARGS below (= ROLLOUT_NUM_GPUS from the bootstrap).
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-8}"
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_TP_SIZE}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
   --sglang-server-concurrency 1024
   --sglang-enable-dp-attention
   --sglang-dp-size ${ROLLOUT_TP_SIZE}
   --sglang-ep-size ${ROLLOUT_TP_SIZE}
   --sglang-moe-dense-tp-size 1
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)

   # 自定义 all-reduce kernel 在此 H200/驱动上做 CUDA graph capture 时报
   # "custom_all_reduce.cuh: CUDA error: invalid argument"。关掉它(回退 NCCL)。
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

   # MoE token dispatch: flex + DeepEP (verified importable in image 392e9980).
   # This OVERRIDES --moe-token-dispatcher-type alltoall from the model config
   # (Megatron honors the LAST occurrence). DeepEP is faster over EFA for the
   # all-to-all expert routing; if a future image lacks deep_ep, drop these two
   # lines and the config's alltoall takes over.
   --moe-token-dispatcher-type flex
   --moe-enable-deepep

   # 关闭 bias-dropout fusion:Megatron 的 bias_dropout_add_fused_train 带 @jit_fuser
   # (torch>=2.2 即 torch.compile);--use-dynamic-batch-size 下每 microbatch 形状不同
   # → 每步重编译撞 recompile_limit(8) → 训练退化卡死(base job 605065 即此)。关掉。
   --no-bias-dropout-fusion
)

CUSTOM_ARGS=(
   # TIS config + weight fn (paired with --use-tis above). mis.yaml: truncate mode,
   # ratio bounds [0.5, 2.0], batch-normalize. The _with_cp fn works at any CP
   # (CP=1 here -> the all_gather_with_cp is a no-op single-rank gather).
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

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
# Query the GCS via ray.cluster_resources() (same source slime uses), NOT
# `ray status` text — right after `ray start --head` the monitor prints
# "No cluster status..." and the text scrape then parses 0 forever (stalls the loop).
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
# 接到 head 的 GCS。runtime-env 把 PYTHONPATH 等传进 job 进程。
# Multi-node: actors are scheduled onto worker nodes and inherit the JOB's
# runtime-env (not the worker shell), so the socket-iface pin + EFA knobs must
# live here too. Force EFA + make cross-node NCCL failures LOUD (TORCH_NCCL
# timeout) instead of a silent hang; LD_LIBRARY_PATH prepends /opt/amazon/efa/lib
# so the aws-ofi-nccl plugin links the AWS libfabric (FABRIC_1.8), not the system one.
MULTINODE_ENV=""
if [ "${ACTOR_NUM_NODES:-1}" -gt 1 ]; then
    MULTINODE_ENV=",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-eth0}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME:-eth0}\",
    \"TP_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-eth0}\",
    \"FI_PROVIDER\": \"${FI_PROVIDER:-efa}\",
    \"FI_EFA_USE_DEVICE_RDMA\": \"1\",
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
#   colocate (ROLLOUT_NUM_GPUS=0): actor and SGLang time-share the same GPUs.
#   disaggregated (ROLLOUT_NUM_GPUS>0): actor occupies ACTOR_NUM_NODES*8 GPUs and
#     SGLang occupies a SEPARATE ROLLOUT_NUM_GPUS GPUs concurrently.
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
