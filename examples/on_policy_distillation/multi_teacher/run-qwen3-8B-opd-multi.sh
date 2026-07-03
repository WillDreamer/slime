#!/bin/bash
# Multi-teacher on-policy distillation (pure OPD, task reward = 0).
#
#   student : MultiStageRL/Qwen3-8B-Base-TauSFT-Tau   (megatron torch_dist, --load)
#   teachers: Qwen3-8B-Base-SeaSFT-Search  (search domain)  -> remote SGLang server
#             Qwen3-8B-Base-TauSFT         (tau    domain)  -> remote SGLang server
#
# This box ONLY trains the student (colocate actor + rollout engine). The two teacher
# servers, the Search retriever, and the Tau user-simulator all live on OTHER nodes and
# are reached over HTTP. See "REMOTE SERVICES" below.
#
# usage: bash examples/on_policy_distillation/multi_teacher/run-qwen3-8B-opd-multi.sh
set -ex

ROOT_DIR=/home/ec2-user
MODEL_ROOT=/home/ec2-user
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ----------------------------------------------------------------------------
# REMOTE SERVICES (edit these 2 teacher URLs to your other nodes)
# ----------------------------------------------------------------------------
# Each teacher is a plain SGLang server started on a remote node with, e.g.:
#   python3 -m sglang.launch_server --model-path <TEACHER_HF> --port 13141 \
#       --tp 1 --context-length 32768 --mem-fraction-static 0.8
# NOTE: teachers must be HF format. The ckpts are megatron torch_dist, so convert once:
#   python tools/convert_torch_dist_to_hf.py ${MODEL_ARGS[@]} \
#       --load  ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-SeaSFT-Search \
#       --save  ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-SeaSFT-Search_hf
#   (likewise for Qwen3-8B-Base-TauSFT -> ..._hf)
export OPD_TEACHER_URL_SEARCH=${OPD_TEACHER_URL_SEARCH:-http://TEACHER_SEARCH_NODE:13141/generate}
export OPD_TEACHER_URL_TAU=${OPD_TEACHER_URL_TAU:-http://TEACHER_TAU_NODE:13142/generate}
export OPD_TEACHER_TIMEOUT=${OPD_TEACHER_TIMEOUT:-600}

# Tau user-simulator (remote, OpenAI-compatible). Same endpoint your tau RL run used.
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://localhost:38000/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
# Search retriever: generate_with_search_tools_*.py hardcodes http://127.0.0.1:8000/retrieve,
# so forward the remote retriever to localhost:8000 (e.g. an ssh -L tunnel), mirroring
# the user-sim tunnel. Nothing to set here; just make :8000 reachable.
export SEARCH_R1_STRIP_THINK=0

# ----------------------------------------------------------------------------
# Data: build the mixed (search pre-templated + tau index) dataset once.
# ----------------------------------------------------------------------------
MIXED_DATA=${MODEL_ROOT}/MultiStageRL/opd_mixed/train.jsonl
if [ ! -f "${MIXED_DATA}" ]; then
  python3 "${SCRIPT_DIR}/prepare_opd_mixed_data.py" \
    --hf ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base \
    --search-parquet ${MODEL_ROOT}/MultiStageRL/nq_hotpotqa_train/train.parquet \
    --tau-jsonl ${ROOT_DIR}/tau-bench/retail_train_tasks.jsonl \
    --out "${MIXED_DATA}"
fi

# ----------------------------------------------------------------------------
# GPUs (student training only)
# ----------------------------------------------------------------------------
GPU_LIST=(0 1 2 3 4 5 6 7)
export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
NUM_GPUS=${#GPU_LIST[@]}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0

source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"   # -> MODEL_ARGS

ROLLOUT_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=256

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base          # tokenizer + config
   --ref-load      ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-TauSFT-Tau
   --load          ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-TauSFT-Tau   # STUDENT
   --save          ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-TauSFT-Tau_OPD/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${MIXED_DATA}
   --input-key prompt          # NOTE: NO --apply-chat-template (search is pre-templated; tau=index)
   --rollout-shuffle           # mixes search+tau within every rollout batch -> both teachers active each step
   --num-rollout 300
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 8    # reward=0 -> grouping is neutral; can lower to save rollout cost
   --rollout-max-response-len 4096   # max(search 4096, tau 2048)
   --rollout-temperature 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

# OPD reward: per-sample teacher logprobs (domain-routed), task reward = 0.
RM_ARGS=(
   --custom-rm-path opd_multi_teacher.reward_func
   --custom-reward-post-process-path opd_multi_teacher.post_process_rewards
)
# IMPORTANT: do NOT add the tau reward-std dynamic-sampling filter
# (check_raw_task_reward_nonzero_std) — with reward=0 it would drop every group.

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

# OPD on top of GRPO; the only signal is the OPD KL penalty.
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.5           # start small: this pulls the RL'd student back toward the teachers
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${NUM_GPUS}
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ----------------------------------------------------------------------------
# launch
# ----------------------------------------------------------------------------
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# multi_teacher (custom fns) + both example dirs (native generates) on PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${ROOT_DIR}/slime/examples/search-r1:${ROOT_DIR}/slime/examples/tau-bench:${ROOT_DIR}/slime:${ROOT_DIR}/Megatron-LM:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export RAY_memory_usage_threshold=0.99
export RAY_TMPDIR="${ROOT_DIR}/ray_out"
rm -rf "$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   --custom-generate-function-path generate_mixed_opd.generate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

#### cleanup
ray stop --force
pkill -9 ray || true
pkill -9 python || true
