#!/bin/bash
# 3-teacher on-policy distillation (pure OPD, task reward = 0).
#
#   student : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF  (megatron torch_dist, --load)
#   teachers (the specialists from the math2sea / sea2tau / tau2if experiments):
#     * math   : MultiStageRL/Qwen3-8B-Base-Math                              -> SGLang server (:13140)
#     * search : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search               -> SGLang server (:13141)
#     * tau    : MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau    -> SGLang server (:13142)
#
# One student rolls out over a MIXED math+search+tau prompt stream; each trajectory is scored by the
# teacher that OWNS its domain (domain-routing, see opd_multi_teacher.py). Goal: distill all three
# specialties back into the end-of-chain IF model, which has drifted from math, search AND tau.
#
# This box ONLY trains the student (colocate actor + rollout engine on all 8 GPUs). The three teacher
# servers, the Search retriever, and the Tau user-simulator all live on OTHER nodes / spare GPUs and
# are reached over HTTP (see "REMOTE SERVICES"). To run a teacher on a spare local GPU instead, just
# launch it there (commands below) and point its URL at 127.0.0.1.
#
# usage: bash examples/on_policy_distillation/multi_teacher/run-qwen3-8B-opd-multi.sh
set -ex

ROOT_DIR=/data1/whx
MODEL_ROOT=/xuanwu-tank/center/whx
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${ROOT_DIR}/slime"   # train.py / tools / examples paths resolve from the slime root

# ----------------------------------------------------------------------------
# REMOTE SERVICES
# ----------------------------------------------------------------------------
# Three teacher SGLang servers (HF format). The HF copies already exist under each ckpt dir, so each
# teacher is a one-liner (run on a remote node, or on a spare local GPU with CUDA_VISIBLE_DEVICES set):
#   # math  (:13140)
#   python3 -m sglang.launch_server --model-path ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math/hf_iter_0000300 \
#       --host 0.0.0.0 --port 13140 --tp 1 --context-length 16384 --mem-fraction-static 0.8
#   # search (:13141)
#   python3 -m sglang.launch_server --model-path ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420 \
#       --host 0.0.0.0 --port 13141 --tp 1 --context-length 16384 --mem-fraction-static 0.8
#   # tau    (:13142)
#   python3 -m sglang.launch_server --model-path ${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau/hf_iter_0000300 \
#       --host 0.0.0.0 --port 13142 --tp 1 --context-length 16384 --mem-fraction-static 0.8
# (Scoring-only servers are light; several can share one H200 by lowering --mem-fraction-static.)
export OPD_TEACHER_URL_MATH=${OPD_TEACHER_URL_MATH:-http://127.0.0.1:13140/generate}
export OPD_TEACHER_URL_SEARCH=${OPD_TEACHER_URL_SEARCH:-http://127.0.0.1:13141/generate}
export OPD_TEACHER_URL_TAU=${OPD_TEACHER_URL_TAU:-http://127.0.0.1:13142/generate}
export OPD_TEACHER_TIMEOUT=${OPD_TEACHER_TIMEOUT:-600}

# Tau user-simulator (remote, OpenAI-compatible) — same endpoint the tau2if run used.
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://131.179.168.120:8000/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
# Search retriever: generate_with_search_tools_qwen_sft_no_drift.py targets
# http://131.179.168.117:8000/retrieve (same as the sea2tau run). Just make it reachable.
RETRIEVE_URL=${RETRIEVE_URL:-http://131.179.168.117:8000/retrieve}
export SEARCH_R1_STRIP_THINK=0

# ---- fail fast: verify every remote dependency is reachable before startup ----
for url in "${OPD_TEACHER_URL_MATH%/generate}" "${OPD_TEACHER_URL_SEARCH%/generate}" "${OPD_TEACHER_URL_TAU%/generate}"; do
  until curl -sf "${url}/health_generate" > /dev/null; do echo "waiting for teacher @ ${url} ..."; sleep 5; done
  echo "teacher reachable @ ${url}"
done
curl -sf -m 10 "${OPENAI_API_BASE}/models" > /dev/null && echo "tau user-sim reachable @ ${OPENAI_API_BASE}" \
  || { echo "ERROR: tau user-sim NOT reachable @ ${OPENAI_API_BASE}" >&2; exit 1; }
curl -sf -m 10 -X POST "${RETRIEVE_URL}" -H 'Content-Type: application/json' \
  -d '{"queries":["ping"],"topk":1,"return_scores":false}' > /dev/null && echo "retriever reachable @ ${RETRIEVE_URL}" \
  || { echo "ERROR: search retriever NOT reachable @ ${RETRIEVE_URL}" >&2; exit 1; }

# ----------------------------------------------------------------------------
# Data: build the mixed (math + search pre-templated + tau index) dataset once.
# --per-domain-target balances the three pools (dapo ~17k / nq_hotpotqa ~170k / retail ~500)
# to ~1:1:1 so every rollout batch keeps all three teachers active.
# ----------------------------------------------------------------------------
MIXED_DATA=${MODEL_ROOT}/MultiStageRL/opd_mixed3/train.jsonl
if [ ! -f "${MIXED_DATA}" ]; then
  python3 "${SCRIPT_DIR}/prepare_opd_mixed_data.py" \
    --hf ${MODEL_ROOT}/Qwen3-8B-Base \
    --math-jsonl     ${MODEL_ROOT}/dapo-math-17k/dapo-math-17k.jsonl \
    --search-parquet ${ROOT_DIR}/Search-R1/data/nq_hotpotqa_train/train.parquet \
    --tau-jsonl      ${ROOT_DIR}/tau-bench/retail_train_tasks.jsonl \
    --per-domain-target 3000 \
    --out "${MIXED_DATA}"
fi

# ----------------------------------------------------------------------------
# GPUs (student training only; teachers are outside this box / on spare GPUs)
# ----------------------------------------------------------------------------
GPU_LIST=(0 1 2 3 4 5 6 7)
export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
NUM_GPUS=${#GPU_LIST[@]}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0

source "${SCRIPT_DIR}/../../../scripts/models/qwen3-8B.sh"   # -> MODEL_ARGS

ROLLOUT_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=256

STUDENT=${MODEL_ROOT}/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3-8B-Base                # tokenizer + config
   --ref-load      ${STUDENT}                                 # (KL-loss-coef=0, so ref is inert; mirror student)
   --load          ${STUDENT}                                 # STUDENT (init point, NOT a resume)
   # STUDENT is a "release" megatron ckpt (weights only, no optimizer/rng/scheduler). Fresh OPD run
   # => load weights only and reset iteration -> 0: --no-load-optim/--no-load-rng skip the (absent)
   # optimizer+rng, --finetune treats it as a fine-tune start (iteration 0) instead of a resume.
   --no-load-optim
   --no-load-rng
   --finetune
   --save          ${MODEL_ROOT}/MultiStageRL/OPD/Qwen3-8B-Base-...-Tau-IF_OPD_MathSearchTau/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${MIXED_DATA}
   --input-key prompt          # NOTE: NO --apply-chat-template (math+search pre-templated; tau=index)
   --rollout-shuffle           # mixes math+search+tau within every rollout batch -> all teachers active each step
   --num-rollout 300
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 8    # reward=0 -> grouping is neutral; can lower to save rollout cost
   --rollout-max-response-len 8192   # max(math 8192, search 4096, tau 2048)
   --rollout-temperature 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

# OPD reward: per-sample teacher logprobs (domain-routed to math/search/tau server), task reward = 0.
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
   --max-tokens-per-gpu 12288        # >= longest single sequence / CP; math resp up to 8192 (+ short prompt)
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
# multi_teacher (custom fns) + both agentic example dirs (search/tau native generates) on PYTHONPATH.
# (math uses the stock slime rollout, so no extra dir is needed for it.)
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
