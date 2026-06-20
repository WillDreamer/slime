#!/bin/bash
#
# Qwen3-8B (dense) IFBench RL training (instruction-following) -- NO reward model.
#
# Train on: allenai/IF_multi_constraints_upto5 (IFEval-G reward, rm_type="multi")
# Eval on:  IFBench_eval.jsonl (IFBench reward, rm_type="ifbench")
#
# Init from the multi-stage SFT checkpoint
#   /xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT
# Hyperparams aligned with the 30B-A3B ifbench reference (GSPO, TIS, KL low-var).
# Run inside the slime_ifbench container.
#
# This is the baseline (rule-based IFEval-G reward only). To additionally use the
# Skywork generative reward model judge, use run-qwen3-8B_ifbench_rm.sh instead.
#

# Cleanup stale processes
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

export PYTHONUNBUFFERED=1
export WANDB_API_KEY=09286f9b4dcf8784b832ad623eb07a6d5541f59a

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# models/ lives one level up (scripts/models/qwen3-8B.sh)
source "${SCRIPT_DIR}/../models/qwen3-8B.sh"

export RAY_TMPDIR=/xuanwu-tank/north/xw27/ray_temp_scai     # local disk, not /xuanwu-tank for heavy IO
rm -rf "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"

# === Checkpoint paths ===
# HF base model: needed by sglang rollout engine to load weights / tokenizer.
HF_CKPT="/xuanwu-tank/center/whx/Qwen3-8B-Base"
# SFT init: Megatron .distcp checkpoint to start GRPO from (latest iter 380).
# When --load points at an empty dir, slime falls back to --ref-load as the
# initial weights AND as the frozen reference model for KL. --finetune resets
# the optimizer/iteration. Required because --use-kl-loss is on.
REF_CKPT="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT"
SLIME_CKPT_DIR="/xuanwu-tank/north/xw27/model/Qwen3-8B_ifbench_slime_scai"
ROLLOUT_DEBUG_DIR="${SLIME_CKPT_DIR}/rollout_debug"
mkdir -p "${SLIME_CKPT_DIR}" "${ROLLOUT_DEBUG_DIR}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_CKPT}"
   --load "${SLIME_CKPT_DIR}/"
   --save "${SLIME_CKPT_DIR}/"
   --save-interval 30
   --save-retain-interval 60
   --finetune
   --start-rollout-id 0
)

# === Training data ===
# Per-row metadata.rm_type="multi" -> dispatches to IFEvalG reward (ifeval.py)
ROLLOUT_ARGS=(
   --prompt-data /xuanwu-tank/north/xw27/multi/slime_scai/examples/ifbench/IF_multi_constraints_upto5_ifbench_en.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type multi
   --num-rollout 500
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 128
   --balance-data
)

# === Evaluation ===
EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data ifbench /xuanwu-tank/north/xw27/multi/slime_scai/examples/ifbench/IFBench_eval.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 8192
   --eval-top-p 1
)

# === Performance / parallelism (8B dense; no MoE/EP) ===
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 18432
)

# === GSPO advantage + KL + TIS (train/inference mismatch correction) ===
GRPO_ARGS=(
   --advantage-estimator gspo
   --use-kl-loss
   --kl-loss-coef 0.1
   --kl-loss-type low_var_kl
   --entropy-coef 0.01
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# === Optimizer ===
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-7
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

# === Wandb ===
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-ifbench
   --wandb-group qwen3-8B-ifbench
)

# === SGLang rollout engine ===
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
)

# === Misc ===
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   --start-rollout-id 0
   # --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
)

# Launch ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir "$RAY_TMPDIR"

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is not set; wandb logging will fail unless ~/.netrc is configured."
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
