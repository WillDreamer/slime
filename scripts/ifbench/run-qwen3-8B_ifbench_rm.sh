#!/bin/bash
#
# Qwen3-8B (dense) IFBench RL training WITH the Skywork reward-model judge.
#
# Identical to run-qwen3-8B_ifbench.sh except it enables --rm-judge-url /
# --rm-judge-threshold, so responses that already pass the rule-based IFEval-G
# check (verified > 0) are additionally scored by the Skywork reward model:
#   verified > 0 and rm_score >  threshold -> final = verified + 1.0
#   verified > 0 and rm_score <= threshold -> final = verified - 0.5
#   verified <= 0                          -> final = verified (unchanged)
# (see slime/rollout/rm_hub/__init__.py::_apply_rm_judge_reward)
#
# PREREQUISITE: start the reward server FIRST and wait until it is ready:
#   bash launch_reward_model.sh         # in a separate shell / node
# Then point RM_JUDGE_URL at it (default assumes same host, port 30000).
#

# Cleanup stale processes (does NOT kill the reward-model sglang server if it is
# on a different node; on the same node, start the RM server AFTER this cleanup).
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

# === Reward-model judge endpoint ===
# Skywork-Reward-V2-Llama-3.1-8B is served on a SEPARATE node: scai7
# (131.179.168.120), GPUs 2,3 / TP=2, bound to 0.0.0.0:30000. Verified reachable
# from this training box (scai6): /health -> 200, /classify -> {"embedding":[score]}.
# Use the IP (not the short hostname) — the trainer runs inside the slime
# container via `ray job submit` and may not resolve `scai7`.
RM_JUDGE_URL="${RM_JUDGE_URL:-http://131.179.168.120:30000/classify}"
RM_JUDGE_THRESHOLD="${RM_JUDGE_THRESHOLD:-0.0}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../models/qwen3-8B.sh"

export RAY_TMPDIR=/xuanwu-tank/north/xw27/ray_temp_scai
rm -rf "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"

# === Checkpoint paths ===
HF_CKPT="/xuanwu-tank/center/whx/Qwen3-8B-Base"
# Start RL from the iter-300 Tau checkpoint (Megatron torch_dist dir; latest_checkpointed_iteration.txt=300).
# NOTE: point --ref-load at this MEGATRON dir, not the hf_iter_0000300/ HF export — raw mode loads torch_dist.
REF_CKPT="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau"
SLIME_CKPT_DIR="/xuanwu-tank/north/xw27/model/Qwen3-8B_ifbench_rm_slime_scai"
ROLLOUT_DEBUG_DIR="${SLIME_CKPT_DIR}/rollout_debug"
mkdir -p "${SLIME_CKPT_DIR}" "${ROLLOUT_DEBUG_DIR}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_CKPT}"
   --load "${SLIME_CKPT_DIR}/"
   --save "${SLIME_CKPT_DIR}/"
   --save-interval 30
   --save-retain-interval 210   # must be a multiple of save-interval (Megatron assert); 210 = 7*30
   # --finetune / --start-rollout-id intentionally omitted so that when SLIME_CKPT_DIR holds a
   # checkpoint, slime resumes cleanly (loads optimizer state + continues rollout_id). When the
   # dir is empty, slime_validate_args auto-sets finetune=True / start_rollout_id=0 and loads
   # --ref-load instead (a fresh start). So this works for both resume and fresh launches.
   # Dump full per-rollout rollout/train tensors for inspection (writes <dir>/rollout_data/<id>.pt).
   --dump-details "${ROLLOUT_DEBUG_DIR}"
)

# === Training data ===
# rm_type="multi" -> IFEvalG rule reward; with --rm-judge-url set, passing
# samples are re-scored by the Skywork RM (see rm_hub/__init__.py).
ROLLOUT_ARGS=(
   --prompt-data /xuanwu-tank/north/xw27/multi/slime_scai/examples/ifbench/IF_multi_constraints_upto5_ifbench_en.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type multi
   --num-rollout 1500
   --rollout-batch-size 256
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   # Filter out over-long prompts at dataset load so prompt+response stays within the
   # model's 32768 ctx window. Without this, a long sample (dataset has prompts up to
   # ~250k tokens) made sglang reject the request (400 -> circuit-breaker 503) and the
   # rollout manager exhausted its 60 retries, crashing train.py with exit 1.
   --rollout-max-prompt-len 24000
   --rollout-max-context-len 32768
   --rollout-temperature 1
   --global-batch-size 2048
   --balance-data

   # === Reproducibility: pin seeds ===
   # --seed: Megatron init/dropout (default 1234, arguments.py:754).
   # --rollout-seed: seeds the data shuffle (default 42, arguments.py:332 ->
   # data_source.py:83). Pinned explicitly so init + prompt order are recorded,
   # not silently defaulted.
   # NOTE: generation is still stochastic here (temperature 1). A per-sample
   # sampling_seed (rollout_seed + i) *is* wired to sglang
   # (sglang_rollout.py:289-291 train, :514-516 eval) but only when
   # --sglang-enable-deterministic-inference is set, which we do NOT enable
   # (batch-invariant ops carry a throughput cost). So rollout *trajectories*
   # are not bit-exact; data order and init are.
   --seed 1234
   --rollout-seed 42

   # === RM judge (Skywork-Reward-V2-Llama-3.1-8B) ===
   --rm-judge-url "${RM_JUDGE_URL}"
   --rm-judge-threshold "${RM_JUDGE_THRESHOLD}"
)

# === Evaluation ===
EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data ifbench /xuanwu-tank/north/xw27/multi/slime_scai/examples/ifbench/IFBench_eval.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 16384
   --eval-top-p 1
)

# === Performance / parallelism (8B dense; no MoE/EP) ===
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # 18432 -> 16384: peak train-phase activation alloc was ~10.4GB with only ~9-11GB
   # free, OOMing at step 297. Lower peak to restore margin alongside expandable_segments.
   --max-tokens-per-gpu 16384
)

# === GSPO advantage + KL + TIS ===
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# === Optimizer ===
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   # Take LR-scheduler params from these args instead of the checkpoint. Needed because
   # changing n-samples-per-prompt (16->8) changed the scheduler's total-iteration count,
   # which mismatches the value stored in iter-49 (2048000 vs 1024000) and trips Megatron's
   # assert on resume. LR is constant 1e-6 so overriding the schedule is a no-op; Adam
   # momentum state is still loaded from the checkpoint.
   --override-opt_param-scheduler
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# === Wandb ===
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-ifbench
   --wandb-group qwen3-8B-ifbench-rm
)

# === SGLang rollout engine ===
# RM runs on a SEPARATE node (scai7), so the trainer keeps all 8 local GPUs to
# itself — mem-fraction restored to 0.7 (the same-node layout used 0.6 to share
# GPUs 2,3 with the RM). Drop back to 0.6 only if you co-locate the RM here.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   # 0.8 -> 0.7: frees ~14GB of SGLang static reservation for the trainer side,
   # adding train-phase headroom after the step-297 OOM (with max-tokens 16384).
   --sglang-mem-fraction-static 0.7
)

# === Misc ===
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
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
