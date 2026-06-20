#!/bin/bash
#
# Launch the Skywork generative reward-model judge as an sglang /classify server.
#
# slime's ifbench reward path (slime/rollout/rm_hub/__init__.py::_query_rm_judge)
# POSTs {"text": "<llama3.1-formatted prompt+response>"} to this server's
# /classify endpoint and reads back a scalar score from result["embedding"][0].
# That score is combined with the rule-based IFEval-G reward by
# _apply_rm_judge_reward(). See ../ifbench/README.md (REWARD_MODEL.md) for details.
#
# The RL training script (run-qwen3-8B_ifbench_rm.sh) points --rm-judge-url at
# "http://<HOST>:<PORT>/classify". Start THIS server first, wait until it is
# ready, then start training.
#
# === GPU placement ===
# The RL job is colocated across all 8 GPUs, so the reward model needs its own
# space. Two supported layouts:
#   (A) Separate node (recommended): run this on a second machine, set HOST to a
#       reachable address (0.0.0.0 to listen on all interfaces) and point the RL
#       script's RM_JUDGE_URL at that host.
#   (B) Same node, dedicated GPU: pin this to one GPU via RM_GPUS and lower the
#       training engine's --sglang-mem-fraction-static so the shared GPU fits both.
#       The companion RL script already drops it to 0.6 for this reason.
#
# All knobs are overridable via environment variables.

set -ex

# Skywork-Reward-V2-Llama-3.1-8B. Pulled from HF on first run if not cached
# (not present locally as of writing). Override with a local path if you have one.
MODEL_PATH="${MODEL_PATH:-Skywork/Skywork-Reward-V2-Llama-3.1-8B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
MEM_FRACTION="${MEM_FRACTION:-0.3}"   # small footprint: shares GPUs 2,3 with the trainer
TP="${TP:-2}"                          # tensor-parallel across the two GPUs below
# Which GPU(s) this server runs on. Default: GPUs 2 and 3 (shared with training).
RM_GPUS="${RM_GPUS:-2,3}"

export CUDA_VISIBLE_DEVICES="${RM_GPUS}"
echo "Launching reward model on GPU(s) ${CUDA_VISIBLE_DEVICES} at ${HOST}:${PORT}"

# --is-embedding turns the causal LM head into a reward/score head and exposes
# the /classify endpoint that slime queries.
python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --is-embedding \
    --host "${HOST}" \
    --port "${PORT}" \
    --context-length "${CONTEXT_LENGTH}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --tp "${TP}"
