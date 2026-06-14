#!/bin/bash
# Run lm-evaluation-harness against a locally-served sglang model.
#
# Usage:
#   ./run_eval.sh <MODEL_NAME> <PORT> <TASK> [HOST] [CONCURRENT]
#
# Examples:
#   ./run_eval.sh Qwen3-30B-A3B_base_math 8050 math500_slime
#   ./run_eval.sh final_search             8051 mmlu_pro_slime 131.179.168.120 16
#   # ↑ lower concurrency for long-tail tasks like mmlu/mmlu_pro to avoid timeouts
#
# Available tasks: math500_slime, mmlu_slime, mmlu_pro_slime,
#                  gpqa_slime, aime24_slime, aime25_slime
#
# Results land in: /xuanwu-tank/north/hhzhang/slime/eval/results_<MODEL>/<MODEL>/
# (network-attached storage — same place base / base_math results live)

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <MODEL_NAME> <PORT> <TASK> [HOST] [CONCURRENT]" >&2
    exit 1
fi

MODEL=$1
PORT=$2
TASK=$3
HOST=${4:-131.179.168.120}
CONCURRENT=${5:-35}

URL="http://${HOST}:${PORT}/v1"
TASKS_DIR="/data1/hhzhang/slime/eval/slime_tasks"
OUT_DIR="/xuanwu-tank/north/hhzhang/slime/eval/results_${MODEL}"
LM_EVAL="/xuanwu-tank/north/xw27/envs/sglang_env/bin/lm_eval"

# lm-eval's local-chat-completions backend uses the openai client under the
# hood, which requires these env vars to be set even though we override the
# base url via --model_args. Setting them to dummy/local values is safe.
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export OPENAI_API_BASE="${URL}"
export OPENAI_BASE_URL="${URL}"

# Sanity check the server is up before kicking off a long eval
if ! curl -s --max-time 3 "${URL}/models" | grep -q '"id"'; then
    echo "[error] sglang server not responding at ${URL}/models" >&2
    echo "[hint] start one with: python -m sglang.launch_server --model-path ... --served-model-name ${MODEL} --port ${PORT}" >&2
    exit 2
fi

echo "[info] model=${MODEL}  port=${PORT}  task=${TASK}"
echo "[info] output -> ${OUT_DIR}"

"${LM_EVAL}" \
    --model local-chat-completions \
    --model_args "model=${MODEL},base_url=${URL}/chat/completions,num_concurrent=${CONCURRENT},max_retries=3,tokenized_requests=False,tokenizer=${MODEL},max_length=32768" \
    --apply_chat_template \
    --batch_size auto \
    --output_path "${OUT_DIR}" \
    --gen_kwargs temperature=0 \
    --log_samples \
    --include_path "${TASKS_DIR}" \
    --tasks "${TASK}"
