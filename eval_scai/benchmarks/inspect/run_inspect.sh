#!/bin/bash
# Run the inspect-ai benchmarks against the served checkpoint:
#   gpqa | mmlu | aime | ifeval | tau2
# Usage: CKPT=<hf ckpt> bash run_inspect.sh <benchmark> [extra inspect args...]
#
# Tasks come from the local inspect_evals checkout installed in sglang_env
# (/xuanwu-tank/north/xw27/multi/slime_scai/eval/inspect_evals, pkg inspect_evals
# 0.3.104). The model is reached through inspect's sglang provider pointed at
# the already-running eval server (SGLANG_BASE_URL), NOT a new server.
# Configs mirror the previous runs in slime/eval/script/run.sh.

set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../env.sh"

BENCH="${1:?usage: run_inspect.sh <gpqa|mmlu|aime|ifeval|tau2> [extra args]}"
shift || true

export SGLANG_BASE_URL="${EVAL_BASE_URL}"
export SGLANG_API_KEY="dummy"
export INSPECT_LOG_DIR="${RESULTS_DIR}/inspect_logs"
mkdir -p "${INSPECT_LOG_DIR}"

MODEL="sglang/${CKPT_NAME}"
# --timeout: long CoT generations on shared servers can take >10 min/request;
# --max-retries 3: transient timeouts shouldn't kill samples (prior config's
# max_retries=0 turned one APITimeoutError into a failed eval).
COMMON=(--model "${MODEL}"
        --max-connections "${INSPECT_MAX_CONNECTIONS}"
        --max-tokens "${INSPECT_MAX_TOKENS}"
        --timeout "${INSPECT_TIMEOUT}"
        --max-retries 3)

case "${BENCH}" in
  gpqa)
    # GPQA-Diamond, 198 questions, chain-of-thought, temp 0 (matches prior runs)
    "${INSPECT_BIN}" eval inspect_evals/gpqa_diamond "${COMMON[@]}" \
        --temperature "${INSPECT_TEMPERATURE}" --epochs 1 -T cot=true "$@"
    ;;
  mmlu)
    # MMLU 5-shot with CoT, temp 0.7 (matches prior runs). Always cot=true.
    "${INSPECT_BIN}" eval inspect_evals/mmlu_5_shot "${COMMON[@]}" \
        --temperature "${MMLU_TEMPERATURE}" -T cot=true "$@"
    ;;
  aime)
    # AIME 2025, 30 problems, temp 0 (matches prior runs)
    "${INSPECT_BIN}" eval inspect_evals/aime2025 "${COMMON[@]}" \
        --temperature "${INSPECT_TEMPERATURE}" "$@"
    ;;
  ifeval)
    # IFEval (google/IFEval, 541 prompts), official strict/loose metrics
    "${INSPECT_BIN}" eval inspect_evals/ifeval "${COMMON[@]}" \
        --temperature "${INSPECT_TEMPERATURE}" "$@"
    ;;
  tau2)
    # Tau2 (sierra-research), inspect-native port with vendored data.
    # The evaluated model also plays the user simulator (self-play) unless
    # you pass e.g. -T user_model=openai/<judge> through extra args.
    # WARNING: very token-hungry; use TAU2_LIMIT for smoke tests.
    EXTRA=()
    [ -n "${TAU2_LIMIT}" ] && EXTRA+=(--limit "${TAU2_LIMIT}")
    [ -n "${TAU2_MESSAGE_LIMIT}" ] && EXTRA+=(-T "message_limit=${TAU2_MESSAGE_LIMIT}")
    for domain in ${TAU2_DOMAINS}; do
        "${INSPECT_BIN}" eval "inspect_evals/tau2_${domain}" "${COMMON[@]}" \
            --temperature "${TAU2_TEMPERATURE}" "${EXTRA[@]}" "$@"
    done
    ;;
  *)
    echo "Unknown benchmark: ${BENCH}" >&2; exit 1
    ;;
esac
