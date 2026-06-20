#!/bin/bash
# Evaluate Qwen3-8B-Base-Math-SeaSFT-Search via whx's existing server :30012,
# in parallel with the math model's run on GPU3 (:30012's GPU is idle).
#
# :30012 has --reasoning-parser baked in; inspect tasks go through
# run_inspect_api.py which sends {"separate_reasoning": false} per request.
# ifbench falls back to reasoning_content; the search benchmark uses /generate
# (parser never applies). BROWSECOMP IS DEFERRED: qwen_client can't inject
# extra_body, so it runs after the math model frees GPU3 (parser-free server).
set -u
EVAL_ROOT="/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai"
PYBIN="/xuanwu-tank/north/xw27/envs/sglang_env/bin/python"
IPY="${EVAL_ROOT}/.venv_inspect/bin/python"

CKPT="/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420"
NAME="Qwen3-8B-Base-Math-SeaSFT-Search"
# :30012 serves the model under its full path; the inspect provider verifies
# the name against the server's model list and AUTO-STARTS a new server on a
# mismatch — so inspect must request the served (path) name, not our alias.
SERVED_NAME="${CKPT}"
PORT=30012
BASE_URL="http://127.0.0.1:${PORT}/v1"
export SGLANG_BASE_URL="${BASE_URL}" SGLANG_API_KEY=dummy
RESULTS="${EVAL_ROOT}/results/${NAME}"
LOGS="${EVAL_ROOT}/logs/${NAME}"
ILOGS="${RESULTS}/inspect_logs"
mkdir -p "${RESULTS}" "${LOGS}" "${ILOGS}"

run_api () {  # task, extra args...
    local task="$1"; shift
    echo ">>> [$(date +%H:%M:%S)] running ${task}"
    "${IPY}" "${EVAL_ROOT}/benchmarks/inspect/run_inspect_api.py" \
        --task "${task}" --model-name "${SERVED_NAME}" --base-url "${BASE_URL}" \
        --log-dir "${ILOGS}" "$@"
}

run_api inspect_evals/gpqa_diamond --temperature 0 --task-arg cot=true
run_api inspect_evals/aime2025     --temperature 0
run_api inspect_evals/ifeval       --temperature 0
run_api inspect_evals/mmlu_5_shot  --temperature 0.7 --task-arg cot=true

echo ">>> [$(date +%H:%M:%S)] running ifbench"
"${PYBIN}" "${EVAL_ROOT}/benchmarks/ifbench/eval_ifbench.py" \
    --base-url "${BASE_URL}" --model "${NAME}" \
    --data /xuanwu-tank/north/xw27/multi/slime_scai/examples/ifbench/IFBench_eval.jsonl \
    --output "${RESULTS}/ifbench.json" \
    --slime-root /xuanwu-tank/north/xw27/multi/slime_scai \
    --max-tokens 8192 --temperature 0.0 --concurrency 24

echo ">>> [$(date +%H:%M:%S)] running search"
"${PYBIN}" "${EVAL_ROOT}/benchmarks/search/eval_search.py" \
    --base-url "http://127.0.0.1:${PORT}" \
    --retriever-url "http://127.0.0.1:8500/retrieve" \
    --tokenizer "${CKPT}" \
    --data /xuanwu-tank/center/whx/Search_data/test.parquet \
    --output "${RESULTS}/search_nq.json" \
    --max-turns 2 --topk 3 --max-new-tokens 2048 --concurrency 48

run_api inspect_evals/tau2_retail  --temperature 0 --message-limit 50
run_api inspect_evals/tau2_airline --temperature 0 --message-limit 50
run_api inspect_evals/tau2_telecom --temperature 0 --message-limit 50

echo ">>> [$(date +%F' '%T)] SEARCH-MODEL SUITE (via 30012) DONE — browsecomp deferred to GPU3"
