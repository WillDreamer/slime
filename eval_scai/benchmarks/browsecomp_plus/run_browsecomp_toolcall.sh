#!/bin/bash
# BrowseComp-Plus with the SEARCH-TASK tool-call agent format.
# Same official pipeline as run_browsecomp.sh, with ONE substitution: the agent
# loop uses browsecomp_toolcall_client.py (<tool_call>/<tool_response>/<answer>
# text format driven against the sglang server) instead of qwen_client.py
# (native function-calling). Dataset, retriever corpus/index and judge are the
# official BrowseComp-Plus ones (BCP_ROOT), so metrics are leaderboard-comparable.
#
# Two phases (BCP_PHASE = predict | judge | all; default all):
#   predict : 1. official retriever  searcher/search_r1_server.py (BM25)
#             2. agent loop           browsecomp_toolcall_client.py -> sglang /generate
#   judge   : 3. LLM-as-judge         scripts_evaluation/evaluate_local.py
#                (chat/completions against BCP_JUDGE_BASE_URL)
#
# The split lets the SAME GPU serve the agent model during `predict`, then be
# freed and reloaded with the judge model (e.g. Qwen3-32B) for `judge`.
#
# Use this for checkpoints trained on the tool-call text template (willhx/*-Search).
# Usage: CKPT=<hf ckpt> BCP_PHASE=predict bash run_browsecomp_toolcall.sh
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../env.sh"

BCP_PHASE="${BCP_PHASE:-all}"
cd "${BCP_ROOT}"
VPY="${BCP_VENV}/bin/python"
RUN_DIR="runs/bm25/${CKPT_NAME}"
mkdir -p "${RUN_DIR}"

run_predict() {
    [ -x "${VPY}" ] || { echo "BCP venv missing at ${BCP_VENV}" >&2; exit 1; }
    [ -d indexes/bm25 ] || { echo "BM25 index missing at ${BCP_ROOT}/indexes/bm25" >&2; exit 1; }
    export JAVA_HOME PATH="${JAVA_HOME}/bin:${PATH}"

    # --- 1. official retriever (/retrieve) ------------------------------------
    _ret_up() { curl -s -m 3 "http://127.0.0.1:${BCP_RETRIEVER_PORT}/retrieve" \
        -H "Content-Type: application/json" -d '{"query":"ping"}' 2>/dev/null | grep -q result; }
    if ! _ret_up; then
        echo "[bcp-tc] starting search_r1_server (BM25) on :${BCP_RETRIEVER_PORT}"
        nohup "${VPY}" searcher/search_r1_server.py \
            --searcher-type bm25 \
            --index-path indexes/bm25 \
            --port "${BCP_RETRIEVER_PORT}" \
            --k "${BCP_TOPK}" \
            > "${LOG_DIR}/bcp_retriever.log" 2>&1 &
        echo $! > "${LOG_DIR}/bcp_retriever.pid"
        for i in $(seq 1 60); do _ret_up && break; sleep 5; done
        _ret_up || { echo "[bcp-tc] retriever failed; see ${LOG_DIR}/bcp_retriever.log" >&2; exit 1; }
    fi

    # --- 2. tool-call agent loop ----------------------------------------------
    QUERY_TSV="topics-qrels/queries.tsv"
    if [ -n "${BCP_LIMIT}" ]; then
        QUERY_TSV="${RUN_DIR}/queries_limit${BCP_LIMIT}.tsv"
        head -n "${BCP_LIMIT}" topics-qrels/queries.tsv > "${QUERY_TSV}"
    fi
    "${PYBIN}" "${SCRIPT_DIR}/browsecomp_toolcall_client.py" \
        --query "${QUERY_TSV}" \
        --model "${CKPT_NAME}" \
        --model-server "${BCP_MODEL_SERVER}" \
        --retriever-url "http://127.0.0.1:${BCP_RETRIEVER_PORT}/retrieve" \
        --tokenizer "${CKPT}" \
        --output-dir "${RUN_DIR}" \
        --max-turns "${BCP_MAX_TURNS}" \
        --topk "${BCP_TOPK}" \
        --max-new-tokens "${BCP_MAX_TOKENS}" \
        --max-response-tokens "${BCP_MAX_RESPONSE_TOKENS}" \
        --concurrency "${BCP_CONCURRENCY}" \
        2>&1 | tee "${LOG_DIR}/bcp_agent.log"
    echo "[bcp-tc] predictions written to ${BCP_ROOT}/${RUN_DIR}"
}

run_judge() {
    [ -x "${VPY}" ] || { echo "BCP venv missing at ${BCP_VENV}" >&2; exit 1; }
    [ -f data/browsecomp_plus_decrypted.jsonl ] || { echo "dataset not decrypted" >&2; exit 1; }
    OPENAI_BASE_URL="${BCP_JUDGE_BASE_URL}" OPENAI_API_KEY="${BCP_JUDGE_API_KEY}" \
    "${VPY}" scripts_evaluation/evaluate_local.py \
        --input_dir "${RUN_DIR}" \
        --ground_truth data/browsecomp_plus_decrypted.jsonl \
        --qrel_evidence topics-qrels/qrel_evidence.txt \
        --model "${BCP_JUDGE_MODEL}" \
        --eval_dir "${RESULTS_DIR}/browsecomp_plus_evals" \
        2>&1 | tee "${LOG_DIR}/bcp_judge.log"
    echo "[bcp-tc] evals in ${RESULTS_DIR}/browsecomp_plus_evals"
}

case "${BCP_PHASE}" in
    predict) run_predict ;;
    judge)   run_judge ;;
    all)     run_predict; run_judge ;;
    *) echo "BCP_PHASE must be predict|judge|all" >&2; exit 1 ;;
esac
