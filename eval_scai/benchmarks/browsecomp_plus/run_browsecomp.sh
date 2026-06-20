#!/bin/bash
# BrowseComp-Plus (texttron/BrowseComp-Plus) against the served checkpoint.
# Pipeline (per docs/qwen.md, adapted to sglang + BM25 retriever):
#   1. BM25 MCP retrieval server  (searcher/mcp_server.py, port ${BCP_MCP_PORT})
#   2. Agent loop                 (search_agent/qwen_client.py -> our sglang server)
#   3. LLM-as-judge scoring       (scripts_evaluation/evaluate_with_openai.py)
#
# Judge note: the official leaderboard judge is Qwen3-32B (evaluate_run.py,
# vllm). To stay self-contained we default to judging with an OpenAI-compatible
# endpoint via evaluate_with_openai.py (env-configurable; defaults to the
# served model itself — set BCP_JUDGE_* for a stronger/leaderboard judge).
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../env.sh"

cd "${BCP_ROOT}"
VPY="${BCP_VENV}/bin/python"
[ -x "${VPY}" ] || { echo "BCP venv missing — run logs/bcp_setup.log steps first" >&2; exit 1; }
[ -f data/browsecomp_plus_decrypted.jsonl ] || { echo "dataset not decrypted" >&2; exit 1; }
[ -d indexes/bm25 ] || { echo "BM25 index missing (scripts_build_index/download_indexes.sh)" >&2; exit 1; }

RUN_DIR="runs/bm25/${CKPT_NAME}"
mkdir -p "${RUN_DIR}"

# --- 1. BM25 MCP server (needs JDK 21 for pyserini; JAVA_HOME set in env.sh) --
# The server mounts at /mcp regardless of transport (mcp_server.py:152).
_mcp_up() { [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${BCP_MCP_PORT}/mcp" 2>/dev/null)" != "000" ]; }
if ! _mcp_up; then
    echo "[bcp] starting BM25 MCP server on :${BCP_MCP_PORT}"
    nohup "${VPY}" searcher/mcp_server.py \
        --searcher-type bm25 \
        --index-path indexes/bm25 \
        --port "${BCP_MCP_PORT}" \
        > "${LOG_DIR}/bcp_mcp.log" 2>&1 &
    echo $! > "${LOG_DIR}/bcp_mcp.pid"
    for i in $(seq 1 24); do _mcp_up && break; sleep 5; done
    _mcp_up || { echo "[bcp] MCP server failed; see ${LOG_DIR}/bcp_mcp.log" >&2; exit 1; }
fi

# --- 2. Agent loop over the 830 queries --------------------------------------
# qwen_client takes a TSV of (qid, query) and auto-resumes from RUN_DIR;
# BCP_LIMIT slices the TSV for smoke tests.
QUERY_TSV="topics-qrels/queries.tsv"
if [ -n "${BCP_LIMIT}" ]; then
    QUERY_TSV="${RUN_DIR}/queries_limit${BCP_LIMIT}.tsv"
    head -n "${BCP_LIMIT}" topics-qrels/queries.tsv > "${QUERY_TSV}"
fi
"${VPY}" search_agent/qwen_client.py \
    --model "${CKPT_NAME}" \
    --model-server "${EVAL_BASE_URL}" \
    --mcp-url "http://127.0.0.1:${BCP_MCP_PORT}/mcp" \
    --max_tokens "${BCP_MAX_TOKENS}" \
    --query "${QUERY_TSV}" \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${LOG_DIR}/bcp_agent.log"

# --- 3. Judge ------------------------------------------------------------------
OPENAI_BASE_URL="${BCP_JUDGE_BASE_URL}" OPENAI_API_KEY="${BCP_JUDGE_API_KEY}" \
"${VPY}" scripts_evaluation/evaluate_with_openai.py \
    --input_dir "${RUN_DIR}" \
    --model "${BCP_JUDGE_MODEL}" \
    --eval_dir "${RESULTS_DIR}/browsecomp_plus_evals" \
    2>&1 | tee "${LOG_DIR}/bcp_judge.log"

echo "[bcp] done; evals in ${RESULTS_DIR}/browsecomp_plus_evals"
