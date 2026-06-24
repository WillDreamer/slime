#!/bin/bash
# Runs INSIDE the `slime-search` container. Idempotently (re)launches the 4
# FAISS-GPU dense retrievers: GPU 1 -> :9000,:9001 ; GPU 2 -> :9002,:9003.
# A port that already answers /retrieve is skipped. Servers on the SAME GPU are
# started sequentially so their ~61GB cpu->gpu index loads don't overlap.
# Uses the conda-forge faiss-gpu env (sm_90 / H200) at /data/Search_data/mamba.
set -u
export MAMBA_ROOT_PREFIX=/data/Search_data/mamba
MM=/data/Search_data/bin/micromamba
RUN=("$MM" run -n retriever python)
SERVER=/home/ec2-user/slime/eval_scai/benchmarks/search/retrieval_server.py
INDEX=/data/Search_data/e5_Flat.index
CORPUS=/data/Search_data/wiki-18.jsonl
LOGDIR=/data/Search_data/retriever_logs
mkdir -p "$LOGDIR"

# 9000/9001 (GPU1, base+Math retrievers) retired 2026-06-21 — base/Math eval done.
PAIRS=("9002:2" "9003:2")

# Load-insensitive idempotency guard: a bound port accepts a TCP connection
# instantly even when the server is saturated with eval traffic. NEVER launch a
# duplicate just because a busy server was slow to answer a request.
port_listening () { timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null; }

is_up () { curl -s -m 5 "http://127.0.0.1:$1/retrieve" -H "Content-Type: application/json" \
             -d '{"queries":["ping"],"topk":1,"return_scores":true}' 2>/dev/null | grep -q result; }

wait_ready () {
  local port=$1
  for i in $(seq 1 180); do is_up "$port" && { echo "[retriever:$port] READY (~$((i*10))s)"; return 0; }; sleep 10; done
  echo "[retriever:$port] ERROR: not ready; see ${LOGDIR}/retriever_${port}.log" >&2; return 1
}

for pair in "${PAIRS[@]}"; do
  port="${pair%%:*}"; gpu="${pair##*:}"
  if port_listening "$port"; then echo "[retriever:$port] port already bound — skip (no duplicate)"; continue; fi
  echo "[retriever:$port] launching on GPU ${gpu}..."
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup "${RUN[@]}" "$SERVER" \
    --index_path "$INDEX" --corpus_path "$CORPUS" \
    --retriever_name e5 --retriever_model intfloat/e5-base-v2 \
    --topk 3 --faiss_gpu --host 0.0.0.0 --port "$port" \
    > "${LOGDIR}/retriever_${port}.log" 2>&1 &
  echo $! > "${LOGDIR}/retriever_${port}.pid"
  wait_ready "$port" || exit 1
done
echo "[_retrievers] all 4 retrievers up."
