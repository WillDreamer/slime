#!/bin/bash
# Search-R1 EM eval driver — RESUME-SAFE. 4 model servers (7000-7003) x 3 runs
# each on the full 51,713-question 7-dataset test set. Each model paired with
# its own retriever (700i <-> 900i); all 4 models evaluate in parallel, the 3
# runs per model are sequential. A run whose search_full.json exists AND parses
# (has summary.em_overall) is skipped — so after a spot stop/start this picks up
# exactly where it left off (completed runs preserved; an interrupted run reruns
# from scratch, since eval_search.py writes results only at run end).
set -u
EVAL=/home/ec2-user/slime/eval_scai/benchmarks/search/eval_search.py
DATA=/data/Search_data/test.parquet
TOK=Qwen/Qwen3-8B-Base
OUT=/data/Search_data/eval_results
RUNS=3
TRIPLES=(
  "qwen-8b-base:7000:9000"
  "Qwen3-8B-Base-Math:7001:9001"
  "Qwen3-8B-Base-Math-SeaSFT-Search:7002:9002"
  "Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau:7003:9003"
)

run_done () {  # search_full.json -> 0 if a valid completed result
  python -c "import json,sys; json.load(open(sys.argv[1]))['summary']['em_overall']" "$1" 2>/dev/null
}

run_model () {  # name genport retport
  local name=$1 gp=$2 rp=$3
  for i in $(seq 1 "$RUNS"); do
    local rd="${OUT}/${name}/run${i}"; mkdir -p "$rd"
    local out="${rd}/search_full.json"
    if [ -f "$out" ] && run_done "$out"; then echo "[${name}] run${i}: complete, skip"; continue; fi
    echo "[${name}] run${i}/${RUNS} START (gen :${gp} retr :${rp})  $(date +%T)"
    python "$EVAL" \
      --base-url "http://127.0.0.1:${gp}" \
      --retriever-url "http://127.0.0.1:${rp}/retrieve" \
      --tokenizer "$TOK" --data "$DATA" \
      --output "$out" \
      --trajectory-output "${rd}/search_full_trajectories.jsonl" \
      --max-turns 5 --topk 3 --max-new-tokens 2048 --concurrency 64 --n-per-dataset 0 \
      > "${rd}/run.log" 2>&1
    if [ -f "$out" ] && run_done "$out"; then
      echo "[${name}] run${i}/${RUNS} DONE em_overall=$(run_done "$out")  $(date +%T)"
    else
      echo "[${name}] run${i}/${RUNS} FAILED (no/invalid output) — see ${rd}/run.log"
    fi
  done
  echo "[${name}] ALL ${RUNS} RUNS COMPLETE"
}

pids=()
for t in "${TRIPLES[@]}"; do
  IFS=: read -r name gp rp <<< "$t"
  run_model "$name" "$gp" "$rp" &
  pids+=($!)
done
wait "${pids[@]}"
echo "ALL_MODELS_ALL_RUNS_COMPLETE"
