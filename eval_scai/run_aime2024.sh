#!/bin/bash
# Run aime2024 x RUNS for ONE model against an existing server. Idempotent:
# a run is skipped if it already has a success aime2024 .eval.
# Config mirrors aime2025 in the master script: max_tokens 30000, temp 0.
# Usage: NAME=.. SERVED=.. BASE_URL=.. [RUNS=3] bash run_aime2024.sh
set -u
EVAL_ROOT=/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai
IPY="$EVAL_ROOT/.venv_inspect/bin/python"
NAME="${NAME:?NAME=results-subdir}"; SERVED="${SERVED:?SERVED=ckpt path}"
BASE_URL="${BASE_URL:?BASE_URL=http://...:port/v1}"; RUNS="${RUNS:-3}"
LOGD="$EVAL_ROOT/logs/$NAME"; mkdir -p "$LOGD"
log(){ echo "[$(date +%F' '%T)] [$NAME/aime2024] $*"; }
for i in $(seq 1 "$RUNS"); do
  RD="$EVAL_ROOT/results/$NAME/run$i"; mkdir -p "$RD/inspect_logs"
  if ls "$RD/inspect_logs/"*aime2024*.eval >/dev/null 2>&1 && "$IPY" - "$RD" <<'PY' 2>/dev/null
import sys, glob
from inspect_ai.log import read_eval_log
ok=any(read_eval_log(f, header_only=True).status=="success"
       for f in glob.glob(f"{sys.argv[1]}/inspect_logs/*aime2024*.eval"))
sys.exit(0 if ok else 1)
PY
  then log "skip run$i (success exists)"; continue; fi
  log "aime2024 -> run$i"
  "$IPY" "$EVAL_ROOT/benchmarks/inspect/run_inspect_api.py" \
    --task inspect_evals/aime2024 --model-name "$SERVED" --base-url "$BASE_URL" \
    --log-dir "$RD/inspect_logs" --max-tokens 30000 --temperature 0 \
    2>&1 | tee -a "$LOGD/run${i}_aime2024.log"
done
log "############ $NAME: aime2024 x$RUNS COMPLETE ############"
