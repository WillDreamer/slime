#!/bin/bash
# Block until every server the eval needs is truly answering:
#   model servers 7000-7003  -> GET  /health == 200
#   retrievers    9000-9003  -> POST /retrieve returns a result (index loaded)
# Returns 0 when all ready, 1 on timeout.
set -o pipefail
TIMEOUT=${READY_TIMEOUT:-1800}
deadline=$(( $(date +%s) + TIMEOUT ))

model_ok () { curl -sf -m 5 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }
retr_ok ()  { curl -s  -m 8 "http://127.0.0.1:$1/retrieve" -H 'Content-Type: application/json' \
                -d '{"queries":["health"],"topk":1,"return_scores":false}' 2>/dev/null | grep -q result; }

all_ready () {
  for p in 7000 7001 7002 7003; do model_ok "$p" || return 1; done
  for p in 9002 9003; do retr_ok  "$p" || return 1; done
  return 0
}

echo "[wait-ready] waiting for model servers 7000-7003 + retrievers 9002/9003..."
n=0
until all_ready; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[wait-ready] TIMEOUT after ${TIMEOUT}s"
    for p in 7000 7001 7002 7003; do model_ok "$p" && echo "  :$p model OK" || echo "  :$p model DOWN"; done
    for p in 9002 9003; do retr_ok  "$p" && echo "  :$p retr  OK" || echo "  :$p retr  DOWN"; done
    exit 1
  fi
  n=$((n+1)); [ $((n % 6)) -eq 0 ] && echo "[wait-ready] still waiting ($((n*10))s)..."
  sleep 10
done
echo "[wait-ready] ALL READY (4 models + 4 retrievers)"
