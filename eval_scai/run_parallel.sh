#!/bin/bash
# Parallel eval: 4 lanes (one per model), each doing tau1 x3 then tau2 x3 on its
# OWN agent server (7000-7003, separate GPUs) + an assigned GLM user-sim.
#   GLM map: base->7006, math->7007, search->7008, tau->7006
# Completed cells are skipped (idempotent), so already-done base/math tau1 skip.
set -u
ROOT=/home/ec2-user/slime/eval_scai
L="$ROOT/logs"; mkdir -p "$L/tau1" "$L/tau2"
log(){ echo "[$(date +%F' '%T)] [parallel] $*"; }
TAU="Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau"
MODELS=(qwen-8b-base Qwen3-8B-Base-Math Qwen3-8B-Base-Math-SeaSFT-Search "$TAU")

log "waiting for all servers (agents 7000-7003, GLM 7006-7008)..."
for i in $(seq 1 80); do
  ok=1; for p in 7000 7001 7002 7003 7006 7007 7008; do
    h=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://127.0.0.1:$p/health" 2>/dev/null)
    [ "$h" = "200" ] || ok=0
  done
  [ "$ok" = "1" ] && break; sleep 15
done
[ "${ok:-0}" = "1" ] || { log "servers not all healthy; abort"; exit 1; }
log "servers healthy; launching 4 parallel lanes (tau1 then tau2 per model)"

lane(){
  local m="$1"
  log "[$m] tau1 start"
  RUNS=3 bash "$ROOT/run_tau1.sh" "$m" > "$L/tau1/par_${m}.log" 2>&1
  log "[$m] tau1 done (rc=$?); tau2 start"
  if [ "$m" = "$TAU" ]; then
    RUNS=3 bash "$ROOT/run_tau2_tau.sh" > "$L/tau2/par_${m}.log" 2>&1
  else
    RUNS=3 ONLY="$m" bash "$ROOT/run_tau2_official.sh" > "$L/tau2/par_${m}.log" 2>&1
  fi
  log "[$m] tau2 done (rc=$?)"
}

for m in "${MODELS[@]}"; do lane "$m" & done
wait
log "############ ALL 4 PARALLEL LANES COMPLETE ############"
