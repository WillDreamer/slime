#!/bin/bash
# Clean full re-run with the FIXED user-sim (GLM reasoning-parser glm45 -> reasoning
# in reasoning_content, content = clean customer message). All 4 models, temp0.
#   order: tau1 (all 4, fast) -> tau2 tau-model (meaningful) -> tau2 base/math/search (slow)
set -u
ROOT=/home/ec2-user/slime/eval_scai
LOGD="$ROOT/logs"; mkdir -p "$LOGD/tau1" "$LOGD/tau2"
log(){ echo "[$(date +%F' '%T)] [allclean] $*"; }

log "waiting for all servers (agents 7000-7003, GLM 7006-7008)..."
for i in $(seq 1 60); do
  ok=1; for p in 7000 7001 7002 7003 7006 7007 7008; do
    h=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:$p/health 2>/dev/null)
    [ "$h" = "200" ] || ok=0
  done
  [ "$ok" = "1" ] && break; sleep 15
done
[ "${ok:-0}" = "1" ] || { log "servers not all healthy; abort"; exit 1; }
log "all servers healthy."

log "=== STEP 1: tau1 x3, all 4 models (temp0, clean user-sim) ==="
RUNS=3 bash "$ROOT/run_tau1.sh" > "$LOGD/tau1/RUN_clean_all.log" 2>&1
log "tau1 all rc=$?"

log "=== STEP 2: tau2 x3, TAU model (meaningful, fast) ==="
RUNS=3 bash "$ROOT/run_tau2_tau.sh" > "$LOGD/tau2/RUN_clean_taumodel.log" 2>&1
log "tau2 tau-model rc=$?"

log "=== STEP 3: tau2 x3, base/math/search (slow ~days each) ==="
RUNS=3 bash "$ROOT/run_tau2_official.sh" > "$LOGD/tau2/RUN_clean_matrix.log" 2>&1
log "tau2 base/math/search rc=$?"
log "############ CLEAN RE-RUN COMPLETE ############"
