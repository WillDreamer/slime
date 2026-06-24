#!/bin/bash
# TAU model (final tool-trained ckpt) eval: wait for 128k :7003 + GLM :7008,
# then tau1 x3, then tau2 x3 (sequential — both use :7003, avoid contention).
set -u
ROOT=/home/ec2-user/slime/eval_scai
LOGD="$ROOT/logs"; mkdir -p "$LOGD/tau1" "$LOGD/tau2"
log(){ echo "[$(date +%F' '%T)] [taumodel] $*"; }

log "waiting for 7003 (tau model) + 7008 (GLM user-sim)..."
h3=000; h8=000
for i in $(seq 1 60); do
  h3=$(curl -s -o /dev/null -w "%{http_code}" --max-time 4 http://127.0.0.1:7003/health 2>/dev/null)
  h8=$(curl -s -o /dev/null -w "%{http_code}" --max-time 4 http://127.0.0.1:7008/health 2>/dev/null)
  [ "$h3" = "200" ] && [ "$h8" = "200" ] && break
  sleep 15
done
log "7003=$h3 7008=$h8"
[ "$h3" = "200" ] && [ "$h8" = "200" ] || { log "servers not healthy; abort"; exit 1; }
log "7003 context: $(grep -aoE 'context_length=[0-9]+' /tmp/glm128/srv_7003.log 2>/dev/null | tail -1)"

log "=== tau1 (tau model) x3, temp0 ==="
RUNS=3 bash "$ROOT/run_tau1.sh" TauSFT-Tau > "$LOGD/tau1/RUN_tau_model.log" 2>&1
log "tau1 tau model rc=$?"

log "=== tau2 (tau model) x3 ==="
RUNS=3 bash "$ROOT/run_tau2_tau.sh" > "$LOGD/tau2/RUN_tau_model_tau2.log" 2>&1
log "tau2 tau model rc=$?"
log "############ TAU MODEL tau1+tau2 COMPLETE ############"
