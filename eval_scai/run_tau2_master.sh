#!/bin/bash
# Master: wait for tau1 to finish (free the 128k servers), precheck that one
# capped task completes+scores, then run the full official tau2 matrix.
set -u
ROOT=/home/ec2-user/slime/eval_scai
TAU2=/home/ec2-user/tau2-bench
LOGD="$ROOT/logs/tau2"; mkdir -p "$LOGD"
log(){ echo "[$(date +%F' '%T)] [tau2master] $*"; }

# 1. Wait for tau1 orchestrators to finish (avoid 128k KV contention).
log "waiting for tau1 to finish before starting tau2..."
while pgrep -f "run_tau1.sh" >/dev/null 2>&1; do sleep 60; done
log "tau1 finished; settling 30s"; sleep 30

# 2. Precheck: one capped task must complete + produce a scored simulation.
cd "$TAU2" || exit 1
rm -rf "$TAU2/data/simulations/precheck"
log "precheck: 1 task base/retail (max_steps=50, max_tokens=1024)"
OPENAI_API_KEY=dummy timeout 900 tau2 run --domain retail \
  --agent-llm openai/qwen-8b-base \
  --agent-llm-args '{"temperature":0,"max_tokens":1024,"api_base":"http://127.0.0.1:7000/v1"}' \
  --user-llm openai/GLM-4.7-Flash \
  --user-llm-args '{"temperature":0,"max_tokens":1024,"api_base":"http://127.0.0.1:7006/v1"}' \
  --num-trials 1 --num-tasks 1 --max-concurrency 1 --max-steps 50 \
  --save-to precheck > "$LOGD/precheck.log" 2>&1
nsims=$(python -c "import json;print(len(json.load(open('$TAU2/data/simulations/precheck/results.json')).get('simulations',[])))" 2>/dev/null || echo 0)
log "precheck n_sims=$nsims"
if [ "${nsims:-0}" -lt 1 ]; then
  log "PRECHECK FAILED (no completed sim) -> aborting full run. See $LOGD/precheck.log"
  exit 1
fi
log "precheck OK ($nsims sim). Launching full official tau2 matrix."

# 3. Full matrix: base+math+search x retail/airline/telecom x num_trials4 x 3 runs.
RUNS=3 NUM_TRIALS=4 CONC=8 MAX_STEPS=50 MAX_TOKENS=1024 bash "$ROOT/run_tau2_official.sh"
log "MASTER DONE"
