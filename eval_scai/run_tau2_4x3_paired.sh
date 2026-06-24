#!/bin/bash
# tau2 (official tau2-bench) x RUNS, ALL 4 models, each paired 1:1 with its own
# GLM-4.7-Flash user-sim, run as 4 PARALLEL lanes.
#   agents : served with --tool-call-parser qwen --reasoning-parser qwen3
#   user   : GLM-4.7-Flash with --tool-call-parser glm47 --reasoning-parser glm45
#   Fix B  : TAU2_LENIENT_TOOL_ARGS=1 (tolerate bare-scalar tool args)
# Runs inside the tau2eval container (tau2-bench at $TAU2, servers on 127.0.0.1).
# Results are copied to a bind-mounted dir so they persist on the old-root volume.
#   Usage:  RUNS=3 bash run_tau2_4x3_paired.sh
set -u
TAU2=/home/ec2-user/tau2-bench
LOGD=/home/ec2-user/slime/eval_scai/logs/tau2_rp; mkdir -p "$LOGD"
OUTD=/home/ec2-user/slime/eval_scai/tau2_rp_results; mkdir -p "$OUTD"
RUNS="${RUNS:-3}"
NUM_TRIALS="${NUM_TRIALS:-4}"      # official pass^k
CONC="${CONC:-16}"
MAX_STEPS="${MAX_STEPS:-200}"      # official leaderboard value
MAX_TOKENS="${MAX_TOKENS:-8192}"   # raised from 2048 (Fix A)
DOMAINS="${DOMAINS:-retail airline telecom}"
export OPENAI_API_KEY=dummy
export TAU2_LENIENT_TOOL_ARGS="${TAU2_LENIENT_TOOL_ARGS:-1}"
cd "$TAU2" || { echo "no tau2-bench at $TAU2"; exit 1; }
log(){ echo "[$(date +%F' '%T)] $*"; }

# name  agent_port  glm_user_port   (1:1 pairing)
SPECS=(
  "qwen-8b-base 7000 7006"
  "Qwen3-8B-Base-Math 7001 7007"
  "Qwen3-8B-Base-Math-SeaSFT-Search 7002 7008"
  "Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau 7003 7009"
)

run_lane(){
  local name=$1 aport=$2 gport=$3
  # stop on <|im_end|>: these base-derived ckpts don't stop after </tool_call>,
  # spraying 100+ repeated tool calls/turn -> instant TOO_MANY_ERRORS. With the
  # stop token they emit exactly one tool call per turn.
  local AARGS="{\"temperature\":0,\"max_tokens\":${MAX_TOKENS},\"stop\":[\"<|im_end|>\"],\"api_base\":\"http://127.0.0.1:${aport}/v1\"}"
  local UARGS="{\"temperature\":0,\"max_tokens\":${MAX_TOKENS},\"api_base\":\"http://127.0.0.1:${gport}/v1\"}"
  for i in $(seq 1 "$RUNS"); do
    for d in $DOMAINS; do
      local SAVE="tau2rp_${name}_${d}_run${i}"
      log "[${name}] ${d} run${i} START (agent :${aport} user :${gport}) -> ${SAVE}"
      tau2 run --domain "$d" \
        --agent-llm "openai/${name}" --agent-llm-args "$AARGS" \
        --user-llm "openai/GLM-4.7-Flash" --user-llm-args "$UARGS" \
        --num-trials "$NUM_TRIALS" --max-concurrency "$CONC" --max-steps "$MAX_STEPS" \
        --save-to "$SAVE" > "${LOGD}/${SAVE}.log" 2>&1
      log "[${name}] ${d} run${i} rc=$? -> ${SAVE}"
      cp -r "${TAU2}/data/simulations/${SAVE}"* "$OUTD/" 2>/dev/null || true
    done
  done
  log "[${name}] LANE COMPLETE"
}

pids=()
for spec in "${SPECS[@]}"; do
  read -r name aport gport <<< "$spec"
  run_lane "$name" "$aport" "$gport" &
  pids+=($!)
done
wait "${pids[@]}"
log "############ ALL 4 LANES x ${RUNS} RUNS COMPLETE ############"
