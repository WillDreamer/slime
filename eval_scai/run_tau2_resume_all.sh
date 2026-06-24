#!/bin/bash
# Idempotent resume of the FULL tau2 eval (current-root anchored).
#   - 4 lanes in parallel; tau2 --auto-resume skips completed save-to dirs and
#     continues partial ones, so this is safe to run on every boot.
#   - search split across two GPUs: runs 1-2 on :7002, run 3 on :7003.
#   - TauSFT-Tau is COMPLETE (data preserved, never re-run here).
# Per-model serving fix encoded in the lane table (see TAU2_SEARCH_SETUP.md).
set -u
TAU2=/home/ec2-user/tau2-bench
LOGD=/home/ec2-user/slime/eval_scai/logs/tau2_rp; mkdir -p "$LOGD"
OUTD=/home/ec2-user/slime/eval_scai/tau2_rp_results; mkdir -p "$OUTD"
NUM_TRIALS="${NUM_TRIALS:-4}"; CONC="${CONC:-16}"; MAX_STEPS="${MAX_STEPS:-200}"; MAX_TOKENS="${MAX_TOKENS:-8192}"
DOMAINS="${DOMAINS:-retail airline telecom}"
export OPENAI_API_KEY=dummy
export TAU2_LENIENT_TOOL_ARGS=1
cd "$TAU2" || { echo "no tau2-bench at $TAU2"; exit 1; }
log(){ echo "[$(date +%F' '%T)] $*"; }

# name | agent_port | glm_port | runs | extra_stop | first_tool_call(0/1)
LANES=(
  "qwen-8b-base|7000|7006|1 2 3||0"
  "Qwen3-8B-Base-Math|7001|7007|1 2 3||0"
  "Qwen3-8B-Base-Math-SeaSFT-Search|7002|7008|1 2|</tool_call>|1"
  "Qwen3-8B-Base-Math-SeaSFT-Search|7003|7009|3|</tool_call>|1"
)

run_lane(){
  local name=$1 ap=$2 gp=$3 runs=$4 estop=$5 first=$6
  local stop='"<|im_end|>"'; [ -n "$estop" ] && stop="\"$estop\",\"<|im_end|>\""
  local AARGS="{\"temperature\":0,\"max_tokens\":$MAX_TOKENS,\"stop\":[$stop],\"api_base\":\"http://127.0.0.1:$ap/v1\"}"
  local UARGS="{\"temperature\":0,\"max_tokens\":$MAX_TOKENS,\"api_base\":\"http://127.0.0.1:$gp/v1\"}"
  local FC=""; [ "$first" = "1" ] && FC="TAU2_FIRST_TOOL_CALL=1"
  for i in $runs; do for d in $DOMAINS; do
    local SAVE="tau2rp_${name}_${d}_run${i}"
    log "[$name:$ap] $d run$i (auto-resume) -> $SAVE"
    env $FC tau2 run --domain "$d" \
      --agent-llm "openai/$name" --agent-llm-args "$AARGS" \
      --user-llm "openai/GLM-4.7-Flash" --user-llm-args "$UARGS" \
      --num-trials "$NUM_TRIALS" --max-concurrency "$CONC" --max-steps "$MAX_STEPS" \
      --auto-resume --save-to "$SAVE" > "$LOGD/${SAVE}.log" 2>&1
    log "[$name:$ap] $d run$i rc=$? -> $SAVE"
    cp -r "$TAU2/data/simulations/$SAVE" "$OUTD/" 2>/dev/null || true
  done; done
  log "[$name:$ap] LANE DONE"
}

pids=()
for L in "${LANES[@]}"; do IFS='|' read -r n a g r e f <<< "$L"; run_lane "$n" "$a" "$g" "$r" "$e" "$f" & pids+=($!); done
wait "${pids[@]}"
log "############ tau2 resume-all COMPLETE ############"
