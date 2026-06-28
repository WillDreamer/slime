#!/bin/bash
# Run tau3 (= tau2-bench v1.0.0, "tau-three") across the 4 multi-stage ckpts,
# with GLM-4.7-Flash as the user simulator. Idempotent / resumable.
#
# WHAT MAKES THIS "tau3" (vs the earlier tau2 runs in eval_scai/run_tau2_*.sh):
#   1. banking_knowledge domain  -- the new tau3 knowledge/RAG domain, run with
#      an OFFLINE retrieval config (bm25: KB_search tool, no API key / sandbox).
#   2. --task-split-name base    -- the tau3 task set with the 75+ task fixes
#      (SABER). `base` = full original task set, the split to use for evaluation.
#   3. same tau2 CLI / serving fixes as before (stop tokens, parsers, lenient
#      args, take-first-tool-call for the search lane).
# Everything is written under /mnt/old-data3 (data-location policy).
#
# Prereqs (run inside the `slime` container):
#   - servers up:  bash serve_tau3_fleet.sh   (agents 7000-7003, GLM 7006-7009)
#   - tau2 CLI installed + patched: bash /mnt/old-data3/home/ec2-user/slime/aws/_setup_tau_deps.sh
set -u

TAU2="${TAU2:-/mnt/old-data3/home/ec2-user/tau2-bench}"
LOGD="${LOGD:-/mnt/old-data3/home/ec2-user/slime/eval_scai/tau3/logs}"
OUTD="${OUTD:-/mnt/old-data3/home/ec2-user/slime/eval_scai/tau3/tau3_rp_results}"
mkdir -p "$LOGD" "$OUTD"

NUM_TRIALS="${NUM_TRIALS:-4}"
CONC="${CONC:-16}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
RUNS="${RUNS:-1 2 3}"
SPLIT="${SPLIT:-base}"                       # tau3 task split (base|train|test)
DOMAINS="${DOMAINS:-retail airline telecom mock banking_knowledge}"
KB_RETRIEVAL="${KB_RETRIEVAL:-bm25}"         # offline: no_knowledge|full_kb|golden_retrieval|grep_only|bm25

# serving fixes (see TAU2_SEARCH_SETUP.md)
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export TAU2_LENIENT_TOOL_ARGS="${TAU2_LENIENT_TOOL_ARGS:-1}"

command -v tau2 >/dev/null || { echo "FATAL: 'tau2' not on PATH -- run _setup_tau_deps.sh in the slime container"; exit 1; }
cd "$TAU2" || { echo "FATAL: no tau2-bench at $TAU2"; exit 1; }
log(){ echo "[$(date +%F' '%T)] $*" | tee -a "$LOGD/MASTER.log"; }

# name | agent_port | glm_port | extra_stop | first_tool_call(0/1)
# 2 GLM replicas (tp=2) on 7006/7007; 2 agent lanes paired to each.
LANES=(
  "qwen-8b-base|7000|7006||0"
  "Qwen3-8B-Base-Math|7001|7006||0"
  "Qwen3-8B-Base-Math-SeaSFT-Search|7002|7007|</tool_call>|1"
  "Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau|7003|7007||0"
)

run_lane(){
  local name=$1 ap=$2 gp=$3 estop=$4 first=$5
  local stop='"<|im_end|>"'; [ -n "$estop" ] && stop="\"$estop\",\"<|im_end|>\""
  local AARGS="{\"temperature\":0,\"max_tokens\":$MAX_TOKENS,\"stop\":[$stop],\"api_base\":\"http://127.0.0.1:$ap/v1\"}"
  local UARGS="{\"temperature\":0,\"max_tokens\":$MAX_TOKENS,\"api_base\":\"http://127.0.0.1:$gp/v1\"}"
  local FC=""; [ "$first" = "1" ] && FC="TAU2_FIRST_TOOL_CALL=1"
  for i in $RUNS; do for d in $DOMAINS; do
    local SAVE="tau3rp_${name}_${d}_run${i}"
    # knowledge domain needs an (offline) retrieval config
    local EXTRA=""; [ "$d" = "banking_knowledge" ] && EXTRA="--retrieval-config $KB_RETRIEVAL"
    log "[$name:$ap] $d run$i split=$SPLIT $EXTRA -> $SAVE"
    env $FC tau2 run --domain "$d" \
      --task-split-name "$SPLIT" $EXTRA \
      --agent-llm "openai/$name"          --agent-llm-args "$AARGS" \
      --user-llm  "openai/GLM-4.7-Flash"  --user-llm-args  "$UARGS" \
      --num-trials "$NUM_TRIALS" --max-concurrency "$CONC" --max-steps "$MAX_STEPS" \
      --auto-resume --save-to "$SAVE" > "$LOGD/${SAVE}.log" 2>&1
    log "[$name:$ap] $d run$i rc=$? -> $SAVE"
    cp -r "$TAU2/data/simulations/$SAVE" "$OUTD/" 2>/dev/null || true
  done; done
  log "[$name:$ap] LANE DONE"
}

pids=()
for L in "${LANES[@]}"; do
  IFS='|' read -r n a g e f <<< "$L"
  run_lane "$n" "$a" "$g" "$e" "$f" &
  pids+=($!)
done
wait "${pids[@]}"
log "############ tau3 run COMPLETE ############"
