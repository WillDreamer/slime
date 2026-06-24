#!/bin/bash
# Re-run ONLY the SeaSFT-Search model on tau2 with the fix for its runaway
# tool-call behavior (Search-R1 SFT removed turn boundaries):
#   - agent stop on </tool_call> (and <|im_end|>) so generation halts after a call
#   - TAU2_FIRST_TOOL_CALL=1 -> take only the first tool call per turn
#   - TAU2_LENIENT_TOOL_ARGS=1 -> Fix B safety net
# Pairs search agent (:7002) with GLM user-sim (:7008). 3 runs x 3 domains.
# Does NOT touch the other 3 lanes. See TAU2_FORMAT_ISSUE.md.
set -u
TAU2=/home/ec2-user/tau2-bench
LOGD=/home/ec2-user/slime/eval_scai/logs/tau2_rp; mkdir -p "$LOGD"
OUTD=/home/ec2-user/slime/eval_scai/tau2_rp_results; mkdir -p "$OUTD"
RUNS="${RUNS:-3}"; NUM_TRIALS="${NUM_TRIALS:-4}"; CONC="${CONC:-16}"
MAX_STEPS="${MAX_STEPS:-200}"; MAX_TOKENS="${MAX_TOKENS:-8192}"
DOMAINS="${DOMAINS:-retail airline telecom}"
NAME=Qwen3-8B-Base-Math-SeaSFT-Search; APORT=7002; GPORT=7008
export OPENAI_API_KEY=dummy
export TAU2_LENIENT_TOOL_ARGS=1
export TAU2_FIRST_TOOL_CALL=1
cd "$TAU2" || exit 1
log(){ echo "[$(date +%F' '%T)] $*"; }
AARGS="{\"temperature\":0,\"max_tokens\":${MAX_TOKENS},\"stop\":[\"</tool_call>\",\"<|im_end|>\"],\"api_base\":\"http://127.0.0.1:${APORT}/v1\"}"
UARGS="{\"temperature\":0,\"max_tokens\":${MAX_TOKENS},\"api_base\":\"http://127.0.0.1:${GPORT}/v1\"}"
for i in $(seq 1 "$RUNS"); do
  for d in $DOMAINS; do
    SAVE="tau2rp_${NAME}_${d}_run${i}"
    rm -rf "${TAU2}/data/simulations/${SAVE}" 2>/dev/null   # clear prior artifact
    log "[search] ${d} run${i} START (agent :${APORT} user :${GPORT}, first-call+stop) -> ${SAVE}"
    tau2 run --domain "$d" \
      --agent-llm "openai/${NAME}" --agent-llm-args "$AARGS" \
      --user-llm "openai/GLM-4.7-Flash" --user-llm-args "$UARGS" \
      --num-trials "$NUM_TRIALS" --max-concurrency "$CONC" --max-steps "$MAX_STEPS" \
      --save-to "$SAVE" > "${LOGD}/${SAVE}.log" 2>&1
    log "[search] ${d} run${i} rc=$? -> ${SAVE}"
    cp -r "${TAU2}/data/simulations/${SAVE}" "$OUTD/" 2>/dev/null || true
  done
done
log "############ SEARCH-ONLY tau2 (${RUNS} runs) COMPLETE ############"
