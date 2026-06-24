#!/bin/bash
# Search model, RUN 3 ONLY, on the repurposed tau server: agent :7003 (now
# serving SeaSFT-Search) paired with GLM :7009 (the tau model's old user-sim).
# Runs in parallel with the main search runner (which handles runs 1 & 2 on
# :7002/:7008). Same fix config: stop on </tool_call>, take-first, Fix B.
set -u
TAU2=/home/ec2-user/tau2-bench
LOGD=/home/ec2-user/slime/eval_scai/logs/tau2_rp; mkdir -p "$LOGD"
OUTD=/home/ec2-user/slime/eval_scai/tau2_rp_results; mkdir -p "$OUTD"
NUM_TRIALS="${NUM_TRIALS:-4}"; CONC="${CONC:-16}"
MAX_STEPS="${MAX_STEPS:-200}"; MAX_TOKENS="${MAX_TOKENS:-8192}"
DOMAINS="${DOMAINS:-retail airline telecom}"
RUN_ID="${RUN_ID:-3}"
NAME=Qwen3-8B-Base-Math-SeaSFT-Search; APORT=7003; GPORT=7009
export OPENAI_API_KEY=dummy
export TAU2_LENIENT_TOOL_ARGS=1
export TAU2_FIRST_TOOL_CALL=1
cd "$TAU2" || exit 1
log(){ echo "[$(date +%F' '%T)] $*"; }
AARGS="{\"temperature\":0,\"max_tokens\":${MAX_TOKENS},\"stop\":[\"</tool_call>\",\"<|im_end|>\"],\"api_base\":\"http://127.0.0.1:${APORT}/v1\"}"
UARGS="{\"temperature\":0,\"max_tokens\":${MAX_TOKENS},\"api_base\":\"http://127.0.0.1:${GPORT}/v1\"}"
for d in $DOMAINS; do
  SAVE="tau2rp_${NAME}_${d}_run${RUN_ID}"
  rm -rf "${TAU2}/data/simulations/${SAVE}" 2>/dev/null
  log "[search-run${RUN_ID}] ${d} START (agent :${APORT} user :${GPORT}) -> ${SAVE}"
  tau2 run --domain "$d" \
    --agent-llm "openai/${NAME}" --agent-llm-args "$AARGS" \
    --user-llm "openai/GLM-4.7-Flash" --user-llm-args "$UARGS" \
    --num-trials "$NUM_TRIALS" --max-concurrency "$CONC" --max-steps "$MAX_STEPS" \
    --save-to "$SAVE" > "${LOGD}/${SAVE}.log" 2>&1
  log "[search-run${RUN_ID}] ${d} rc=$? -> ${SAVE}"
  cp -r "${TAU2}/data/simulations/${SAVE}" "$OUTD/" 2>/dev/null || true
done
log "############ SEARCH RUN${RUN_ID} COMPLETE ############"
