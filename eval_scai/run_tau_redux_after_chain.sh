#!/bin/bash
# After the tau->base cheap chain (CHAIN_PID) finishes and frees GPU3:
#   1. serve TAU on GPU3:8400 -> mmlu_redux x3 + aime2024 x3 -> stop
#   2. serve BASE on GPU3:8400 -> aime2024 x3 -> stop
# (base already gets aime2025/cheap suite from the chain; mmlu_redux for base
#  was not requested, so base only needs aime2024 here.)
set -u
EVAL_ROOT=/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai
PYBIN=/xuanwu-tank/north/xw27/envs/sglang_env/bin/python
CHAIN_PID="${CHAIN_PID:?CHAIN_PID=pid of run_tau_then_base_cheap.sh}"
DLOG="$EVAL_ROOT/logs/tau_redux_after_chain.log"
log(){ echo "[$(date +%F' '%T)] [post-chain] $*" | tee -a "$DLOG"; }

serve_on_gpu3() {  # ckpt -> sets SPID; waits healthy
  local ckpt="$1" name="$2"
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:8400/health 2>/dev/null)" = "200" ]; then SPID=""; return 0; fi
  CUDA_VISIBLE_DEVICES=3 nohup "$PYBIN" -m sglang.launch_server \
    --model-path "$ckpt" --served-model-name "$ckpt" \
    --host 0.0.0.0 --port 8400 --tp 1 --mem-fraction-static 0.85 \
    --tool-call-parser qwen --trust-remote-code \
    > "$EVAL_ROOT/logs/$name/redux_aime_server.log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:8400/health 2>/dev/null)" = 200 ] && return 0
    sleep 15
  done
  return 1
}
stop_gpu3() { [ -n "${SPID:-}" ] && { kill "$SPID" 2>/dev/null; sleep 8; kill -9 "$SPID" 2>/dev/null; }; sleep 3; }

log "waiting for cheap chain pid $CHAIN_PID to finish..."
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done
log "chain finished; GPU3 free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i 3 2>/dev/null)"

# --- TAU: redux x3 + aime2024 x3 ---
TAU_CKPT=/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau/hf_iter_0000300
TAU_NAME=Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau
mkdir -p "$EVAL_ROOT/logs/$TAU_NAME"
log "serving TAU on GPU3:8400"
if serve_on_gpu3 "$TAU_CKPT" "$TAU_NAME"; then
  log "TAU ready (pid ${SPID:-reused}); mmlu_redux x3 then aime2024 x3"
  NAME="$TAU_NAME" SERVED="$TAU_CKPT" BASE_URL=http://127.0.0.1:8400/v1 RUNS=3 bash "$EVAL_ROOT/run_mmlu_redux.sh" 2>&1 | tee -a "$DLOG"
  NAME="$TAU_NAME" SERVED="$TAU_CKPT" BASE_URL=http://127.0.0.1:8400/v1 RUNS=3 bash "$EVAL_ROOT/run_aime2024.sh" 2>&1 | tee -a "$DLOG"
else log "ERROR: TAU server unhealthy; skipping tau"; fi
stop_gpu3

# --- BASE: aime2024 x3 ---
BASE_CKPT=/xuanwu-tank/center/whx/Qwen3-8B-Base
BASE_NAME=Qwen3-8B-Base
mkdir -p "$EVAL_ROOT/logs/$BASE_NAME"
log "serving BASE on GPU3:8400"
if serve_on_gpu3 "$BASE_CKPT" "$BASE_NAME"; then
  log "BASE ready (pid ${SPID:-reused}); aime2024 x3"
  NAME="$BASE_NAME" SERVED="$BASE_CKPT" BASE_URL=http://127.0.0.1:8400/v1 RUNS=3 bash "$EVAL_ROOT/run_aime2024.sh" 2>&1 | tee -a "$DLOG"
else log "ERROR: BASE server unhealthy; skipping base"; fi
stop_gpu3

log "############ post-chain redux+aime2024 COMPLETE ############"
