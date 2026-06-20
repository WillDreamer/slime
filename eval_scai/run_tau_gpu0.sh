#!/bin/bash
# Serve TAU model on GPU0:8400, then run mmlu_redux x3 (idempotent: run1 already
# done -> fills run2,run3) and aime2024 x3, then stop the server.
set -u
EVAL_ROOT=/xuanwu-tank/north/xw27/multi/slime_scai/eval_scai
PYBIN=/xuanwu-tank/north/xw27/envs/sglang_env/bin/python
TAU_CKPT=/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau/hf_iter_0000300
NAME=Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau
GPU=0; PORT=8400; BASE_URL="http://127.0.0.1:${PORT}/v1"
LOGD="$EVAL_ROOT/logs/$NAME"; mkdir -p "$LOGD"
DLOG="$EVAL_ROOT/logs/tau_gpu0.log"
log(){ echo "[$(date +%F' '%T)] [tau/gpu0] $*" | tee -a "$DLOG"; }

log "serving tau on GPU${GPU}:${PORT}"
CUDA_VISIBLE_DEVICES=$GPU nohup "$PYBIN" -m sglang.launch_server \
  --model-path "$TAU_CKPT" --served-model-name "$TAU_CKPT" \
  --host 0.0.0.0 --port "$PORT" --tp 1 --mem-fraction-static 0.85 \
  --tool-call-parser qwen --trust-remote-code > "$LOGD/gpu0_server.log" 2>&1 &
SPID=$!
for _ in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:${PORT}/health 2>/dev/null)" = 200 ] && break
  sleep 15
done
[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://127.0.0.1:${PORT}/health 2>/dev/null)" = 200 ] || { log "ERROR: server not healthy; see $LOGD/gpu0_server.log"; exit 1; }
log "tau ready (pid $SPID); mmlu_redux x3 (run1 done -> fills run2,3) then aime2024 x3"

NAME="$NAME" SERVED="$TAU_CKPT" BASE_URL="$BASE_URL" RUNS=3 bash "$EVAL_ROOT/run_mmlu_redux.sh" 2>&1 | tee -a "$DLOG"
NAME="$NAME" SERVED="$TAU_CKPT" BASE_URL="$BASE_URL" RUNS=3 bash "$EVAL_ROOT/run_aime2024.sh" 2>&1 | tee -a "$DLOG"

log "stopping tau server $SPID"; kill "$SPID" 2>/dev/null; sleep 8; kill -9 "$SPID" 2>/dev/null
log "############ tau GPU0 redux+aime2024 COMPLETE ############"
