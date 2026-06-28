#!/bin/bash
# Serve the fleet for tau3 (= tau2-bench v1.0.0) evaluation.
# Topology for 8x H100 80GB (GLM-4.7-Flash is a 59G MoE -> needs tp=2 to fit):
#   - 4 AGENT models, tp=1, GPUs 0-3, ports 7000-7003.
#       tool-call parser ONLY (qwen); reasoning parser OFF on purpose (these
#       ckpts emit <tool_call> with no <think> block -> a reasoning parser would
#       route the tool call into reasoning_content and zero the score).
#   - 2 GLM-4.7-Flash user-sim replicas, tp=2, GPUs 4-5 and 6-7, ports 7006-7007.
#       GLM keeps BOTH parsers (glm47 tool-call + glm45 reasoning).
# Run INSIDE the serving container (host net, all 8 GPUs, hf_cache mounted).
# Idempotent: a port already /health-green is skipped.
set -u

LOG_DIR="${LOG_DIR:-/home/ec2-user/slime/eval_scai/tau3/logs/fleet}"
mkdir -p "${LOG_DIR}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
GLM_MODEL="${GLM_MODEL:-zai-org/GLM-4.7-Flash}"
GLM_CTX="${GLM_CTX:-32768}"     # bound KV cache (model max is 202k; user-sim turns are short)

# gpu(s)  port  hf-model-path                                          served-name  tp
AGENTS=(
  "0 7000 Qwen/Qwen3-8B-Base                                     qwen-8b-base                                  1"
  "1 7001 willhx/Qwen3-8B-Base-Math                              Qwen3-8B-Base-Math                            1"
  "2 7002 willhx/Qwen3-8B-Base-Math-SeaSFT-Search                Qwen3-8B-Base-Math-SeaSFT-Search              1"
  "3 7003 willhx/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau     Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau   1"
)
# gpus   port  tp
USERSIMS=(
  "4,5 7006 2"
  "6,7 7007 2"
)

healthy(){ curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/health" 2>/dev/null | grep -q 200; }

# agents: tool-call parser only, no reasoning parser
for spec in "${AGENTS[@]}"; do
  read -r GPUS PORT MODEL NAME TP <<< "$spec"
  if healthy "$PORT"; then echo "[fleet] :$PORT healthy ($NAME) -- skip"; continue; fi
  echo "[fleet] GPU $GPUS :$PORT tp$TP <- $MODEL ($NAME)"
  CUDA_VISIBLE_DEVICES="$GPUS" setsid nohup python -m sglang.launch_server \
      --model-path "$MODEL" --served-model-name "$NAME" \
      --host 0.0.0.0 --port "$PORT" --tp "$TP" \
      --mem-fraction-static "$MEM_FRACTION" --tool-call-parser qwen \
      > "${LOG_DIR}/server_${PORT}.log" 2>&1 &
  echo $! > "${LOG_DIR}/server_${PORT}.pid"
done

# user-sims: GLM, both parsers, tp=2
for spec in "${USERSIMS[@]}"; do
  read -r GPUS PORT TP <<< "$spec"
  if healthy "$PORT"; then echo "[fleet] :$PORT healthy (GLM) -- skip"; continue; fi
  echo "[fleet] GPU $GPUS :$PORT tp$TP <- $GLM_MODEL (GLM-4.7-Flash)"
  CUDA_VISIBLE_DEVICES="$GPUS" setsid nohup python -m sglang.launch_server \
      --model-path "$GLM_MODEL" --served-model-name "GLM-4.7-Flash" \
      --host 0.0.0.0 --port "$PORT" --tp "$TP" \
      --mem-fraction-static "$MEM_FRACTION" --context-length "$GLM_CTX" \
      --tool-call-parser glm47 --reasoning-parser glm45 \
      > "${LOG_DIR}/server_${PORT}.log" 2>&1 &
  echo $! > "${LOG_DIR}/server_${PORT}.pid"
done

echo "[fleet] launch issued. logs: ${LOG_DIR}"
echo "[fleet] wait for health:"
echo "  for p in 7000 7001 7002 7003 7006 7007; do until curl -sf http://127.0.0.1:\$p/health >/dev/null; do sleep 5; done; echo \"\$p up\"; done"
