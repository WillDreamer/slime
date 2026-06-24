#!/bin/bash
# ============================================================================
# Full resume after a spot interruption that brings the experiment up on a
# NEW machine via AMI / root-volume transfer (everything — docker images,
# containers, /home/ec2-user data — arrives intact).
#
# It (0) waits for docker, (1) ensures the 3 containers exist & run,
# (2) relaunches EVERY server on its fixed port/GPU (idempotent), (3) waits for
# the servers, (4) resumes the tau1 + tau2 eval (idempotent: completed cells are
# skipped; tau2-bench auto-resumes each --save-to).
#
# Idempotent throughout: a port already bound is left alone (load-insensitive
# TCP check), a running container is not disturbed. Safe to re-run any time.
#
# Topology restored (all agents NATIVE 32k, no YaRN; GLM = clean user-sim):
#   slime      (GPU0)   :8000 Skywork RM (embed) , :7005 Qwen3-32B
#   slime-srv  (GPU1-7) :7000 base :7001 math :7002 search :7003 tau (tool-call qwen)
#                       :7006/:7007/:7008 GLM-4.7-Flash (tool-call glm47 + reasoning glm45)
#   slime-search (GPU2) :9002/:9003 FAISS retrievers (Search-R1 eval — COMPLETE/retired)
#
# tau eval resume: see aws/_setup_tau_deps.sh + aws/_run_tau.sh + README-tau-eval.md
# ============================================================================
set -o pipefail
AWS=/home/ec2-user/slime/aws
IMG=slimerl/slime:latest
log(){ echo "[$(date -u +%FT%TZ)] [resume] $*"; }
port_bound(){ timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null; }

# --- 0) docker daemon ---------------------------------------------------------
log "waiting for docker daemon..."
for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null 2>&1 || { log "FATAL: docker unavailable"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { log "image missing, pulling $IMG"; docker pull "$IMG"; }

# --- 1) ensure the 3 containers (create if missing, else start) ---------------
ensure_container(){  # name  [extra -v args...]
  local name=$1; shift
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then log "container $name already running"; return; fi
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then log "docker start $name"; docker start "$name" >/dev/null; return; fi
  log "creating container $name"
  docker run -d --name "$name" --gpus all --network host --ipc host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 --restart unless-stopped \
    -v /home/ec2-user/slime:/home/ec2-user/slime "$@" \
    "$IMG" sleep infinity >/dev/null
}
ensure_container slime
ensure_container slime-srv    -v /home/ec2-user/hf_cache:/root/.cache/huggingface
ensure_container slime-search -v /home/ec2-user/Search_data:/data/Search_data

# eval deps (only missing if slime-search was freshly recreated)
docker exec slime-search bash -lc 'python -c "import aiohttp" 2>/dev/null || pip install -q aiohttp' >/dev/null 2>&1 || true

# --- 2) launch every sglang server (idempotent: skip bound ports) -------------
# spec: container|port|gpu|model|served_name|mem|extra-args
SERVERS=(
  "slime|8000|0|Skywork/Skywork-Reward-V2-Llama-3.1-8B||0.3|--is-embedding --context-length 16384"
  "slime|7005|0|Qwen/Qwen3-32B|Qwen3-32B|0.65|"
  "slime-srv|7000|3|Qwen/Qwen3-8B-Base|qwen-8b-base|0.85|--tool-call-parser qwen"
  "slime-srv|7001|4|willhx/Qwen3-8B-Base-Math|Qwen3-8B-Base-Math|0.85|--tool-call-parser qwen"
  "slime-srv|7002|5|willhx/Qwen3-8B-Base-Math-SeaSFT-Search|Qwen3-8B-Base-Math-SeaSFT-Search|0.85|--tool-call-parser qwen"
  "slime-srv|7003|6|willhx/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau|Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau|0.85|--tool-call-parser qwen"
  "slime-srv|7006|1|zai-org/GLM-4.7-Flash|GLM-4.7-Flash|0.85|--tool-call-parser glm47 --reasoning-parser glm45"
  "slime-srv|7007|7|zai-org/GLM-4.7-Flash|GLM-4.7-Flash|0.85|--tool-call-parser glm47 --reasoning-parser glm45"
  "slime-srv|7008|2|zai-org/GLM-4.7-Flash|GLM-4.7-Flash|0.85|--tool-call-parser glm47 --reasoning-parser glm45"
)
mkdir -p "$AWS/logs"
for spec in "${SERVERS[@]}"; do
  IFS='|' read -r cont port gpu model name mem extra <<< "$spec"
  if port_bound "$port"; then log ":$port already bound — skip"; continue; fi
  namearg=""; [ -n "$name" ] && namearg="--served-model-name $name"
  log "launch :$port gpu=$gpu ($model) in $cont"
  docker exec -d "$cont" bash -lc \
    "CUDA_VISIBLE_DEVICES=$gpu setsid nohup python -m sglang.launch_server \
       --model-path '$model' $namearg --host 0.0.0.0 --port $port \
       --tp 1 --mem-fraction-static $mem $extra \
       > /home/ec2-user/slime/aws/logs/server_$port.log 2>&1"
done

# --- 3) wait for the model/GLM servers to answer ------------------------------
# NOTE: the Search-R1 eval is COMPLETE and its FAISS retrievers (9002/9003) were
# retired to free GPU2 for GLM-4.7-Flash :7008. To bring the retrievers + eval
# back, re-add a retriever launch (aws/_retrievers.sh) on a free GPU and call
# aws/_run_eval.sh — both scripts remain in this folder.
log "waiting for sglang servers to answer..."
deadline=$(( $(date +%s) + 1800 ))
for port in 8000 7000 7001 7002 7003 7005 7006 7007 7008; do
  until curl -sf -m4 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    [ "$(date +%s)" -ge "$deadline" ] && { log "WARN: :${port} not ready before timeout"; break; }
    sleep 5
  done
  curl -sf -m4 "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && log ":${port} ready"
done
log "all servers up."

# --- 4) resume the tau1 + tau2 eval (idempotent) ------------------------------
# Runs in the `slime` container: ensures tau deps (tau_bench + official
# tau2-bench) then (re)launches run_parallel.sh. tau1 skips completed temp0
# cells; tau2 (tau2-bench) auto-resumes each --save-to. Safe no-op if all done.
log "ensuring tau deps + resuming tau1/tau2 eval (slime container)..."
docker exec slime bash -lc '/home/ec2-user/slime/aws/_setup_tau_deps.sh' \
  >> "$AWS/logs/setup_tau_deps.log" 2>&1 || log "WARN: tau dep setup hit an error (see logs/setup_tau_deps.log)"
docker exec slime bash -lc '/home/ec2-user/slime/aws/_run_tau.sh' 2>&1 | sed 's/^/[resume][tau] /'
log "DONE. servers up + tau eval resumed."
