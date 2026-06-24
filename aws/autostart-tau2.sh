#!/bin/bash
# Current-root-anchored auto-resume for the tau2 eval, after a spot stop/start.
# Self-contained on the LIVE root (/home/ec2-user): repo, hf_cache, and the
# experiment data dir all live here -- no dependency on /mnt/old-root.
#   (0) wait for docker   (1) ensure container `tau2run` (current-root binds +
#   persistent data bind)   (2) ensure tau deps + patches   (3) launch 8 servers
#   (idempotent)   (4) wait health   (5) resume the eval (--auto-resume).
# Idempotent throughout; safe to run any time. Install via aws/tau2eval.service.
set -o pipefail
IMG=slimerl/slime:latest
CON=tau2run
HOME_EC2=/home/ec2-user
DATA=$HOME_EC2/tau2_data            # persistent sim data on the live root
log(){ echo "[$(date -u +%FT%TZ)] [tau2-resume] $*"; }
port_bound(){ timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null; }

log "waiting for docker..."; for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null 2>&1 || { log "FATAL: docker down"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { log "pulling $IMG"; docker pull "$IMG"; }

# (1) container -------------------------------------------------------------
mkdir -p "$DATA"
if docker ps --format '{{.Names}}' | grep -qx "$CON"; then log "$CON running";
elif docker ps -a --format '{{.Names}}' | grep -qx "$CON"; then log "starting $CON"; docker start "$CON" >/dev/null;
else
  log "creating $CON (current-root binds)"
  docker run -d --name "$CON" --gpus all --network host --ipc host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 --restart unless-stopped \
    -v "$HOME_EC2/slime:$HOME_EC2/slime" \
    -v "$HOME_EC2/hf_cache:/root/.cache/huggingface" \
    -v "$DATA:$HOME_EC2/tau2-bench/data" \
    "$IMG" sleep infinity >/dev/null
fi

# (2) tau deps + patches (idempotent) ---------------------------------------
docker exec "$CON" bash -lc "$HOME_EC2/slime/aws/_setup_tau_deps.sh" >>/tmp/tau2_setup.log 2>&1 || log "WARN: setup_tau_deps issue"

# (3) launch 8 servers ------------------------------------------------------
# port|gpu|model|served_name|mem|extra
SERVERS=(
  "7000|0|Qwen/Qwen3-8B-Base|qwen-8b-base|0.85|--tool-call-parser qwen"
  "7001|1|willhx/Qwen3-8B-Base-Math|Qwen3-8B-Base-Math|0.85|--tool-call-parser qwen"
  "7002|2|willhx/Qwen3-8B-Base-Math-SeaSFT-Search|Qwen3-8B-Base-Math-SeaSFT-Search|0.85|--tool-call-parser qwen"
  "7003|3|willhx/Qwen3-8B-Base-Math-SeaSFT-Search|Qwen3-8B-Base-Math-SeaSFT-Search|0.85|--tool-call-parser qwen"
  "7006|4|zai-org/GLM-4.7-Flash|GLM-4.7-Flash|0.85|--tool-call-parser glm47 --reasoning-parser glm45"
  "7007|5|zai-org/GLM-4.7-Flash|GLM-4.7-Flash|0.85|--tool-call-parser glm47 --reasoning-parser glm45"
  "7008|6|zai-org/GLM-4.7-Flash|GLM-4.7-Flash|0.85|--tool-call-parser glm47 --reasoning-parser glm45"
  "7009|7|zai-org/GLM-4.7-Flash|GLM-4.7-Flash|0.85|--tool-call-parser glm47 --reasoning-parser glm45"
)
docker exec "$CON" bash -lc "mkdir -p $HOME_EC2/slime/aws/logs"
for spec in "${SERVERS[@]}"; do
  IFS='|' read -r port gpu model name mem extra <<< "$spec"
  if port_bound "$port"; then log ":$port already bound — skip"; continue; fi
  log "launch :$port gpu=$gpu ($name)"
  docker exec -d "$CON" bash -lc "CUDA_VISIBLE_DEVICES=$gpu nohup python -m sglang.launch_server \
    --model-path '$model' --served-model-name '$name' --host 0.0.0.0 --port $port \
    --tp 1 --mem-fraction-static $mem $extra \
    > $HOME_EC2/slime/aws/logs/server_$port.log 2>&1"
done

# (4) wait for all servers --------------------------------------------------
log "waiting for servers 7000-7003,7006-7009..."
deadline=$(( $(date +%s) + 2400 ))
for port in 7000 7001 7002 7003 7006 7007 7008 7009; do
  until docker exec "$CON" bash -lc "curl -sf -m4 http://127.0.0.1:$port/health >/dev/null 2>&1"; do
    [ "$(date +%s)" -ge "$deadline" ] && { log "WARN: :$port not ready"; break; }
    sleep 8
  done
done
log "servers up."

# (5) resume the eval (idempotent; --auto-resume skips completed) ------------
if docker exec "$CON" bash -lc 'pgrep -f "[r]un_tau2_resume_all.sh" >/dev/null'; then
  log "eval already running — not relaunching."
else
  docker exec -d "$CON" bash -lc "cd $HOME_EC2/slime/eval_scai && nohup bash run_tau2_resume_all.sh > logs/tau2_rp/MASTER_resume.log 2>&1"
  log "tau2 resume-all (re)launched -> eval_scai/logs/tau2_rp/MASTER_resume.log"
fi
log "DONE."
