#!/bin/bash
# ============================================================================
# tau3 eval auto-resume after a spot interruption — brings the WHOLE experiment
# back up on a NEW machine, then (re)launches the eval. Idempotent throughout;
# safe to run at every boot and any time by hand.
#
# Anchored on the DATA EBS volume mounted at /mnt/old-data3 — repo, hf_cache,
# tau2-bench, AND the docker image (via docker data-root) all live there, so a
# fresh box needs NO downloads (no image pull, no model download).
#
# Steps:  0) mount /mnt/old-data3   1) point docker data-root at it
#         2) wait docker + ensure image   3) ensure container tau3-srv
#         4) ensure tau2 installed+patched   5) wait GPUs visible in container
#         6) launch 6 servers (4 agents tp1 + 2 GLM tp2)   7) wait health
#         8) (re)launch run_tau3.sh (--auto-resume skips finished cells)
#
# Install as a boot service with aws/install_tau3_autostart.sh.
# ============================================================================
set -o pipefail
DATA=/mnt/old-data3
ROOT=$DATA/home/ec2-user
IMG=slimerl/slime:latest
CON=tau3-srv
EVALDIR=/home/ec2-user/slime/eval_scai/tau3          # path INSIDE the container
HOSTEVAL=$ROOT/slime/eval_scai/tau3                  # same dir on the host
NGPU="${NGPU:-8}"
log(){ echo "[$(date -u +%FT%TZ)] [tau3-resume] $*"; }
port_bound(){ timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null; }

# --- 0) ensure the data volume is mounted at /mnt/old-data3 ------------------
if ! mountpoint -q "$DATA"; then
  log "$DATA not mounted; probing xfs partitions for the tau3 data..."
  mkdir -p "$DATA" /mnt/_tau3probe
  for dev in $(lsblk -prno NAME,FSTYPE | awk '$2=="xfs"{print $1}'); do
    mountpoint -q "$DATA" && break
    mount -o nouuid,ro "$dev" /mnt/_tau3probe 2>/dev/null || continue
    if [ -e /mnt/_tau3probe/home/ec2-user/slime/eval_scai/tau3/run_tau3.sh ]; then
      umount /mnt/_tau3probe 2>/dev/null
      mount -o nouuid "$dev" "$DATA" && log "mounted data volume $dev at $DATA"
    else
      umount /mnt/_tau3probe 2>/dev/null
    fi
  done
fi
mountpoint -q "$DATA" || { log "FATAL: $DATA not mounted — attach the data EBS volume and retry"; exit 1; }

# --- 1) point docker data-root at the data volume, then ALWAYS restart docker
#        so it binds the *now-mounted* volume. dockerd is started by systemd at
#        boot, which on a spot resume happens BEFORE this script mounts the data
#        volume — so dockerd comes up on an empty data-root and can't see the
#        cached image (the image looks "missing" -> a pointless pull, or the
#        container fails to create). Restarting docker here, after the mount, is
#        what makes the cached image visible again. Cheap and idempotent. -------
DJ=/etc/docker/daemon.json
log "ensuring docker data-root -> $DATA/var/lib/docker and restarting docker on the mounted volume"
systemctl stop docker docker.socket 2>/dev/null; sleep 2
python3 - "$DJ" "$DATA/var/lib/docker" <<'PY'
import json,sys,os
p,dr=sys.argv[1],sys.argv[2]
try: c=json.load(open(p))
except Exception: c={}
c["data-root"]=dr
c.setdefault("runtimes",{}).setdefault("nvidia",{"path":"nvidia-container-runtime","args":[]})
os.makedirs(os.path.dirname(p),exist_ok=True)
json.dump(c,open(p,"w"),indent=2)
PY
systemctl start docker; sleep 4

# --- 2) wait for docker; ensure image ---------------------------------------
log "waiting for docker daemon..."
for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null 2>&1 || { log "FATAL: docker unavailable"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { log "image missing -> docker pull $IMG"; docker pull "$IMG" || { log "FATAL: image $IMG not present and pull failed (data-root/mount issue?)"; exit 1; }; }

# --- 3) ensure container tau3-srv -------------------------------------------
if docker ps --format '{{.Names}}' | grep -qx "$CON"; then log "$CON running"
elif docker ps -a --format '{{.Names}}' | grep -qx "$CON"; then log "docker start $CON"; docker start "$CON" >/dev/null
else
  log "creating $CON (mounts from $ROOT, HF offline)"
  docker run -d --name "$CON" --gpus all --network host --ipc host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 --restart unless-stopped \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$ROOT/hf_cache:/root/.cache/huggingface" \
    -v "$ROOT/slime:/home/ec2-user/slime" \
    -v "$ROOT/tau2-bench:/home/ec2-user/tau2-bench" \
    "$IMG" sleep infinity >/dev/null
fi

# --- 4) ensure tau2 installed + patched (idempotent) ------------------------
docker exec "$CON" bash -lc '
  python -c "import tau2" 2>/dev/null || pip install -e /home/ec2-user/tau2-bench >/tmp/tau2_install.log 2>&1
  python -c "import rank_bm25" 2>/dev/null || pip install rank_bm25 >>/tmp/tau2_install.log 2>&1   # banking_knowledge BM25 retrieval dep
  python /home/ec2-user/slime/aws/_patch_tau2_lenient_args.py   /home/ec2-user/tau2-bench
  python /home/ec2-user/slime/aws/_patch_tau2_first_toolcall.py /home/ec2-user/tau2-bench
' >/tmp/tau3_setup.log 2>&1 || log "WARN: tau2 setup hit an issue (see /tmp/tau3_setup.log)"

# --- 5) wait until the container can see the GPUs ---------------------------
for i in $(seq 1 30); do
  docker exec "$CON" bash -lc "python -c \"import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>=$NGPU else 1)\"" 2>/dev/null && break
  log "GPUs not ready in container yet ($i)..."; sleep 5
done

# --- 6) launch the 6 servers (idempotent; serve script skips healthy ports) --
log "launching servers (4 agents tp1 + 2 GLM tp2)..."
docker exec "$CON" bash -lc "bash $EVALDIR/serve_tau3_fleet.sh" 2>&1 | sed 's/^/[serve] /'

# --- 7) wait for all 6 servers ----------------------------------------------
log "waiting for servers 7000-7003, 7006-7007..."
deadline=$(( $(date +%s) + 2700 ))
ready=0
for port in 7000 7001 7002 7003 7006 7007; do
  until docker exec "$CON" bash -lc "curl -sf -m4 http://127.0.0.1:$port/health >/dev/null 2>&1"; do
    [ "$(date +%s)" -ge "$deadline" ] && { log "WARN: :$port not ready before timeout"; break; }
    sleep 8
  done
  if docker exec "$CON" bash -lc "curl -sf -m4 http://127.0.0.1:$port/health >/dev/null 2>&1"; then
    log ":$port ready"; ready=$((ready+1))
  fi
done
if [ "$ready" -lt 6 ]; then
  log "FATAL: only $ready/6 servers healthy — NOT launching eval. Inspect $HOSTEVAL/logs/fleet/server_*.log"
  exit 1
fi
log "all 6 servers healthy"

# --- 8) (re)launch the eval (idempotent; tau2 --auto-resume skips done) ------
if docker exec "$CON" bash -lc 'pgrep -f "[r]un_tau3.sh" >/dev/null'; then
  log "eval already running — not relaunching."
else
  log "launching run_tau3.sh (detached)"
  docker exec -d "$CON" bash -lc "
    export TAU2=/home/ec2-user/tau2-bench
    export LOGD=$EVALDIR/logs
    export OUTD=$EVALDIR/tau3_rp_results
    cd $EVALDIR
    nohup bash run_tau3.sh > logs/run_tau3.nohup 2>&1 &"
fi
log "DONE. servers up + tau3 eval running. Watch: docker exec $CON tail -f $EVALDIR/logs/MASTER.log"
