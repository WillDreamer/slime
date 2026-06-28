# tau3 eval — spot-interruption auto-resume

Adapted from the old `aws.zip` / `slime/aws/autostart-tau2.sh` so a **fresh
spot machine** automatically brings the whole tau3 experiment back up and
relaunches the eval. Everything is anchored on the `/mnt/old-data3` data EBS
volume, so no downloads are needed on a new box.

## Files
| File | Role |
|---|---|
| `autostart-tau3.sh` | the bringup: mount data vol → set docker data-root → container → servers → eval (idempotent) |
| `tau3eval.service` | systemd oneshot that runs the autostart at boot |
| `install_tau3_autostart.sh` | copies the script to `/usr/local/bin`, installs+enables the service |

## What the autostart does (all idempotent)
0. Mounts the data volume at `/mnt/old-data3` (probes xfs partitions for the tau3 marker).
1. Sets docker `data-root` → `/mnt/old-data3/var/lib/docker` (reuses the baked `slimerl/slime:latest` image — **no pull**).
2. Waits for docker; ensures the image.
3. Ensures container `tau3-srv` (mounts `hf_cache`, `slime`, `tau2-bench` from the data volume; `HF_HUB_OFFLINE=1`).
4. Ensures `tau2` is installed + patched (lenient-args, first-tool-call).
5. Waits until the container sees the GPUs.
6. Launches the 6 servers via `serve_tau3_fleet.sh` — 4 agents (tp1, GPU0-3, ports 7000-3) + 2 GLM-4.7-Flash (tp2, GPU4-5 & 6-7, ports 7006-7), GLM tp2 because it's a 59G MoE that won't fit on one 80G H100.
7. Waits for all 6 `/health`.
8. (Re)launches `run_tau3.sh` detached — `tau2 --auto-resume` skips already-finished `tau3rp_*` cells, so an interrupted run continues where it left off.

## Install (once per machine, or bake into the AMI)
```bash
sudo /mnt/old-data3/home/ec2-user/slime/eval_scai/tau3/aws/install_tau3_autostart.sh
# or install AND start immediately:
sudo /mnt/old-data3/home/ec2-user/slime/eval_scai/tau3/aws/install_tau3_autostart.sh start
```
After a spot stop/start the service fires on boot and the eval resumes by itself.

## Prerequisite on a brand-new machine
The data EBS volume must be **attached** to the instance (the script mounts it,
but cannot attach it). Attach it in the AWS console / CLI, or include it in the
launch template's block-device mapping. Everything else (image, weights, repo,
patches) is already on that volume.

## Manual run / watch
```bash
sudo /usr/local/bin/autostart-tau3.sh          # run the bringup by hand
docker exec tau3-srv tail -f /home/ec2-user/slime/eval_scai/tau3/logs/MASTER.log
python3 /mnt/old-data3/home/ec2-user/slime/eval_scai/tau3/aggregate_tau3.py
```
