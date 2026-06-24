# Auto-resume on a NEW machine (AMI / volume transfer)

When a spot interruption moves the experiment to a fresh instance launched from
an **AMI / root-volume snapshot** of this box, everything is already present on
disk — the `slimerl/slime:latest` image, the 3 containers, and all
`/home/ec2-user` data (the 61 GB FAISS index, ~320 GB of model weights in
`hf_cache`, the micromamba faiss env, the slime repo, the eval results). On boot,
systemd (`searcheval.service`) runs `aws/autostart-resume.sh`, which brings the
whole topology back and resumes the eval. Zero manual steps.

## Topology it restores

| Port | GPU | Server | Container |
|---|---|---|---|
| 8000 | 0 | Skywork reward model (`--is-embedding`) | slime |
| 7005 | 0 | Qwen3-32B | slime |
| 7000 | 3 | Qwen/Qwen3-8B-Base (`qwen-8b-base`) | slime-srv |
| 7001 | 4 | willhx/Qwen3-8B-Base-Math | slime-srv |
| 7002 | 5 | willhx/…-SeaSFT-Search | slime-srv |
| 7003 | 6 | willhx/…-SeaSFT-Search-TauSFT-Tau | slime-srv |
| 7006 | 1 | zai-org/GLM-4.7-Flash (`--tool-call-parser glm47`) | slime-srv |
| 7007 | 7 | zai-org/GLM-4.7-Flash (`--tool-call-parser glm47`) | slime-srv |
| 7008 | 2 | zai-org/GLM-4.7-Flash (`--tool-call-parser glm47`) | slime-srv |

The Search-R1 eval is COMPLETE (12 runs — see
`Search_data/eval_results/<model>/run*/search_full.json`). Its FAISS retrievers
(9002/9003, GPU2) were retired to free GPU2 for GLM-4.7-Flash `:7008`. To bring
the eval back: launch retrievers via `aws/_retrievers.sh` on a free GPU and call
`aws/_run_eval.sh` (both still in this folder); they are no longer invoked by
`autostart-resume.sh`.

## What `autostart-resume.sh` does on boot

```
0) wait for docker; pull slimerl/slime:latest only if the image is missing
1) ensure the 3 containers: create (docker run, correct mounts/GPUs/--restart) if
   absent, else start. Mounts:
     slime         -v slime:slime
     slime-srv     + -v hf_cache:/root/.cache/huggingface
     slime-search  + -v Search_data:/data/Search_data
   (+ pip install aiohttp in slime-search if it was freshly recreated)
2) launch every sglang server (table in the script) on its port/GPU — each in its
   OWN process group (setsid), skipping any port already bound (TCP check)
3) launch the 2 FAISS retrievers (aws/_retrievers.sh)
4) wait until the eval-critical servers answer (aws/wait-ready.sh)
5) resume the eval (aws/_run_eval.sh -> run_search_eval_all.sh): completed runs
   are skipped (idempotent). All 12 done -> no-op until new work is queued.
```

Everything is **idempotent and safe to re-run**: bound ports are skipped (a
load-insensitive TCP-connect check, never a duplicate that fights for a GPU),
running containers untouched, a running eval not relaunched.

## Files

| File | Role |
|---|---|
| `autostart-resume.sh` | host entrypoint: containers → all servers → wait → resume eval |
| `_retrievers.sh` | idempotent launch of the 2 FAISS retrievers (own process groups) |
| `wait-ready.sh` | block until 7000–7003 + 9002/9003 answer |
| `_run_eval.sh` / `run_search_eval_all.sh` | resume-safe eval driver |
| `searcheval.service` | systemd boot unit (enabled) |

## Install (already done on this box; re-do per fresh AMI base)

```bash
sudo cp /home/ec2-user/slime/aws/searcheval.service /etc/systemd/system/
sudo systemctl enable searcheval
```

> **Do NOT run `systemctl daemon-reload` while the GPU containers are running** —
> it resets the containers' NVIDIA device cgroup, and new GPU processes then see
> 0 GPUs (existing ones keep working). Editing the unit file + a fresh boot is
> enough; a reload is only needed to update a *running* systemd and can be done
> when no GPU containers are up (or just reboot).

## Watch / operate

```bash
tail -f /home/ec2-user/Search_data/eval_all.log                          # eval driver
docker exec slime-srv    tail -f /home/ec2-user/slime/aws/logs/server_7002.log
docker exec slime-search tail -f /data/Search_data/retriever_logs/retriever_9002.log
sudo systemctl start searcheval     # bring everything up now (idempotent no-op if up)
```

## Verify

Real test = launch a new instance from an AMI of this box and watch
`tail -f /home/ec2-user/Search_data/eval_all.log`: containers come up, servers
launch, wait-ready passes, eval reports completed runs as "complete, skip".

## Caveats

- Assumes the **full root volume / AMI** carries images + containers + data. If
  only a data volume transfers (fresh docker), the script still recreates the
  containers from the image and re-installs aiohttp, but the image must be
  pullable. If the machine is truly blank, the ~320 GB of weights + index must be
  re-staged first — not handled here.
- Edit the `SERVERS` table in `autostart-resume.sh` to change what comes back.
