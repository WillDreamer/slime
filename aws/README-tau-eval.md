# Auto-resume of the tau1 + tau2 eval on a NEW machine

This extends `autostart-resume.sh` (run on boot by `searcheval.service`) to also
resume the **tau-bench evals** after a spot stop/start, with zero manual steps.

## What gets resumed

Four checkpoints (one per agent server), each run through **tau1** (training-
aligned, `examples/tau-bench` harness) then **tau2** (official
`sierra-research/tau2-bench`):

| agent | port/GPU | user-sim (GLM) |
|---|---|---|
| qwen-8b-base | :7000 / GPU3 | :7006 |
| Qwen3-8B-Base-Math | :7001 / GPU4 | :7007 |
| …-SeaSFT-Search | :7002 / GPU5 | :7008 |
| …-TauSFT-Tau (tau) | :7003 / GPU6 | :7006 |

All agent servers run at **native 32k (no YaRN)**; the 3 GLM user-sims run with
**`--tool-call-parser glm47 --reasoning-parser glm45`** so the user-sim's
reasoning goes to `reasoning_content` and the customer message (`content`) is
clean. (Without the reasoning parser the trace leaks into the message and
corrupts the sim — that's the one config the resume MUST keep.)

## Config (matches training where possible)

- temp 0 (agent + user); tau1 `max_response_len` 2048, `max_num_steps` 30
- tau2: `num_trials=4` (pass^k), `max_steps=200`, `max_tokens=2048`,
  `max_concurrency=16`, 3 outer runs
- tool-call parser = `Qwen25Detector` (same as training's `qwen25` adapter)

## How resume works (idempotent)

`autostart-resume.sh` step 4, after the servers answer:
1. `docker exec slime _setup_tau_deps.sh` — re-clones/installs `tau_bench`
   (JD-ETH `feature/litellm-retry`) and official `tau2-bench` **only if missing**
   (no-op on an intact AMI transfer; repair on a fresh container).
2. `docker exec slime _run_tau.sh` — (re)launches `eval_scai/run_parallel.sh`
   unless it's already running.

`run_parallel.sh` runs 4 lanes in parallel (one per model: tau1 → tau2 on its own
agent server). Resume safety:
- **tau1**: `run_tau1.sh` skips any `tau1_<env>.json` already written at temp 0.
- **tau2**: `run_tau2_official.sh` / `run_tau2_tau.sh` rely on `tau2 run --save-to`,
  which **auto-resumes** an existing simulation dir (completed trials skipped).

So a mid-experiment interruption picks up where it left off; a finished
experiment is a no-op.

## Results land in (all on the transferred root volume)

- tau1: `eval_scai/results/<model>/run<i>/tau1_<env>.json`
- tau2: `tau2-bench/data/simulations/tau2off_<model>_<domain>_run<i>/results.json`
- aggregate both: `docker exec slime python eval_scai/aggregate_tau.py`

## Watch / operate

```bash
docker exec slime tail -f /home/ec2-user/slime/eval_scai/logs/MASTER_parallel.log
docker exec slime tail -f /home/ec2-user/slime/eval_scai/logs/tau1/par_<model>.log
docker exec slime tail -f /home/ec2-user/slime/eval_scai/logs/tau2/par_<model>.log
docker exec slime bash /home/ec2-user/slime/aws/_run_tau.sh   # force (re)launch now
```

## Files (this folder)

| File | Role |
|---|---|
| `_setup_tau_deps.sh` | ensure tau_bench + tau2-bench in the slime container (idempotent) |
| `_run_tau.sh` | (re)launch `run_parallel.sh` unless already running |
| `autostart-resume.sh` | step 4 calls the two above after servers are up |

## Caveats

- `tau_bench` and `tau2-bench` live **container-local** (`/home/ec2-user/tau-bench`,
  `/home/ec2-user/tau2-bench`), so they survive an AMI/container transfer but are
  re-cloned by `_setup_tau_deps.sh` if the container is recreated fresh (needs
  network + GitHub).
- The `.venv_inspect` (old inspect-port tau2) is unused by this path and not
  required for resume.
