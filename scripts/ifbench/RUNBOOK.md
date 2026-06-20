# IFBench RL Training — Runbook

Qwen3-8B (dense) instruction-following RL post-training with slime, using the
**IFEval-G rule reward** plus an optional **Skywork reward-model judge**, and a
held-out **IFBench** eval. This runbook covers (1) what the training does,
(2) how to run it, (3) checkpoint/resume behavior, and (4) operations
(changing run length, monitoring, troubleshooting).

> For the per-file manifest of the `slime_scai` port and external dependencies,
> see `SETUP.md` in this directory. This runbook is the operational guide.

---

## 0. Two copies of this setup

| Repo | Purpose | State |
|---|---|---|
| `…/multi/slime` | the **live** training repo (currently running) | running, heading to rollout 1500 |
| `…/multi/slime_scai` | a **self-contained** copy (this repo) | ready to run; isolated paths |

They use **distinct** checkpoint dirs and Ray temp dirs, so both can exist
without colliding. Everything below applies to either — just match the working
directory (`-w`) and the container name to the repo you mean to run.

---

## 1. What the training does

- **Model:** Qwen3-8B dense (`scripts/models/qwen3-8B.sh`), started from a
  multi-stage SFT checkpoint (`--ref-load`), which is also the frozen KL
  reference.
- **Algorithm:** GRPO (RM variant) / GSPO (baseline), low-variance KL,
  TIS train/infer mismatch correction, constant LR `1e-6`.
- **Rollout:** 256 prompts × 8 samples/prompt = 2048 samples per rollout;
  global batch 2048 → **1 optimizer step per rollout**. `--num-rollout` is the
  **absolute** target (`for rollout_id in range(start, num_rollout)`).
- **Reward (training data, `rm_type="multi"`):**
  1. Rule-based IFEval-G check (`rm_hub/ifeval.py` → `ifeval_g/`).
  2. **Only if** the rule reward > 0 **and** `--rm-judge-url` is set, the
     response is additionally scored by the Skywork RM (`/classify`), and the
     two are combined (`rm_hub/__init__.py::_apply_rm_judge_reward`):
     - rule>0 and rm_score >  threshold → reward **+1.0**
     - rule>0 and rm_score <= threshold → reward **−0.5**
     - rule≤0 → unchanged (RM never queried)
- **Eval (held-out, `rm_type="ifbench"`):** every `--eval-interval` rollouts,
  the **current** policy generates on `IFBench_eval.jsonl` and is scored by the
  official IFBench **strict** checker (all-or-nothing per prompt, raw output —
  no markdown/line cleaning, no loose variant). See `rm_hub/ifbench.py`.

---

## 2. Prerequisites (external — not in this repo)

| Need | Where |
|---|---|
| Docker image `slime-ifbench:latest` | bundles slime deps + Megatron-LM (`/root/Megatron-LM/`) + sglang |
| HF base model | `/xuanwu-tank/center/whx/Qwen3-8B-Base` |
| SFT/ref checkpoint | `/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau` |
| Skywork RM (RM variant only) | `Skywork/Skywork-Reward-V2-Llama-3.1-8B` (HF) |
| IFBench eval repo | sibling of repo root: `…/multi/IFBench` (auto-cloned if absent) |
| GPUs | 8× (trainer colocates all 8) |

---

## 3. How to run

### Step 1 — (RM variant only) start the Skywork reward server *first*
On the RM node / a free GPU:
```bash
HOST=0.0.0.0 PORT=30000 bash scripts/ifbench/launch_reward_model.sh
```
Knobs (env vars): `MODEL_PATH`, `HOST`, `PORT`, `CONTEXT_LENGTH` (16384),
`MEM_FRACTION` (0.3), `TP` (2), `RM_GPUS` (2,3). Wait until it is serving before
the first rollout completes (otherwise those samples score by rule reward only).

### Step 2 — start training (RM variant)
```bash
docker run -d --name ifbench_rm_scai --gpus all --network host --ipc host \
  -v /xuanwu-tank:/xuanwu-tank \
  -w /xuanwu-tank/north/xw27/multi/slime_scai \
  slime-ifbench:latest \
  -lc "bash scripts/ifbench/run-qwen3-8B_ifbench_rm.sh \
       > /xuanwu-tank/north/xw27/multi/slime_scai/scripts/ifbench/ifbench_rm_run.log 2>&1"
```
- `-w …/slime_scai` makes `python3 train.py` import the **local** `slime/`
  package (with the merged RM-judge code), not any image-installed copy.
- Override the RM endpoint if needed: add `-e RM_JUDGE_URL=http://<host>:<port>/classify`.

**Baseline (no RM):** same command with `run-qwen3-8B_ifbench.sh` and a
different `--name` (e.g. `ifbench_scai`). No reward server required.

### Step 3 — watch it come up
```bash
tail -f scripts/ifbench/ifbench_rm_run.log
```
Healthy startup sequence: Ray starts → Megatron loads checkpoint
(`successfully loaded checkpoint … at iteration N`) → `rollout N+1: {...}` lines
begin. Eval lines appear every `--eval-interval` rollouts.

---

## 4. Checkpoints & resume (important)

- **Save cadence:** `--save-interval 30` → checkpoints at iter 30, 60, … Only the
  retained ones (`--save-retain-interval 210`) survive pruning, plus the latest.
- **Resume is automatic and state-complete.** When `--load` points at a dir
  containing `latest_checkpointed_iteration.txt`, slime resumes:
  - model weights ✓
  - **Adam optimizer momentum** ✓ (`no_load_optim` stays false)
  - `rollout_id` continues (`start = loaded_iter + 1`)
  - **data iterator position** continues (next prompts, not a replay)
  - LR is constant `1e-6`, so unaffected
- **Fresh start vs resume:** if `--load` is empty / has no checkpoint, slime sets
  `--finetune` + `no_load_optim` and loads weights only from `--ref-load`
  (Adam reset, iteration 0). The scripts deliberately omit `--finetune` so the
  *same script* resumes when a checkpoint exists and starts fresh when it doesn't.
- **Restart cost:** restarting loses only the rollouts since the last save
  (≤ one save-interval, i.e. ≤30 rollouts) — never the whole run.

### Splitting a run (e.g. 500 + 500) vs one 1000-step run
Equivalent **in expectation** — resume restores weights, Adam, and data
position. Not bit-identical, because generation is stochastic
(`temperature 1`, deterministic inference not enabled) — but that's the same
noise any two runs have. No double-counting of data, no LR discontinuity.

---

## 5. Operations

### Change the run length (`--num-rollout`)
`--num-rollout` is the absolute target, so to extend a finished/running job:
1. Edit `--num-rollout` in `scripts/ifbench/run-qwen3-8B_ifbench_rm.sh`.
2. Relaunch (the running process won't pick up the edit; it must be restarted).
   Resume continues from the latest checkpoint toward the new target.
3. `--override-opt_param-scheduler` is already set, so the LR-scheduler's
   total-iteration count changing (e.g. `num_rollout × 256 × 8`) does **not**
   trip Megatron's assert; constant LR makes the override a no-op for the LR.

### Restart cleanly (stop running container, drain GPUs, relaunch)
```bash
docker stop -t 60 ifbench_rm_scai && docker rm -f ifbench_rm_scai
# wait until `nvidia-smi --query-compute-apps=pid --format=csv,noheader` is empty
docker run -d --name ifbench_rm_scai … (as in Step 2)
```
> If you reuse the container name, `docker rm -f` it first — a leftover
> `Exited` container will cause `docker run` to fail with a name conflict.

### Monitor progress
```bash
# latest saved checkpoint
cat /xuanwu-tank/north/xw27/model/Qwen3-8B_ifbench_rm_slime_scai/latest_checkpointed_iteration.txt
# live rollout / eval
grep -aoE "rollout [0-9]+" scripts/ifbench/ifbench_rm_run.log | tail -1
```
Eval (IFBench strict accuracy) and reward curves are also logged to wandb
(project `slime-ifbench`).

---

## 6. Key paths (this `slime_scai` copy)

| Item | Path |
|---|---|
| Train data | `examples/ifbench/IF_multi_constraints_upto5_ifbench_en.jsonl` |
| Eval data | `examples/ifbench/IFBench_eval.jsonl` |
| Checkpoints / save dir | `/xuanwu-tank/north/xw27/model/Qwen3-8B_ifbench_rm_slime_scai/` |
| Ray temp | `/xuanwu-tank/north/xw27/ray_temp_scai` (wiped at startup) |
| Run log | `scripts/ifbench/ifbench_rm_run.log` |
| RM endpoint | `RM_JUDGE_URL` (default `http://131.179.168.120:30000/classify`) |

(The live `…/multi/slime` repo uses the same names **without** the `_scai`
suffix — that is what keeps the two runs isolated.)

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `docker run` name conflict | leftover `Exited` container — `docker rm -f <name>` then relaunch |
| Startup OOM | GPUs not drained from a previous job — wait until `nvidia-smi` compute-apps is empty before relaunch |
| Rewards near 0 / RM not applied | RM server not reachable — `_query_rm_judge` falls back to 0.0 (rule reward only). Confirm `RM_JUDGE_URL` and that the server is up |
| Megatron scheduler assert on resume | total-iteration count changed (e.g. `num_rollout`/`n-samples`) — ensure `--override-opt_param-scheduler` is set (it is) |
| Adam state not loaded | `--finetune` set, or `--load` empty/HF-only — point `--load` at the Megatron checkpoint dir and drop `--finetune` |
| Eval seems too strict | by design: IFBench uses the **strict** checker on raw output (no markdown/line cleaning, no loose variant) |

---

## 8. What was done to create this `slime_scai` setup

`slime_scai` is a separate clone of `WillDreamer/slime.git` (its own
OPD/critic/parsing history) that **lacked** the ifbench RM-judge feature. It was
added by a **surgical merge** (no shared git history with the live repo, so this
was file-level, additive — nothing of the fork's own code was removed):

- **`slime/rollout/rm_hub/__init__.py`** — added `import re`, `_query_rm_judge()`,
  `_apply_rm_judge_reward()`, and the `("ifeval","multi")` dispatch branch.
- **`slime/utils/arguments.py`** — added `--rm-judge-url`, `--rm-judge-threshold`.
- **Added** `slime/rollout/rm_hub/ifeval.py` + `ifeval_g/` (training rule reward);
  **aligned** `ifbench.py` (robust `record_id` key).
- **Copied** `scripts/ifbench/` (run scripts, RM launcher, README) and
  `examples/ifbench/` data, with all absolute paths **rewritten** to `slime_scai`
  and isolated `_scai` checkpoint/Ray dirs.
- Verified: `py_compile` passes; dispatch + args present; data byte-identical.

See `SETUP.md` for the full file-by-file manifest and the external-dependency
table.
