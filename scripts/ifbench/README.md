# Qwen3-8B IFBench RL (with optional reward-model judge)

Instruction-following RL post-training for **Qwen3-8B (dense)**, started from the
multi-stage SFT checkpoint, training with the **IFEval-G rule-based reward** and
optionally a **Skywork generative reward-model judge**.

* **Train data:** `examples/ifbench/IF_multi_constraints_upto5_ifbench_en.jsonl`
  — each row carries `metadata.rm_type="multi"` and an `instruction_id_list`;
  the reward is computed by the vendored IFEval-G checkers.
* **Eval data:** `examples/ifbench/IFBench_eval.jsonl` — scored with the held-out
  IFBench 58-instruction reward (`rm_type="ifbench"`).
* **Init checkpoint (SFT):**
  `/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT`
  (Megatron `.distcp`, latest iter 380). Produced by whx's multi-stage SFT
  pipeline (`run_qwen3_8B.sh`, Math → SeaSFT → Search → TauSFT).
* **HF base (sglang/tokenizer):** `/xuanwu-tank/center/whx/Qwen3-8B-Base`.
* Adapted from the 30B-A3B reference `scripts/run-qwen3-30B-A3B_ifbench_v2.sh`,
  swapping the MoE model for the dense 8B (`scripts/models/qwen3-8B.sh`, no
  expert/EP args) and pointing at the 8B SFT checkpoint.

---

## Files in this directory

| File | Purpose |
|------|---------|
| `run-qwen3-8B_ifbench.sh` | **Baseline RL** — IFEval-G rule reward only. No reward model. Single self-contained `ray job submit` on 8 GPUs. |
| `launch_reward_model.sh` | **Reward-model server** — starts the Skywork reward model as an sglang `/classify` endpoint that the trainer queries. Run this *first* when using the RM variant. |
| `run-qwen3-8B_ifbench_rm.sh` | **RL + reward model** — same as the baseline but adds `--rm-judge-url` / `--rm-judge-threshold`, so responses that pass the rule check are additionally scored by the Skywork RM. |
| `README.md` | This document. |

---

## How the reward model is wired in (relevant codebase)

All reward dispatch lives in **`slime/rollout/rm_hub/__init__.py`**:

* **`async_rm(args, sample)`** — for `rm_type ∈ {"ifeval", "multi"}` it calls
  `compute_ifeval_reward(...)` (rule-based, **`slime/rollout/rm_hub/ifeval.py`**
  → `ifeval_g/`). If `--rm-judge-url` is set **and** the rule reward is > 0, it
  then queries the reward model and combines the two scores.
* **`_query_rm_judge(url, prompt, response)`** — extracts the raw user text from
  the Qwen3 chat-formatted prompt, re-wraps it in the **Llama-3.1 chat template**
  (the Skywork model's template), POSTs `{"text": ...}` to the server's
  `/classify` endpoint, and reads the scalar back from `result["embedding"][0]`.
  On any failure it returns `0.0` so training is never blocked.
* **`_apply_rm_judge_reward(verified, rm_score, threshold)`** — combines them:
  * `verified > 0` and `rm_score >  threshold` → `verified + 1.0`
  * `verified > 0` and `rm_score <= threshold` → `verified - 0.5`
  * `verified <= 0` → `verified` (unchanged — the RM is never even queried)
* The held-out **eval** reward (`rm_type="ifbench"`) is computed by
  **`slime/rollout/rm_hub/ifbench.py`** and is independent of the RM judge.

CLI flags are registered in **`slime/utils/arguments.py`**
(`add_reward_model_arguments`): `--rm-type`, `--rm-judge-url`,
`--rm-judge-threshold` (argparse default `0.5`).

**Correctness check (done):** the dispatch path, the `/classify` request/response
handling, and the argument wiring are all consistent — `--rm-judge-url`/
`--rm-judge-threshold` are parsed, read via `getattr(args, ...)`, and only the
`ifeval`/`multi` branch consumes them, exactly matching the training data's
`rm_type="multi"`. No code changes were required. One thing to know: the RM is
only ever applied to samples that **already pass** the rule check, so it acts as
a *quality refinement* on correct responses, not as the primary signal.

---

## How to run

### A) Baseline (no reward model)

```bash
cd /xuanwu-tank/north/xw27/multi/slime
bash scripts/ifbench/run-qwen3-8B_ifbench.sh
```

Writes checkpoints to `/xuanwu-tank/north/xw27/model/Qwen3-8B_ifbench_slime/`,
logs to wandb project `slime-ifbench`, group `qwen3-8B-ifbench`.

### B) RL with the Skywork reward model

This needs **two processes**. The trainer colocates across all 8 GPUs, so the
reward model needs its own space — pick one layout:

**Layout 1 — separate node (recommended).** On a second machine:
```bash
HOST=0.0.0.0 PORT=30000 bash scripts/ifbench/launch_reward_model.sh
```
Then on the training node, raise the trainer's mem fraction back to 0.7 (edit
`SGLANG_ARGS` in the RM script) and start training, pointing at the RM host:
```bash
cd /xuanwu-tank/north/xw27/multi/slime
RM_JUDGE_URL="http://<rm-node-ip>:30000/classify" \
  bash scripts/ifbench/run-qwen3-8B_ifbench_rm.sh
```

**Layout 2 — same node, GPUs 2 and 3 (default).** Start training first (it grabs
all 8 GPUs with a reduced 0.6 mem fraction), then launch the RM pinned to GPUs
2,3 (TP=2) with a small mem fraction so it fits alongside the trainer's engine on
those two GPUs:
```bash
cd /xuanwu-tank/north/xw27/multi/slime
# shell 1: trainer (already uses --sglang-mem-fraction-static 0.6)
bash scripts/ifbench/run-qwen3-8B_ifbench_rm.sh
# shell 2: reward model on GPUs 2,3 — these are the script defaults
bash scripts/ifbench/launch_reward_model.sh
```
The RM defaults (`RM_GPUS=2,3`, `TP=2`, `MEM_FRACTION=0.3`,
`RM_JUDGE_URL=http://127.0.0.1:30000/classify`) already match the trainer.

> The reward model query only fires for rollouts that pass the rule check, so a
> brief unavailability of the RM at startup just means those few samples score by
> the rule reward alone (the query falls back to `0.0`). For clean results, have
> the RM server **ready before** the first rollout completes.

### Reward-model server knobs (`launch_reward_model.sh`)

All overridable via env vars:

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_PATH` | `Skywork/Skywork-Reward-V2-Llama-3.1-8B` | Pulled from HF on first run (not cached locally). Set a local path to avoid the download. |
| `HOST` | `127.0.0.1` | Use `0.0.0.0` to accept connections from another node. |
| `PORT` | `30000` | Must match `RM_JUDGE_URL` in the trainer. |
| `CONTEXT_LENGTH` | `16384` | Prompt + response budget for scoring. |
| `MEM_FRACTION` | `0.3` | Small, because it shares GPUs 2,3 with the trainer. Raise to `0.9` on a dedicated node. |
| `TP` | `2` | Tensor-parallel size — matches the two GPUs below. |
| `RM_GPUS` | `2,3` | `CUDA_VISIBLE_DEVICES` for the server. |

### Trainer RM knobs (`run-qwen3-8B_ifbench_rm.sh`)

| Var | Default | Notes |
|-----|---------|-------|
| `RM_JUDGE_URL` | `http://127.0.0.1:30000/classify` | Endpoint of the reward server. |
| `RM_JUDGE_THRESHOLD` | `0.0` | `alpha` in `_apply_rm_judge_reward`; reward `+1` above it, `-0.5` at/below it. |

---

## Key hyperparameters (shared by both training scripts)

* GSPO advantage, low-variance KL (`--kl-loss-coef 0.1`), entropy `0.01`,
  clip `0.2`/`0.28`, TIS train/infer mismatch correction.
* `lr 5e-7` constant, Adam (β `0.9`/`0.98`), CPU-offloaded precision-aware optimizer.
* Rollout: batch 16 × 8 samples/prompt, response len 8192, 500 rollouts,
  global batch 128.
* Parallelism (8B dense): TP=4, PP=1, CP=1, full recompute, dynamic batch
  (`max-tokens-per-gpu 18432`). No MoE/EP args (the 30B reference's
  `--expert-*` / `--sglang-ep-size` flags are intentionally dropped).
* Eval every 10 rollouts on IFBench (4 samples/prompt).
