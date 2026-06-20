# IFBench RL training in `slime_scai` — setup & manifest

This documents everything required to run the **Qwen3-8B IFBench RL** training
(rule-based IFEval-G reward + optional Skywork reward-model judge) from this
`slime_scai` checkout. It was assembled by porting the ifbench feature from the
working `../slime` repo into this (divergent) fork.

> `slime_scai` is a separate clone of `WillDreamer/slime.git` with its own history
> (OPD / critic / two-phase-parsing work). It did **not** originally contain the
> ifbench RM-judge feature; the pieces below were merged in surgically so the
> fork's own code was preserved.

---

## 1. What was added to this repo (the merge)

**Reward code** (`slime/rollout/rm_hub/`):
- `ifeval.py` — **added**. Rule-based IFEval-G reward used in *training*
  (`rm_type="multi"` / `"ifeval"`). Imports only `./ifeval_g` + stdlib.
- `ifeval_g/` — **added**. Vendored instruction checkers
  (`__init__.py`, `instructions.py`, `instructions_registry.py`, `instructions_util.py`).
- `ifbench.py` — **aligned** (1 line: robust `hash()` key for non-numeric
  `record_id`). Held-out IFBench reward used in *eval* (`rm_type="ifbench"`).
- `__init__.py` — **surgically edited** (additive): added `import re`,
  `_query_rm_judge()`, `_apply_rm_judge_reward()`, and the
  `elif rm_type in ("ifeval", "multi"):` dispatch branch. Nothing removed.

**CLI args** (`slime/utils/arguments.py`):
- `add_reward_model_arguments()` — **surgically edited** (additive): added
  `--rm-judge-url` and `--rm-judge-threshold` (default `0.5`). Nothing removed.

**Scripts** (`scripts/ifbench/`):
- `run-qwen3-8B_ifbench_rm.sh` — RL + Skywork RM judge (main).
- `run-qwen3-8B_ifbench.sh` — baseline (rule reward only, no RM).
- `launch_reward_model.sh` — Skywork RM `/classify` server.
- `README.md` — original feature doc (paths refer to the source `../slime`).

**Data** (`examples/ifbench/`):
- `IF_multi_constraints_upto5_ifbench_en.jsonl` (~225M) — train (`rm_type="multi"`).
- `IFBench_eval.jsonl` (~277K) — held-out eval (`rm_type="ifbench"`).

All script paths were rewritten to point at **this** repo (see §4).

---

## 2. External dependencies — NOT in this repo (must be present to run)

| Dependency | What / where | Notes |
|---|---|---|
| **Docker image** `slime-ifbench:latest` | bundles slime deps + **Megatron-LM** (`/root/Megatron-LM/`) + **sglang** | Built from `docker/Dockerfile` (base `nightly-dev-20260225a`). The training/rollout backends live here, not in this tree. |
| **HF base model** | `/xuanwu-tank/center/whx/Qwen3-8B-Base` | sglang rollout weights + tokenizer. Shared path, kept as-is. |
| **SFT/ref checkpoint** | `/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau` (Megatron torch_dist, iter-300) | `--ref-load`: initial weights + frozen KL reference. Shared path, kept as-is. |
| **Skywork reward model** | `Skywork/Skywork-Reward-V2-Llama-3.1-8B` (HF; pulled on first run) | Served via `launch_reward_model.sh`. Only needed for the **RM** variant. |
| **IFBench eval repo** | resolved at `<repo_parent>/IFBench` → `/xuanwu-tank/north/xw27/multi/IFBench` | See §5. A copy also exists at `slime_scai/IFBench/` but the code looks at the *parent dir*. |

---

## 3. How to run

Both processes run in the `slime-ifbench:latest` container with `/xuanwu-tank`
mounted and the working dir set to **this** repo.

### A) Start the Skywork RM server (RM variant only) — do this first
On the RM node (or a free GPU), e.g.:
```bash
HOST=0.0.0.0 PORT=30000 bash scripts/ifbench/launch_reward_model.sh
```
The trainer's `RM_JUDGE_URL` defaults to `http://131.179.168.120:30000/classify`
— override it to match where you launched the server.

### B) Start training (RM variant)
```bash
docker run -d --name ifbench_rm_scai --gpus all --network host --ipc host \
  -v /xuanwu-tank:/xuanwu-tank \
  -w /xuanwu-tank/north/xw27/multi/slime_scai \
  slime-ifbench:latest \
  -lc "bash scripts/ifbench/run-qwen3-8B_ifbench_rm.sh > /xuanwu-tank/north/xw27/multi/slime_scai/scripts/ifbench/ifbench_rm_run.log 2>&1"
```
`-w .../slime_scai` ensures `python3 train.py` imports the **local** (merged)
`slime/` package, not any image-installed copy.

Baseline (no RM): same command with `run-qwen3-8B_ifbench.sh` and a different
container name.

---

## 4. Path rewrites applied (vs the source `../slime` scripts)

| Original (`../slime`) | Rewritten (here) | Why |
|---|---|---|
| `…/multi/slime/examples/ifbench/…` | `…/multi/slime_scai/examples/ifbench/…` | use this repo's data copies |
| `model/Qwen3-8B_ifbench_rm_slime` | `model/Qwen3-8B_ifbench_rm_slime_scai` | **distinct save dir** — must not collide with the live `../slime` run |
| `model/Qwen3-8B_ifbench_slime` (baseline) | `…_slime_scai` | same |
| `xw27/ray_temp` | `xw27/ray_temp_scai` | scripts `rm -rf $RAY_TMPDIR` at startup — a shared temp would wipe the live run's Ray state |

**Kept shared (intentionally):** `HF_CKPT`, `REF_CKPT` (whx checkpoints),
`RM_JUDGE_URL`. Review these if you move to a different node.

---

## 5. Gotcha: IFBench eval-repo resolution

`slime/rollout/rm_hub/ifbench.py` computes the IFBench repo path as
`Path(__file__).resolve().parents[3].parent / "IFBench"`, i.e.
`<repo_root_parent>/IFBench`. Running from `slime_scai`, that is
`/xuanwu-tank/north/xw27/multi/IFBench` (which exists ✓). The
`slime_scai/IFBench/` copy is therefore **redundant** and not used by the eval
path. If you relocate this tree, ensure an `IFBench` repo exists as a *sibling*
of the repo root (or it will be auto-cloned from GitHub there).

---

## 6. Verification checklist

- [x] `python3 -m py_compile slime/rollout/rm_hub/__init__.py slime/utils/arguments.py …` → OK
- [x] `--rm-judge-url` / `--rm-judge-threshold` present in `arguments.py`
- [x] `_query_rm_judge` / `_apply_rm_judge_reward` / `ifeval`+`multi` branch present in `rm_hub/__init__.py`
- [x] `ifeval.py` + `ifeval_g/` present
- [x] data files present (`IF_multi_constraints_upto5_ifbench_en.jsonl`, `IFBench_eval.jsonl`)
- [x] no stale `multi/slime/examples` refs in scripts
- [ ] (at runtime) RM server reachable at `RM_JUDGE_URL`
- [ ] (at runtime) image `slime-ifbench:latest` available on the node
