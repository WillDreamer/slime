# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is slime

slime is a high-performance LLM post-training framework for RL scaling, built on Megatron-LM (training) + SGLang (inference) + Ray (orchestration). It powers the RL training behind GLM-4.5 through GLM-5 models and supports Qwen3, DeepSeek V3/R1, Llama 3, and others.

## Build & Install

```bash
pip install -e .                    # Editable install (pulls requirements.txt)
bash build_conda.sh                 # Conda environment setup
python setup.py bdist_wheel         # Build wheel
```

Requirements: Python 3.10+, CUDA 12.x (or ROCm), Ray, Megatron-LM, SGLang.

## Code Quality

Pre-commit hooks enforce formatting. Install and run:

```bash
pre-commit install
pre-commit run --all-files
```

Tool chain (in order): **ruff** (E/F/B/UP rules, line-length 320) → **autoflake** (remove unused imports) → **isort** (black-compatible, line-length 119) → **black** (line-length 119).

Known first-party packages for isort: `slime`, `slime_plugins`.

## Testing

```bash
pytest                                              # All tests
pytest -m unit                                      # Unit tests only
pytest -m "not integration"                         # Skip integration
pytest tests/test_qwen2.5_0.5B_gsm8k_short.py      # Single test file
```

Markers: `unit`, `integration`, `system`, `acceptance`, `docs`, `skipduringci`, `pleasefixme`. CI is triggered by PR labels (`run-ci-short`, `run-ci-sglang-config`) and runs E2E tests on self-hosted GPU runners.

## Architecture

### Three-module design

1. **Training (Megatron)** — `slime/backends/megatron_utils/`: Distributed training via `MegatronTrainRayActor`. Handles PPO loss (`loss.py`), checkpointing, weight sync to rollout servers.
2. **Rollout (SGLang)** — `slime/rollout/` + `slime/ray/rollout.py`: Data generation via SGLang inference engines. `RolloutManager` coordinates engines, manages dataset buffering, and exposes metrics. `sglang_rollout.py` implements the main generation pipeline: tokenize → batch → SGLang inference → reward computation → dynamic filtering.
3. **Data Buffer** — Bridges prompt initialization, custom data sources, and rollout generation.

### Ray orchestration — `slime/ray/`

- `placement_group.py`: GPU allocation across training and rollout actors (PACK strategy)
- `actor_group.py`: `RayTrainGroup` manages distributed training actors
- `rollout.py`: `RolloutManager` + `SGLangEngine` coordinate inference with dynamic port allocation
- `train_actor.py`: Base training actor interface

### Training loops

- `train.py`: Synchronous — rollout → train → checkpoint → weight update → eval
- `train_async.py`: Asynchronous — overlaps rollout generation with training via pre-fetching

### Key subsystems

- **Arguments** (`slime/utils/arguments.py`, ~1760 lines): Three namespaces — Megatron args (direct), SGLang args (`--sglang-` prefix), slime-specific args (cluster, training, rollout, model, eval, system config). Supports per-parameter training/freezing via regex (`--only-train-params-name-list`, `--freeze-params-name-list`).
- **PPO** (`slime/utils/ppo_utils.py`): GAE, advantage normalization, entropy regularization, KL penalty.
- **Reward models** (`slime/rollout/rm_hub/`): Pluggable reward functions loaded via `--custom-rm-path`.
- **Dynamic filtering** (`slime/rollout/filter_hub/`): Sample-level masking/filtering hooks.
- **MoE expert balancing** (`slime/utils/expert_balance.py`, `routing_replay.py`): Expert utilization tracking and routing replay for MoE models.
- **Plugins** (`slime_plugins/`): Model implementations (GLM4/5, Qwen3.5), Megatron bridges, custom attention ops, rollout buffers.

### Extension points

- Custom rollout function: `--rollout-function-path my_module.fn`
- Custom reward model: `--custom-rm-path my_module.fn`
- Custom model provider: `--custom-model-provider-path my_module.fn`

## Training scripts

Model-specific scripts live in `scripts/` (e.g., `run-qwen3-*.sh`, `run-glm*.sh`, `run-deepseek-r1.sh`). These set model-specific arguments and invoke `train.py` or `train_async.py`.

## Current work: MoE expert balance tracking (MOE branch)

Goal: measure how evenly tokens are distributed across experts during RL training of MoE models, and study the routing mismatch between inference (SGLang) and training (Megatron) sides.

### Key files added/modified

- `slime/utils/expert_balance.py` — Core metrics (load_balance_ratio, CV, normalized entropy, dead experts) computed per MoE layer. Saves raw `.npz` data when `MOE_BALANCE_DATA_DIR` is set.
- `slime/utils/routing_replay.py` — `TrainingExpertBalanceTracker` collects routing decisions on the training side.
- `slime/ray/rollout.py` — Inference-side integration via `compute_expert_balance_from_samples`.
- `slime/utils/train_metric_utils.py` — Writes training-side balance metrics to wandb (`train_moe_balance/` prefix).
- `slime/backends/megatron_utils/actor.py` — Hooks for forward-only mode and balance tracking.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `MOE_BALANCE_TRACKING=1` | Enable training-side per-layer balance stats (opt-in, has ~1-3% overhead) |
| `MOE_BALANCE_FORWARD_ONLY=1` | Skip backward/optimizer — only run forward pass (for cheaper balance measurement) |
| `MOE_BALANCE_DATA_DIR=/path` | Save raw expert counts as `.npz` files for offline visualization |

### Debug workflow (no full RL loop needed)

1. **Rollout-only** (`--debug-rollout-only`): Launch SGLang inference only, save rollout data with `--save-debug-rollout-data`. Requires `--use-rollout-routing-replay` for expert routing info.
2. **Forward-only** (`--load-debug-rollout-data` + `MOE_BALANCE_FORWARD_ONLY=1`): Load saved rollout data into Megatron, run forward pass only. Uses far less GPU memory (no gradients/optimizer).
3. **Offline analysis**: Load `.npz` files from `$MOE_BALANCE_DATA_DIR/{rollout,train}/step_*.npz` for heatmaps and CV comparisons.

### Experiment setup

Target model: Qwen3-30B-A3B (128 experts). Running on Anvil cluster (Slurm + Apptainer). See `debug_moe_balance.sbatch` for the working job script and `MOE_BALANCE_TRACKING.md` for full documentation.
