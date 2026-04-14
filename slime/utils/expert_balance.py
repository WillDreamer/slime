"""Utilities for measuring MoE expert load balance.

Provides metrics to quantify how evenly tokens are distributed across experts
during both inference (SGLang rollout) and training (Megatron forward pass).

Metrics computed per MoE layer:
- **expert_counts**: raw token count per expert
- **load_balance_ratio**: max_count / mean_count (1.0 = perfect balance)
- **coefficient_of_variation**: std / mean (0.0 = perfect balance)
- **entropy**: Shannon entropy of the distribution (higher = more balanced)
- **max_expert / min_expert**: most/least loaded expert indices
- **dead_expert_count**: number of experts receiving zero tokens

Raw data saving:
- Set ``MOE_BALANCE_DATA_DIR`` env var to enable saving per-step raw expert
  counts as ``.npz`` files for offline visualization.
"""

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_expert_balance_per_layer(
    routed_experts: np.ndarray,
    num_experts: int,
    layer_idx: int | None = None,
) -> dict[str, Any]:
    """Compute balance metrics for a single MoE layer.

    Args:
        routed_experts: Expert indices of shape ``[num_tokens, top_k]`` (int).
        num_experts: Total number of experts in this layer.
        layer_idx: Optional layer index, used only for logging.

    Returns:
        Dictionary of scalar metrics.
    """
    flat = routed_experts.reshape(-1)
    counts = np.bincount(flat, minlength=num_experts).astype(np.float64)

    total_tokens = counts.sum()
    if total_tokens == 0:
        return {
            "load_balance_ratio": 0.0,
            "cv": 0.0,
            "entropy": 0.0,
            "dead_expert_count": num_experts,
            "max_expert": -1,
            "min_expert": -1,
        }

    mean = counts.mean()
    std = counts.std()

    # Load balance ratio: how much more load the busiest expert has vs average
    load_balance_ratio = float(counts.max() / mean) if mean > 0 else 0.0

    # Coefficient of variation (lower = more balanced)
    cv = float(std / mean) if mean > 0 else 0.0

    # Shannon entropy (higher = more balanced, max = log2(num_experts))
    probs = counts / total_tokens
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = float(np.log2(num_experts)) if num_experts > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    dead_expert_count = int(np.sum(counts == 0))

    return {
        "load_balance_ratio": round(load_balance_ratio, 4),
        "cv": round(cv, 4),
        "entropy": round(entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "dead_expert_count": dead_expert_count,
        "max_expert": int(np.argmax(counts)),
        "max_expert_count": int(counts.max()),
        "min_expert": int(np.argmin(counts)),
        "min_expert_count": int(counts.min()),
    }


def compute_expert_balance_from_samples(
    samples: list,
    num_experts: int,
) -> dict[str, float]:
    """Aggregate expert balance metrics across all samples in a rollout batch.

    Collects ``rollout_routed_experts`` from each sample, concatenates along the
    token dimension, and computes per-layer and summary balance metrics.

    Args:
        samples: List of Sample objects. Each may have a
            ``rollout_routed_experts`` attribute of shape
            ``[seq_len, num_layers, top_k]``.
        num_experts: Total number of experts.

    Returns:
        Flat dict of metrics keyed like ``"moe_balance/layer_{i}/cv"``,
        plus summary keys like ``"moe_balance/mean_cv"``.
    """
    # Collect all routed_experts arrays
    expert_arrays = []
    for s in samples:
        re = getattr(s, "rollout_routed_experts", None)
        if re is not None:
            expert_arrays.append(re)

    if not expert_arrays:
        return {}

    # Concatenate along token dimension: [total_tokens, num_layers, top_k]
    all_experts = np.concatenate(expert_arrays, axis=0)
    num_tokens, num_layers, top_k = all_experts.shape

    log_dict: dict[str, float] = {}

    # Per-layer metrics
    layer_cvs = []
    layer_lbrs = []
    layer_entropies = []
    layer_dead_counts = []

    for layer_idx in range(num_layers):
        layer_data = all_experts[:, layer_idx, :]  # [num_tokens, top_k]
        metrics = compute_expert_balance_per_layer(layer_data, num_experts, layer_idx)

        for key, val in metrics.items():
            log_dict[f"layer_{layer_idx}/{key}"] = val

        layer_cvs.append(metrics["cv"])
        layer_lbrs.append(metrics["load_balance_ratio"])
        layer_entropies.append(metrics["normalized_entropy"])
        layer_dead_counts.append(metrics["dead_expert_count"])

    # Summary statistics across layers
    log_dict["mean_cv"] = round(float(np.mean(layer_cvs)), 4)
    log_dict["max_cv"] = round(float(np.max(layer_cvs)), 4)
    log_dict["mean_load_balance_ratio"] = round(float(np.mean(layer_lbrs)), 4)
    log_dict["max_load_balance_ratio"] = round(float(np.max(layer_lbrs)), 4)
    log_dict["mean_normalized_entropy"] = round(float(np.mean(layer_entropies)), 4)
    log_dict["min_normalized_entropy"] = round(float(np.min(layer_entropies)), 4)
    log_dict["total_dead_experts"] = int(np.sum(layer_dead_counts))
    log_dict["num_tokens"] = num_tokens
    log_dict["num_layers"] = num_layers

    logger.info(
        f"MoE expert balance: {num_tokens} tokens, {num_layers} layers, "
        f"mean_cv={log_dict['mean_cv']}, mean_lbr={log_dict['mean_load_balance_ratio']}, "
        f"dead_experts={log_dict['total_dead_experts']}"
    )

    return log_dict


def compute_expert_balance_from_tensor(
    top_indices: "torch.Tensor",
    num_experts: int,
    layer_idx: int | None = None,
) -> dict[str, float]:
    """Compute balance metrics from a torch tensor (training side).

    Args:
        top_indices: Tensor of shape ``[num_tokens, top_k]`` on any device.
        num_experts: Total number of experts.
        layer_idx: Optional layer index for keying results.

    Returns:
        Dict of scalar metrics.
    """
    arr = top_indices.detach().cpu().numpy()
    return compute_expert_balance_per_layer(arr, num_experts, layer_idx)


# ======================== Raw expert counts saving ========================


def _get_data_dir() -> str | None:
    """Return the directory for saving raw expert count data, or None if disabled."""
    return os.environ.get("MOE_BALANCE_DATA_DIR")


def _compute_expert_counts_matrix(
    all_experts: np.ndarray,
    num_experts: int,
) -> np.ndarray:
    """Compute per-layer expert token counts.

    Args:
        all_experts: Expert indices of shape ``[num_tokens, num_layers, top_k]``.
        num_experts: Total number of experts.

    Returns:
        Array of shape ``[num_layers, num_experts]`` — token count per expert per layer.
    """
    num_tokens, num_layers, top_k = all_experts.shape
    counts = np.zeros((num_layers, num_experts), dtype=np.int64)
    for layer_idx in range(num_layers):
        flat = all_experts[:, layer_idx, :].reshape(-1)
        counts[layer_idx] = np.bincount(flat, minlength=num_experts)
    return counts


def save_rollout_expert_counts(
    samples: list,
    num_experts: int,
    step: int,
) -> None:
    """Save per-layer per-expert token counts from rollout samples to disk.

    Saves a ``.npz`` file containing:
    - ``expert_counts``: array of shape ``[num_layers, num_experts]``
    - ``step``: the rollout step number
    - ``num_tokens``: total number of tokens aggregated

    The file is saved to ``$MOE_BALANCE_DATA_DIR/rollout/step_{step}.npz``.
    Does nothing if ``MOE_BALANCE_DATA_DIR`` is not set.
    """
    data_dir = _get_data_dir()
    if data_dir is None:
        return

    expert_arrays = []
    for s in samples:
        re = getattr(s, "rollout_routed_experts", None)
        if re is not None:
            expert_arrays.append(re)

    if not expert_arrays:
        return

    all_experts = np.concatenate(expert_arrays, axis=0)
    counts = _compute_expert_counts_matrix(all_experts, num_experts)

    save_dir = os.path.join(data_dir, "rollout")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"step_{step}.npz")
    np.savez_compressed(
        save_path,
        expert_counts=counts,
        step=step,
        num_tokens=all_experts.shape[0],
    )
    logger.info(f"Saved rollout expert counts to {save_path}, shape={counts.shape}")


def save_training_expert_counts(
    layer_data: dict[int, np.ndarray],
    num_experts: int,
    step: int,
) -> None:
    """Save per-layer per-expert token counts from training to disk.

    Args:
        layer_data: Mapping from layer index to expert indices array
            of shape ``[num_tokens, top_k]``.
        num_experts: Total number of experts.
        step: The training step number.

    Saves to ``$MOE_BALANCE_DATA_DIR/train/step_{step}.npz``.
    Does nothing if ``MOE_BALANCE_DATA_DIR`` is not set.
    """
    data_dir = _get_data_dir()
    if data_dir is None:
        return

    if not layer_data:
        return

    num_layers = max(layer_data.keys()) + 1
    counts = np.zeros((num_layers, num_experts), dtype=np.int64)
    total_tokens = 0
    for layer_idx, arr in layer_data.items():
        flat = arr.reshape(-1)
        counts[layer_idx] = np.bincount(flat, minlength=num_experts)
        total_tokens += arr.shape[0]

    save_dir = os.path.join(data_dir, "train")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"step_{step}.npz")
    np.savez_compressed(
        save_path,
        expert_counts=counts,
        step=step,
        num_tokens=total_tokens,
    )
    logger.info(f"Saved training expert counts to {save_path}, shape={counts.shape}")
