"""Analyze MoE expert balance from static model measurement (no RL training).

Reads .npz files produced by forward-only balance debug runs and reports
expert load distribution metrics. The model weights are NOT updated during
these runs — this measures the baseline routing behavior of a pretrained
MoE model.

Usage:
    python tools/analyze_moe_balance.py /data1/hhzhang/moe_balance/output
    python tools/analyze_moe_balance.py /data1/hhzhang/moe_balance/output --num-layers 48 --num-experts 128
"""

import argparse
import os

import numpy as np


def load_and_aggregate(data_dir: str, side: str, num_layers: int) -> tuple[np.ndarray, int, int]:
    """Load all step_*.npz files for a side and aggregate by layer.

    Returns:
        (counts, total_tokens, n_steps) where counts has shape [num_layers, num_experts].
    """
    side_dir = os.path.join(data_dir, side)
    if not os.path.isdir(side_dir):
        return None, 0, 0

    all_counts = None
    total_tokens = 0
    n_steps = 0

    for step in range(10000):
        path = os.path.join(side_dir, f"step_{step}.npz")
        if not os.path.exists(path):
            break
        d = np.load(path)
        counts = d["expert_counts"]

        if all_counts is None:
            num_experts = counts.shape[1]
            all_counts = np.zeros((num_layers, num_experts), dtype=np.int64)

        # Aggregate microbatches if needed (train side records per-microbatch per-layer)
        if counts.shape[0] > num_layers:
            n_mb = counts.shape[0] // num_layers
            counts = counts.reshape(n_mb, num_layers, num_experts).sum(axis=0)
        elif counts.shape[0] < num_layers:
            padded = np.zeros((num_layers, num_experts), dtype=counts.dtype)
            padded[: counts.shape[0]] = counts
            counts = padded

        all_counts += counts
        total_tokens += int(d["num_tokens"])
        n_steps += 1

    return all_counts, total_tokens, n_steps


def compute_metrics(counts: np.ndarray) -> dict:
    """Compute per-layer and summary balance metrics."""
    num_layers, num_experts = counts.shape

    mean = counts.mean(axis=1)
    std = counts.std(axis=1)
    cv = std / (mean + 1e-8)

    probs = counts / (counts.sum(axis=1, keepdims=True) + 1e-8)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    max_entropy = np.log(num_experts)
    norm_entropy = entropy / max_entropy

    dead = (counts == 0).sum(axis=1)
    lbr = counts.max(axis=1) / (mean + 1e-8)

    return {
        "cv": cv,
        "norm_entropy": norm_entropy,
        "dead": dead,
        "lbr": lbr,
        "mean_tokens_per_expert": mean,
    }


def print_summary(name: str, counts: np.ndarray, total_tokens: int, n_steps: int):
    """Print a summary of balance metrics."""
    m = compute_metrics(counts)
    num_layers, num_experts = counts.shape

    print(f"=== {name} ({n_steps} steps, {total_tokens:,} tokens) ===")
    print(f"  Tokens per expert per layer (avg): {m['mean_tokens_per_expert'].mean():.0f}")
    print()
    print(f"  {'Metric':<25s} {'Mean':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for key, label in [
        ("cv", "CV"),
        ("norm_entropy", "Normalized Entropy"),
        ("lbr", "Load Balance Ratio"),
        ("dead", "Dead Experts"),
    ]:
        vals = m[key]
        print(f"  {label:<25s} {vals.mean():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f}")

    print()
    worst = np.argsort(m["cv"])[-5:][::-1]
    print(f"  Top-5 imbalanced layers (by CV):")
    for i, layer in enumerate(worst):
        print(
            f"    Layer {layer:>2d}: CV={m['cv'][layer]:.4f}, "
            f"entropy={m['norm_entropy'][layer]:.4f}, "
            f"dead={m['dead'][layer]}, "
            f"LBR={m['lbr'][layer]:.2f}"
        )
    print()


def print_mismatch(r_counts: np.ndarray, t_counts: np.ndarray):
    """Print routing mismatch between rollout and train."""
    r_norm = r_counts / (r_counts.sum(axis=1, keepdims=True) + 1e-8)
    t_norm = t_counts / (t_counts.sum(axis=1, keepdims=True) + 1e-8)

    l1 = np.abs(r_norm - t_norm).sum(axis=1)
    cosine = (r_norm * t_norm).sum(axis=1) / (
        np.linalg.norm(r_norm, axis=1) * np.linalg.norm(t_norm, axis=1) + 1e-8
    )

    print(f"=== Routing Mismatch (rollout vs train) ===")
    print(f"  {'Metric':<25s} {'Mean':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'L1 Distance':<25s} {l1.mean():>8.4f} {l1.min():>8.4f} {l1.max():>8.4f}")
    print(f"  {'Cosine Similarity':<25s} {cosine.mean():>8.4f} {cosine.min():>8.4f} {cosine.max():>8.4f}")

    print()
    worst = np.argsort(l1)[-5:][::-1]
    print(f"  Top-5 most different layers (by L1):")
    for layer in worst:
        print(f"    Layer {layer:>2d}: L1={l1[layer]:.4f}, cosine={cosine[layer]:.4f}")
    print()


def load_per_step(data_dir: str, side: str, num_layers: int) -> list[dict]:
    """Load per-step metrics for stability analysis across runs."""
    side_dir = os.path.join(data_dir, side)
    if not os.path.isdir(side_dir):
        return []

    step_metrics = []
    for step in range(10000):
        path = os.path.join(side_dir, f"step_{step}.npz")
        if not os.path.exists(path):
            break
        d = np.load(path)
        counts = d["expert_counts"]
        num_experts = counts.shape[1]

        if counts.shape[0] > num_layers:
            n_mb = counts.shape[0] // num_layers
            counts = counts.reshape(n_mb, num_layers, num_experts).sum(axis=0)

        m = compute_metrics(counts)
        step_metrics.append({
            "step": step,
            "tokens": int(d["num_tokens"]),
            "mean_cv": float(m["cv"].mean()),
            "mean_entropy": float(m["norm_entropy"].mean()),
            "mean_lbr": float(m["lbr"].mean()),
            "total_dead": int(m["dead"].sum()),
        })

    return step_metrics


def print_stability(name: str, step_metrics: list[dict]):
    """Print per-step stability statistics for cross-run comparison."""
    if not step_metrics:
        return

    cvs = [s["mean_cv"] for s in step_metrics]
    entropies = [s["mean_entropy"] for s in step_metrics]
    lbrs = [s["mean_lbr"] for s in step_metrics]
    tokens = [s["tokens"] for s in step_metrics]

    print(f"=== {name} — Per-Step Stability ({len(step_metrics)} steps) ===")
    print(f"  {'Metric':<25s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for vals, label in [
        (cvs, "CV (layer-avg)"),
        (entropies, "Norm Entropy (layer-avg)"),
        (lbrs, "LBR (layer-avg)"),
        (tokens, "Tokens per step"),
    ]:
        a = np.array(vals, dtype=np.float64)
        print(f"  {label:<25s} {a.mean():>8.4f} {a.std():>8.4f} {a.min():>8.4f} {a.max():>8.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze MoE expert balance data")
    parser.add_argument("data_dir", help="Path to balance output directory (containing rollout/ and train/)")
    parser.add_argument("--num-layers", type=int, default=48, help="Number of MoE layers")
    parser.add_argument("--num-experts", type=int, default=128, help="Number of experts per layer")
    args = parser.parse_args()

    r_counts, r_tokens, r_steps = load_and_aggregate(args.data_dir, "rollout", args.num_layers)
    t_counts, t_tokens, t_steps = load_and_aggregate(args.data_dir, "train", args.num_layers)

    if r_counts is not None and r_steps > 0:
        print_summary("Rollout (SGLang)", r_counts, r_tokens, r_steps)

    if t_counts is not None and t_steps > 0:
        print_summary("Train (Megatron)", t_counts, t_tokens, t_steps)

    if r_counts is not None and t_counts is not None and r_steps > 0 and t_steps > 0:
        print_mismatch(r_counts, t_counts)

    # Per-step stability
    for side, name in [("rollout", "Rollout (SGLang)"), ("train", "Train (Megatron)")]:
        step_metrics = load_per_step(args.data_dir, side, args.num_layers)
        print_stability(name, step_metrics)


if __name__ == "__main__":
    main()
