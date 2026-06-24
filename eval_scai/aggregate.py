"""Aggregate benchmark results across run1/run2/run3 -> mean ± std per model.

Usage: .venv_inspect/bin/python aggregate.py [results_dir]
Reads results/<model>/run*/ (inspect .eval + ifbench.json + search_full.json)
and prints a per-benchmark mean ± std table. MMLU may have fewer runs than the
others (1 for the rambling RL checkpoints) — reported with its own n.
"""
import glob
import json
import os
import statistics
import sys

from inspect_ai.log import read_eval_log

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
INSPECT_METRIC = {  # task substring -> (metric key, label)
    "gpqa-diamond": ("accuracy", "gpqa"),
    "aime2025": ("accuracy", "aime"),
    "ifeval": ("final_acc", "ifeval"),
    "mmlu-5-shot": ("accuracy", "mmlu"),
    "tau2-retail": ("accuracy", "tau2_retail"),
    "tau2-airline": ("accuracy", "tau2_airline"),
    "tau2-telecom": ("accuracy", "tau2_telecom"),
}


def collect(model_dir):
    vals = {}  # bench -> list of run values
    for run_dir in sorted(glob.glob(os.path.join(model_dir, "run*"))):
        # inspect tasks
        for f in glob.glob(os.path.join(run_dir, "inspect_logs", "*.eval")):
            try:
                log = read_eval_log(f, header_only=True)
            except Exception:
                continue
            if log.status != "success" or not log.results:
                continue
            for sub, (mkey, label) in INSPECT_METRIC.items():
                if sub in os.path.basename(f):
                    for s in log.results.scores:
                        if mkey in s.metrics:
                            vals.setdefault(label, []).append(s.metrics[mkey].value)
        # tau1 (training-aligned tau-bench v1): runN/tau1_<env>.json
        for p in glob.glob(os.path.join(run_dir, "tau1_*.json")):
            env = os.path.basename(p)[len("tau1_"):-len(".json")]
            try:
                vals.setdefault(f"tau1_{env}", []).append(json.load(open(p))["summary"]["accuracy"])
            except Exception:
                pass
        # ifbench
        p = os.path.join(run_dir, "ifbench.json")
        if os.path.exists(p):
            vals.setdefault("ifbench", []).append(json.load(open(p))["summary"]["accuracy"])
        # search (full)
        p = os.path.join(run_dir, "search_full.json")
        if os.path.exists(p):
            vals.setdefault("search_em", []).append(json.load(open(p))["summary"]["em_overall"])
        # browsecomp judged
        for p in glob.glob(os.path.join(run_dir, "browsecomp_plus_evals", "**", "evaluation_summary.json"), recursive=True):
            try:
                vals.setdefault("browsecomp", []).append(json.load(open(p)).get("Accuracy (%)", 0) / 100.0)
            except Exception:
                pass
    return vals


def fmt(xs):
    if not xs:
        return "  -  "
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{100*m:5.1f}±{100*s:4.1f} (n={len(xs)})"


ORDER = ["gpqa", "aime", "ifeval", "ifbench", "search_em", "mmlu", "browsecomp",
         "tau1_retail", "tau1_airline",
         "tau2_retail", "tau2_airline", "tau2_telecom"]
models = sorted(glob.glob(os.path.join(ROOT, "*")))
rows = {os.path.basename(m): collect(m) for m in models if os.path.isdir(m)}
print(f"{'benchmark':14s} " + " ".join(f"{os.path.basename(m)[:22]:>22s}" for m in models if os.path.isdir(m)))
for b in ORDER:
    print(f"{b:14s} " + " ".join(f"{fmt(rows[os.path.basename(m)].get(b, [])):>22s}" for m in models if os.path.isdir(m)))
