#!/usr/bin/env python3
"""Report-convention aggregator for the tau3 (tau2-bench v1.0.0) eval.

Difference from ``aggregate_tau3.py``: pass^1 here uses the **full expected
denominator** (``num_tasks * num_trials``), i.e. every intended simulation.
Simulations that never produced a reward (infrastructure_error / context
overflow) are counted as FAILURES, not dropped. This is the conservative,
apples-to-apples number used in TAU3_EVAL_REPORT.md.

    sudo python3 aggregate_tau3_report.py [SIM_DIR]

SIM_DIR defaults to the tau3_rp_results dir next to this script. results.json
files are root-owned (mode 0600), so run under sudo.
"""
import json, glob, os, re, sys, collections, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tau3_rp_results")
ORDER = ["qwen-8b-base", "Qwen3-8B-Base-Math",
         "Qwen3-8B-Base-Math-SeaSFT-Search",
         "Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau"]
DOMAINS = ["retail", "airline", "telecom", "mock", "banking_knowledge"]

rows = collections.defaultdict(dict)   # (model, dom) -> run -> stats
for d in sorted(glob.glob(os.path.join(SIM, "tau3rp_*"))):
    m = re.match(r"tau3rp_(.+)_(retail|airline|telecom|mock|banking_knowledge)_run(\d+)$",
                 os.path.basename(d))
    if not m:
        continue
    model, dom, run = m.group(1), m.group(2), int(m.group(3))
    rj = os.path.join(d, "results.json")
    if not os.path.exists(rj):
        continue
    try:
        data = json.load(open(rj))
    except Exception:
        continue
    sims = data.get("simulations", [])
    ntrials = (data.get("info") or {}).get("num_trials") or 4
    ntasks = len(data.get("tasks") or [])
    expected = ntasks * ntrials
    scored = succ = 0
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward")
        if r is None:
            continue
        scored += 1
        if float(r) >= 0.999:
            succ += 1
    rows[(model, dom)][run] = dict(succ=succ, scored=scored, expected=expected)

print("PER-RUN  pass^1 = successes / EXPECTED (missing sims = failures)\n")
for model in ORDER:
    if not any((model, dm) in rows for dm in DOMAINS):
        continue
    print(model)
    for dom in DOMAINS:
        if (model, dom) not in rows:
            continue
        for r in sorted(rows[(model, dom)]):
            x = rows[(model, dom)][r]
            p1 = 100 * x["succ"] / x["expected"] if x["expected"] else 0
            cov = 100 * x["scored"] / x["expected"] if x["expected"] else 0
            print(f"   {dom:18s} run{r}: pass^1={p1:5.1f}%  "
                  f"({x['succ']:3d}/{x['expected']:<4d})  cov {cov:3.0f}%")
    print()

print("=" * 60)
print("AGGREGATE  mean +/- std across ALL runs (full-expected denom)\n")
for model in ORDER:
    cells = []
    for dom in DOMAINS:
        if (model, dom) not in rows:
            continue
        vals = [100 * rows[(model, dom)][r]["succ"] / rows[(model, dom)][r]["expected"]
                for r in sorted(rows[(model, dom)]) if rows[(model, dom)][r]["expected"]]
        if not vals:
            continue
        mn = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        cells.append(f"{dom}={mn:.1f}+/-{sd:.1f}(n{len(vals)})")
    if cells:
        print(f"{model}\n   " + "   ".join(cells) + "\n")
