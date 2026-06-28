#!/usr/bin/env python3
"""Aggregate tau3 results: pass^1 (mean binary reward) per model/domain,
mean +/- std across runs, plus coverage (progress). Reads tau3rp_* sim dirs.

  python3 aggregate_tau3.py [SIM_DIR]
SIM_DIR defaults to the tau2-bench simulations dir under /mnt/old-data3.
"""
import json, glob, os, re, sys, collections, statistics, math

SIM = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/old-data3/home/ec2-user/tau2-bench/data/simulations"
ORDER = ["qwen-8b-base", "Qwen3-8B-Base-Math",
         "Qwen3-8B-Base-Math-SeaSFT-Search",
         "Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau"]
DOMAINS = ["retail", "airline", "telecom", "mock", "banking_knowledge"]

rows = collections.defaultdict(dict)   # (model,dom) -> run -> stats
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
    by_task = collections.defaultdict(list)
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward")
        if r is None:
            continue
        by_task[s.get("task_id")].append(float(r))
    scored = sum(len(v) for v in by_task.values())
    succ = sum(1 for v in by_task.values() for r in v if r >= 0.999)
    rows[(model, dom)][run] = dict(
        scored=scored, expected=ntasks * ntrials, ntasks=ntasks,
        touched=len(by_task), p1=(succ / scored if scored else None))

print("PER-RUN  (pass^1 = mean binary reward; cov = scored/expected)\n")
for model in ORDER:
    if not any((model, d) in rows for d in DOMAINS):
        continue
    print(model)
    for dom in DOMAINS:
        if (model, dom) not in rows:
            continue
        for r in sorted(rows[(model, dom)]):
            x = rows[(model, dom)][r]
            cov = 100 * x["scored"] / x["expected"] if x["expected"] else 0
            p1 = f"{100*x['p1']:5.1f}%" if x["p1"] is not None else "  -  "
            print(f"   {dom:18s} run{r}: pass^1={p1}  "
                  f"cov {x['scored']:4d}/{x['expected']:<4d} ({cov:3.0f}%)  "
                  f"tasks {x['touched']}/{x['ntasks']}")
    print()

print("=" * 64)
print("AGGREGATE  (mean +/- std across runs; runs with <20 sims dropped)\n")
for model in ORDER:
    cells = []
    for dom in DOMAINS:
        if (model, dom) not in rows:
            continue
        vals = [rows[(model, dom)][r]["p1"] * 100 for r in sorted(rows[(model, dom)])
                if rows[(model, dom)][r]["scored"] >= 20 and rows[(model, dom)][r]["p1"] is not None]
        if not vals:
            continue
        mn = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        cells.append(f"{dom}={mn:.1f}+/-{sd:.1f}(n{len(vals)})")
    if cells:
        print(f"{model}\n   " + "   ".join(cells) + "\n")
