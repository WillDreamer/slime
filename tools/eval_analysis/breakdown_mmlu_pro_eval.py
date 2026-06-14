#!/usr/bin/env python3
"""
mmlu_pro_slime category breakdown across checkpoints, with fixed/broken
matrices and per-question pattern table.

MMLU-Pro uses 14 native categories already in doc.category — no mapping
needed (unlike MMLU's 57→4).

Usage:
  python breakdown_mmlu_pro_eval.py base=…jsonl base_math=…jsonl final_search=…jsonl
  python breakdown_mmlu_pro_eval.py … --csv out.csv
  python breakdown_mmlu_pro_eval.py … --flapping-table
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# Canonical category order (size-descending in the dataset, useful for tables).
CATEGORY_ORDER = [
    "math", "physics", "chemistry", "law", "engineering", "other",
    "economics", "health", "psychology", "business", "biology",
    "philosophy", "computer science", "history",
]


def breakdown(path, name):
    by_category = defaultdict(lambda: [0, 0])
    by_doc = {}
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            cat = s.get("doc", {}).get("category", "?")
            correct = (s.get("exact_match") == 1.0)
            by_category[cat][1] += 1
            if correct:
                by_category[cat][0] += 1
            by_doc[s["doc_id"]] = (correct, cat, s.get("target", ""))
    return dict(name=name, path=str(path),
                by_category=dict(by_category),
                by_doc=by_doc,
                n=sum(v[1] for v in by_category.values()),
                correct=sum(v[0] for v in by_category.values()))


def _acc(c, t):
    return f"{c/t:.4f}" if t else "  N/A"


def print_per_ckpt(r):
    print(f"\n=== {r['name']} ===")
    print(f"  file: {r['path']}")
    print(f"  overall: {r['correct']}/{r['n']} = {r['correct']/r['n']:.4f}")
    print(f"\n  By category (size-descending):")
    cats = sorted(r["by_category"].items(), key=lambda x: -x[1][1])
    for cat, (c, t) in cats:
        print(f"    {cat:20s} {c:4d}/{t:4d}  =  {_acc(c, t)}")


def print_compare(results):
    if len(results) < 2:
        return
    print("\n" + "=" * 90)
    print("CATEGORY ACROSS CKPTS")
    print("=" * 90)
    header = f"{'category':18s} " + " ".join(f"{r['name']:>17s}" for r in results) + f"  {'Δ first→last':>14s}"
    print(header)
    print("-" * len(header))
    rows = []
    for cat in CATEGORY_ORDER:
        sizes = [r["by_category"].get(cat, (0, 0)) for r in results]
        if not any(t for c, t in sizes):
            continue
        accs = [c / t if t else None for c, t in sizes]
        d = (accs[-1] - accs[0]) if (accs[0] is not None and accs[-1] is not None) else None
        rows.append((cat, sizes, accs, d))
    rows.sort(key=lambda x: x[1][-1][1], reverse=True)
    for cat, sizes, accs, d in rows:
        cells = []
        for (c, t) in sizes:
            cells.append(f"{_acc(c, t):>9s} ({c:>4d}/{t:>4d})")
        delta = f"{d:+.4f}" if d is not None else ""
        print(f"  {cat:16s} " + " ".join(cells) + f"  {delta:>14s}")


def _classify_pattern(pattern):
    if len(set(pattern)) == 1:
        return "stable"
    n = len(pattern)
    if n == 2:
        a, b = pattern
        return "fixed" if (not a and b) else "broken"
    if n == 3:
        a, b, c = pattern
        if not a and b and c: return "fixed_at_bm"
        if not a and not b and c: return "fixed_at_fs"
        if a and not b and not c: return "broken_at_bm"
        if a and b and not c: return "broken_at_fs"
        if a and not b and c: return "regressed_then_recovered"
        if not a and b and not c: return "fixed_then_broken"
    if n == 4:
        # ckpts ordered (base, base_math, final_search, tau2)
        if pattern == (False, True, True, True):    return "fixed_at_bm"
        if pattern == (False, False, True, True):   return "fixed_at_fs"
        if pattern == (False, False, False, True):  return "fixed_at_tau2"
        if pattern == (True, False, False, False):  return "broken_at_bm"
        if pattern == (True, True, False, False):   return "broken_at_fs"
        if pattern == (True, True, True, False):    return "broken_at_tau2"
        return "volatile_" + "".join("1" if x else "0" for x in pattern)
    return "other"


def print_transitions(results):
    if len(results) < 2:
        return
    print("\n" + "=" * 90)
    print("TRANSITION ANALYSIS")
    print("=" * 90)

    all_ids = set(results[0]["by_doc"].keys())
    for r in results[1:]:
        all_ids &= set(r["by_doc"].keys())
    all_ids = sorted(all_ids)

    persistent_correct = sum(1 for d in all_ids if all(r["by_doc"][d][0] for r in results))
    persistent_wrong = sum(1 for d in all_ids if all(not r["by_doc"][d][0] for r in results))
    flapping = len(all_ids) - persistent_correct - persistent_wrong
    print(f"  total: {len(all_ids)}    persistent ✓: {persistent_correct}    "
          f"persistent ✗: {persistent_wrong}    flapping: {flapping}")

    pairs = [(results[i], results[i + 1]) for i in range(len(results) - 1)]
    if len(results) > 2:
        pairs.append((results[0], results[-1]))

    for a, b in pairs:
        fixed = [d for d in all_ids if not a["by_doc"][d][0] and b["by_doc"][d][0]]
        broken = [d for d in all_ids if a["by_doc"][d][0] and not b["by_doc"][d][0]]
        print(f"\n  --- {a['name']} → {b['name']}    "
              f"fixed={len(fixed)}  broken={len(broken)}  net={len(fixed)-len(broken):+d} ---")

        fixed_cat = defaultdict(int); broken_cat = defaultdict(int)
        for d in fixed:
            fixed_cat[a["by_doc"][d][1]] += 1
        for d in broken:
            broken_cat[a["by_doc"][d][1]] += 1
        print(f"    {'category':20s} {'fixed':>6s}  {'broken':>6s}  {'net':>6s}")
        for cat in CATEGORY_ORDER:
            f, b_ = fixed_cat.get(cat, 0), broken_cat.get(cat, 0)
            if f + b_ == 0:
                continue
            print(f"    {cat:20s} {f:>6d}  {b_:>6d}  {f-b_:>+6d}")


def print_pattern_summary(results):
    if len(results) not in (3, 4):
        return
    print("\n" + "=" * 90)
    print("PER-QUESTION PATTERN SUMMARY")
    print("=" * 90)
    all_ids = set(results[0]["by_doc"].keys())
    for r in results[1:]:
        all_ids &= set(r["by_doc"].keys())
    all_ids = sorted(all_ids)

    pat_counts = defaultdict(int)
    pat_by_cat = defaultdict(lambda: defaultdict(int))
    for d in all_ids:
        pat = tuple(r["by_doc"][d][0] for r in results)
        name = _classify_pattern(pat)
        pat_counts[name] += 1
        cat = results[0]["by_doc"][d][1]
        pat_by_cat[name][cat] += 1

    pat_order = ["stable",
                 "fixed_at_bm", "fixed_at_fs", "fixed_at_tau2",
                 "broken_at_bm", "broken_at_fs", "broken_at_tau2",
                 "regressed_then_recovered", "fixed_then_broken", "other"]
    for k in sorted(pat_counts):
        if k not in pat_order and k.startswith("volatile_"):
            pat_order.append(k)
    print(f"\n  {'pattern':30s} {'total':>7s}  top categories")
    for name in pat_order:
        if pat_counts.get(name, 0) == 0:
            continue
        top = sorted(pat_by_cat[name].items(), key=lambda x: -x[1])[:4]
        top_str = ", ".join(f"{c}({n})" for c, n in top)
        print(f"  {name:30s} {pat_counts[name]:>7d}  {top_str}")


def write_csv(results, csv_path):
    import csv
    all_ids = sorted(set(results[0]["by_doc"].keys()))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["doc_id", "category", "target"] + [r["name"] for r in results] + ["pattern"]
        w.writerow(header)
        for d in all_ids:
            _, cat, target = results[0]["by_doc"][d]
            row = [d, cat, target]
            pat = []
            for r in results:
                v = r["by_doc"][d][0]
                row.append(1 if v else 0)
                pat.append(v)
            row.append(_classify_pattern(tuple(pat)))
            w.writerow(row)
    print(f"\n[csv] wrote {len(all_ids)} rows to {csv_path}")


def parse_input(arg):
    if "=" in arg and not arg.startswith(("/", ".")):
        n, p = arg.split("=", 1)
    else:
        p = arg; n = Path(p).stem
    return n, Path(p)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    results = []
    for inp in args.inputs:
        name, path = parse_input(inp)
        if not path.exists():
            print(f"[skip] {path}", file=sys.stderr); continue
        r = breakdown(path, name)
        results.append(r)
        print_per_ckpt(r)

    if len(results) > 1:
        print_compare(results)
        print_transitions(results)
        print_pattern_summary(results)
        if args.csv:
            write_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()
