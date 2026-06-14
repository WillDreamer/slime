#!/usr/bin/env python3
"""
mmlu_slime category × subject breakdown across checkpoints, with fixed/broken
matrices and per-question pattern table. Equivalent to breakdown_math_eval.py.

Layout:
  1. Per-ckpt: by category (4 rows), by subject (57 rows, sorted by n)
  2. Cross-ckpt comparison: category × ckpt accuracy + Δ
  3. Subject-level Δ table (ranked by base→last delta)
  4. Fixed/broken category × ckpt counts (4 categories)
  5. Top-N subject-level fixed/broken movers per pair
  6. Per-question flapping table (skipped by default, --flapping-table to show)
  7. CSV export with doc_id, subject, category, target, per-ckpt correctness, pattern

Usage:
  python breakdown_mmlu_eval.py base=…jsonl base_math=…jsonl final_search=…jsonl
  python breakdown_mmlu_eval.py … --csv out.csv
  python breakdown_mmlu_eval.py … --flapping-table
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mmlu_subjects import SUBJECT_TO_CATEGORY, CATEGORY_ORDER, category_of  # noqa: E402


# ---------- per-file ----------

def breakdown(path, name):
    by_subject = defaultdict(lambda: [0, 0])         # subject -> [c, t]
    by_category = defaultdict(lambda: [0, 0])
    by_doc = {}                                      # doc_id -> (correct, subject, category, target)
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            doc = s.get("doc", {})
            subj = doc.get("subject", "?")
            cat = category_of(subj)
            correct = (s.get("exact_match") == 1.0)
            by_subject[subj][1] += 1
            by_category[cat][1] += 1
            if correct:
                by_subject[subj][0] += 1
                by_category[cat][0] += 1
            by_doc[s["doc_id"]] = (correct, subj, cat, s.get("target", ""))
    return dict(name=name, path=str(path),
                by_subject=dict(by_subject),
                by_category=dict(by_category),
                by_doc=by_doc,
                n=sum(v[1] for v in by_category.values()),
                correct=sum(v[0] for v in by_category.values()))


# ---------- printing ----------

def _acc(c, t):
    return f"{c/t:.4f}" if t else "  N/A"


def print_per_ckpt(r):
    print(f"\n=== {r['name']} ===")
    print(f"  file: {r['path']}")
    print(f"  overall: {r['correct']}/{r['n']} = {r['correct']/r['n']:.4f}")

    print(f"\n  By category:")
    for cat in CATEGORY_ORDER:
        c, t = r["by_category"].get(cat, (0, 0))
        print(f"    {cat:18s} {c:5d}/{t:5d}  =  {_acc(c, t)}")

    print(f"\n  By subject (sorted by n, largest first):")
    for subj, (c, t) in sorted(r["by_subject"].items(), key=lambda x: -x[1][1]):
        print(f"    {subj:42s} {c:4d}/{t:4d}  =  {_acc(c, t)}   [{category_of(subj)}]")


def print_compare_categories(results):
    if len(results) < 2:
        return
    print("\n" + "=" * 90)
    print("CATEGORY ACROSS CKPTS")
    print("=" * 90)
    header = f"{'category':18s} " + " ".join(f"{r['name']:>17s}" for r in results) + f"  {'Δ first→last':>14s}"
    print(header)
    print("-" * len(header))
    for cat in CATEGORY_ORDER:
        cells, accs = [], []
        for r in results:
            c, t = r["by_category"].get(cat, (0, 0))
            accs.append(c / t if t else None)
            cells.append(f"{_acc(c, t):>9s} ({c:>5d}/{t:>5d})")
        delta = ""
        if accs[0] is not None and accs[-1] is not None:
            delta = f"{accs[-1] - accs[0]:+.4f}"
        print(f"  {cat:16s} " + " ".join(cells) + f"  {delta:>14s}")


def print_subject_deltas(results, top_n=15):
    if len(results) < 2:
        return
    a, z = results[0], results[-1]
    rows = []
    for subj, (c, t) in z["by_subject"].items():
        ca, ta = a["by_subject"].get(subj, (0, 0))
        if not ta or not t:
            continue
        rows.append((c / t - ca / ta, subj, (ca, ta), (c, t)))

    print("\n" + "=" * 90)
    print(f"TOP {top_n} SUBJECT-LEVEL Δ ({a['name']} → {z['name']})")
    print("=" * 90)
    print(f"\n  Biggest gains:")
    for d, subj, ab, zb in sorted(rows, key=lambda x: -x[0])[:top_n]:
        if d <= 0: continue
        print(f"    {subj:42s} [{category_of(subj):16s}]  "
              f"{ab[0]:>4d}/{ab[1]:<4d} ({ab[0]/ab[1]:.3f})  →  "
              f"{zb[0]:>4d}/{zb[1]:<4d} ({zb[0]/zb[1]:.3f})   Δ={d:+.4f}")
    print(f"\n  Biggest losses:")
    for d, subj, ab, zb in sorted(rows, key=lambda x: x[0])[:top_n]:
        if d >= 0: continue
        print(f"    {subj:42s} [{category_of(subj):16s}]  "
              f"{ab[0]:>4d}/{ab[1]:<4d} ({ab[0]/ab[1]:.3f})  →  "
              f"{zb[0]:>4d}/{zb[1]:<4d} ({zb[0]/zb[1]:.3f})   Δ={d:+.4f}")


# ---------- transitions / fixed-broken ----------

def _classify_pattern(pattern):
    n = len(pattern)
    if len(set(pattern)) == 1:
        return "stable"
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
    print("TRANSITION ANALYSIS (per-question fixed / broken)")
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
        print(f"\n  --- {a['name']}  →  {b['name']}    "
              f"fixed={len(fixed)}  broken={len(broken)}  net={len(fixed)-len(broken):+d} ---")

        # category × {fixed, broken} matrix
        fixed_cat = defaultdict(int); broken_cat = defaultdict(int)
        for d in fixed:
            fixed_cat[a["by_doc"][d][2]] += 1
        for d in broken:
            broken_cat[a["by_doc"][d][2]] += 1
        print(f"    {'category':18s} {'fixed':>8s}  {'broken':>8s}  {'net':>6s}")
        for cat in CATEGORY_ORDER:
            f, b_ = fixed_cat.get(cat, 0), broken_cat.get(cat, 0)
            print(f"    {cat:18s} {f:>8d}  {b_:>8d}  {f-b_:>+6d}")

        # top subject movers (largest |fixed - broken| per subject)
        by_subj_f = defaultdict(int); by_subj_b = defaultdict(int)
        for d in fixed:  by_subj_f[a["by_doc"][d][1]] += 1
        for d in broken: by_subj_b[a["by_doc"][d][1]] += 1
        movers = []
        for subj in set(by_subj_f) | set(by_subj_b):
            net = by_subj_f.get(subj, 0) - by_subj_b.get(subj, 0)
            movers.append((net, subj, by_subj_f.get(subj, 0), by_subj_b.get(subj, 0)))
        movers.sort()
        print(f"\n    biggest LOSSES (broken > fixed):")
        for net, subj, f, b_ in movers[:5]:
            if net >= 0: break
            print(f"      {subj:42s} fixed={f}  broken={b_}  net={net:+d}")
        print(f"    biggest GAINS (fixed > broken):")
        for net, subj, f, b_ in reversed(movers[-5:]):
            if net <= 0: break
            print(f"      {subj:42s} fixed={f}  broken={b_}  net={net:+d}")


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
        cat = results[0]["by_doc"][d][2]
        pat_by_cat[name][cat] += 1

    pat_order = ["stable",
                 "fixed_at_bm", "fixed_at_fs", "fixed_at_tau2",
                 "broken_at_bm", "broken_at_fs", "broken_at_tau2",
                 "regressed_then_recovered", "fixed_then_broken", "other"]
    # Append any volatile_<bitstring> patterns seen (only relevant for n=4 mode)
    for k in sorted(pat_counts):
        if k not in pat_order and k.startswith("volatile_"):
            pat_order.append(k)
    print(f"\n  {'pattern':30s} {'total':>7s}  " + " ".join(f"{c:>17s}" for c in CATEGORY_ORDER))
    for name in pat_order:
        if pat_counts.get(name, 0) == 0:
            continue
        cells = " ".join(f"{pat_by_cat[name].get(c, 0):>17d}" for c in CATEGORY_ORDER)
        print(f"  {name:30s} {pat_counts[name]:>7d}  {cells}")


def print_flapping_table(results, max_print=80):
    if len(results) < 2:
        return
    all_ids = set(results[0]["by_doc"].keys())
    for r in results[1:]:
        all_ids &= set(r["by_doc"].keys())
    all_ids = sorted(all_ids)
    flapping = []
    for d in all_ids:
        pat = tuple(r["by_doc"][d][0] for r in results)
        if len(set(pat)) > 1:
            flapping.append((d, pat))
    if not flapping:
        return
    print("\n" + "=" * 100)
    print(f"FLAPPING SAMPLES — first {min(max_print, len(flapping))} of {len(flapping)}")
    print("=" * 100)
    name_w = max(4, max(len(r["name"]) for r in results))
    head_marks = " ".join(f"{r['name'][:name_w]:>{name_w}s}" for r in results)
    print(f"\n  {'doc_id':>6s}  {'category':18s}  {'subject':36s}  "
          f"{head_marks}  pattern")
    flapping.sort(key=lambda x: (results[0]["by_doc"][x[0]][2], results[0]["by_doc"][x[0]][1], x[0]))
    for d, pat in flapping[:max_print]:
        _, subj, cat, _ = results[0]["by_doc"][d]
        marks = " ".join(f"{('✓' if v else '✗'):>{name_w}s}" for v in pat)
        print(f"  {d:>6d}  {cat:18s}  {subj[:36]:36s}  {marks}  {_classify_pattern(pat)}")
    if len(flapping) > max_print:
        print(f"  ... +{len(flapping) - max_print} more (use --csv to dump all)")


def write_csv(results, csv_path):
    import csv
    all_ids = sorted(set(results[0]["by_doc"].keys()))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["doc_id", "category", "subject", "target"]
        for r in results:
            header.append(r["name"])
        header.append("pattern")
        w.writerow(header)
        for d in all_ids:
            _, subj, cat, target = results[0]["by_doc"][d]
            row = [d, cat, subj, target]
            pat = []
            for r in results:
                v = r["by_doc"][d][0]
                row.append(1 if v else 0)
                pat.append(v)
            row.append(_classify_pattern(tuple(pat)))
            w.writerow(row)
    print(f"\n[csv] wrote {len(all_ids)} rows to {csv_path}")


# ---------- CLI ----------

def parse_input(arg):
    if "=" in arg and not arg.startswith(("/", ".")):
        n, p = arg.split("=", 1)
    else:
        p = arg; n = Path(p).stem
    return n, Path(p)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="Labeled samples_mmlu_slime_*.jsonl paths.")
    p.add_argument("--csv", default=None,
                   help="Path to write per-doc-id CSV.")
    p.add_argument("--flapping-table", action="store_true",
                   help="Print per-question flapping table (long).")
    p.add_argument("--top-n-subjects", type=int, default=15,
                   help="How many top movers to show per pairwise transition (default 15).")
    args = p.parse_args()

    results = []
    for inp in args.inputs:
        name, path = parse_input(inp)
        if not path.exists():
            print(f"[skip] {path} does not exist", file=sys.stderr); continue
        r = breakdown(path, name)
        results.append(r)
        print_per_ckpt(r)

    if len(results) > 1:
        print_compare_categories(results)
        print_subject_deltas(results, top_n=args.top_n_subjects)
        print_transitions(results)
        print_pattern_summary(results)
        if args.flapping_table:
            print_flapping_table(results)
        if args.csv:
            write_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()
