#!/usr/bin/env python3
"""
gpqa_slime breakdown across checkpoints, with fixed/broken matrices and
per-question pattern table.

GPQA Diamond is small (198 samples) — fits the math500-style detail mode:
  by High-level domain (Physics / Chemistry / Biology) AND by Subdomain (13).
  Per-doc flapping table prints in full by default.

Usage:
  python breakdown_gpqa_eval.py base=…jsonl base_math=…jsonl final_search=…jsonl
  python breakdown_gpqa_eval.py … --csv out.csv
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


DOMAIN_ORDER = ["Physics", "Chemistry", "Biology"]


import re as _re
_BOXED_LETTER_RE = _re.compile(r"\\boxed\{\(?([A-D])\)?\}")


def _has_boxed(response):
    """True if the response (post-think) contains a \\boxed{X} where X in A-D."""
    if not response:
        return False
    text = response.split("</think>")[-1] if "</think>" in response else response
    return bool(_BOXED_LETTER_RE.search(text))


def breakdown(path, name):
    by_domain = defaultdict(lambda: [0, 0])
    by_subdomain = defaultdict(lambda: [0, 0])
    by_doc = {}
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            doc = s.get("doc", {})
            dom = doc.get("High-level domain", "?")
            sub = doc.get("Subdomain", "?")
            correct = (s.get("exact_match") == 1.0)
            # Has a \\boxed{X} in the post-think tail?
            fr = s.get("filtered_resps")
            resp = ""
            if isinstance(fr, list) and fr:
                x = fr[0]
                resp = x if isinstance(x, str) else (x[0] if isinstance(x, list) and x else "")
            via_boxed = _has_boxed(resp)
            by_domain[dom][1] += 1
            by_subdomain[sub][1] += 1
            if correct:
                by_domain[dom][0] += 1
                by_subdomain[sub][0] += 1
            by_doc[s["doc_id"]] = (correct, dom, sub, s.get("target", ""),
                                   doc.get("Question", "")[:80], via_boxed)
    return dict(name=name, path=str(path),
                by_domain=dict(by_domain),
                by_subdomain=dict(by_subdomain),
                by_doc=by_doc,
                n=sum(v[1] for v in by_domain.values()),
                correct=sum(v[0] for v in by_domain.values()),
                n_via_boxed=sum(1 for v in by_doc.values() if v[5]),
                n_via_boxed_correct=sum(1 for v in by_doc.values() if v[5] and v[0]))


def _acc(c, t):
    return f"{c/t:.4f}" if t else "  N/A"


def print_per_ckpt(r):
    print(f"\n=== {r['name']} ===")
    print(f"  file: {r['path']}")
    print(f"  overall: {r['correct']}/{r['n']} = {r['correct']/r['n']:.4f}")
    nvb = r["n_via_boxed"]
    nvbc = r["n_via_boxed_correct"]
    print(f"  via \\boxed{{X}} format: {nvb}/{r['n']} ({nvb/r['n']*100:.1f}%)  "
          f"acc on those: {nvbc}/{nvb} ({(nvbc/nvb if nvb else 0):.4f})")
    print(f"\n  By high-level domain:")
    for dom in DOMAIN_ORDER:
        c, t = r["by_domain"].get(dom, (0, 0))
        print(f"    {dom:18s} {c:3d}/{t:3d}  =  {_acc(c, t)}")
    print(f"\n  By subdomain (sorted by n):")
    for sub, (c, t) in sorted(r["by_subdomain"].items(), key=lambda x: -x[1][1]):
        print(f"    {sub:35s} {c:3d}/{t:3d}  =  {_acc(c, t)}")


def print_compare(results):
    if len(results) < 2:
        return

    print("\n" + "=" * 90)
    print("HIGH-LEVEL DOMAIN ACROSS CKPTS")
    print("=" * 90)
    header = f"{'domain':14s} " + " ".join(f"{r['name']:>17s}" for r in results) + f"  {'Δ first→last':>14s}"
    print(header)
    print("-" * len(header))
    for dom in DOMAIN_ORDER:
        cells, accs = [], []
        for r in results:
            c, t = r["by_domain"].get(dom, (0, 0))
            accs.append(c / t if t else None)
            cells.append(f"{_acc(c, t):>9s} ({c:>3d}/{t:>3d})")
        d = f"{accs[-1]-accs[0]:+.4f}" if (accs[0] is not None and accs[-1] is not None) else ""
        print(f"  {dom:12s} " + " ".join(cells) + f"  {d:>14s}")

    print("\n" + "=" * 90)
    print("SUBDOMAIN ACROSS CKPTS")
    print("=" * 90)
    print(f"{'subdomain':35s} " + " ".join(f"{r['name']:>17s}" for r in results) + f"  {'Δ first→last':>14s}")
    print("-" * 90)
    all_subs = sorted({s for r in results for s in r["by_subdomain"]})
    rows = []
    for sub in all_subs:
        sizes = [r["by_subdomain"].get(sub, (0, 0)) for r in results]
        accs = [c / t if t else None for c, t in sizes]
        d = (accs[-1] - accs[0]) if (accs[0] is not None and accs[-1] is not None) else None
        rows.append((d if d is not None else 0, sub, sizes, accs))
    rows.sort(key=lambda x: x[2][-1][1] or 0, reverse=True)  # by n in last ckpt desc
    for _, sub, sizes, accs in rows:
        cells = []
        for (c, t) in sizes:
            cells.append(f"{_acc(c, t):>9s} ({c:>2d}/{t:>2d})")
        d = (accs[-1] - accs[0]) if (accs[0] is not None and accs[-1] is not None) else None
        d_str = f"{d:+.4f}" if d is not None else ""
        print(f"  {sub:33s} " + " ".join(cells) + f"  {d_str:>14s}")


def _classify_pattern(pattern):
    if len(set(pattern)) == 1: return "stable"
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

    pwd_dom = defaultdict(int); pwd_sub = defaultdict(int)
    for d in all_ids:
        if all(not r["by_doc"][d][0] for r in results):
            pwd_dom[results[0]["by_doc"][d][1]] += 1
            pwd_sub[results[0]["by_doc"][d][2]] += 1
    print(f"\n  persistent-wrong by domain: {dict(pwd_dom)}")
    print(f"  persistent-wrong by subdomain (top): "
          f"{sorted(pwd_sub.items(), key=lambda x:-x[1])[:5]}")

    pairs = [(results[i], results[i + 1]) for i in range(len(results) - 1)]
    if len(results) > 2:
        pairs.append((results[0], results[-1]))

    for a, b in pairs:
        fixed = [d for d in all_ids if not a["by_doc"][d][0] and b["by_doc"][d][0]]
        broken = [d for d in all_ids if a["by_doc"][d][0] and not b["by_doc"][d][0]]
        print(f"\n  --- {a['name']} → {b['name']}    "
              f"fixed={len(fixed)}  broken={len(broken)}  net={len(fixed)-len(broken):+d} ---")

        fixed_dom = defaultdict(int); broken_dom = defaultdict(int)
        for d in fixed: fixed_dom[a["by_doc"][d][1]] += 1
        for d in broken: broken_dom[a["by_doc"][d][1]] += 1
        print(f"    {'domain':14s} {'fixed':>6s} {'broken':>6s} {'net':>6s}")
        for dom in DOMAIN_ORDER:
            f, b_ = fixed_dom.get(dom, 0), broken_dom.get(dom, 0)
            print(f"    {dom:14s} {f:>6d} {b_:>6d} {f-b_:>+6d}")

        fs, bs = defaultdict(int), defaultdict(int)
        for d in fixed: fs[a["by_doc"][d][2]] += 1
        for d in broken: bs[a["by_doc"][d][2]] += 1
        if fs or bs:
            print(f"\n    by subdomain (fixed / broken / net):")
            all_sub = set(fs) | set(bs)
            rows = sorted([(s, fs.get(s, 0), bs.get(s, 0)) for s in all_sub],
                          key=lambda x: -(x[1] - x[2]))
            for s, f, b_ in rows:
                if f == 0 and b_ == 0: continue
                print(f"      {s:35s} {f:>3d} / {b_:>3d}  net={f-b_:+d}")


def print_pattern_summary(results):
    if len(results) != 3:
        return
    print("\n" + "=" * 90)
    print("PER-QUESTION PATTERN SUMMARY")
    print("=" * 90)
    all_ids = set(results[0]["by_doc"].keys())
    for r in results[1:]:
        all_ids &= set(r["by_doc"].keys())
    all_ids = sorted(all_ids)

    pat_counts = defaultdict(int)
    pat_by_dom = defaultdict(lambda: defaultdict(int))
    for d in all_ids:
        pat = tuple(r["by_doc"][d][0] for r in results)
        name = _classify_pattern(pat)
        pat_counts[name] += 1
        pat_by_dom[name][results[0]["by_doc"][d][1]] += 1
    pat_order = ["stable",
                 "fixed_at_bm", "fixed_at_fs", "fixed_at_tau2",
                 "broken_at_bm", "broken_at_fs", "broken_at_tau2",
                 "regressed_then_recovered", "fixed_then_broken", "other"]
    # Append any volatile_<bitstring> buckets seen (4-ckpt mode)
    for k in sorted(pat_counts):
        if k not in pat_order and k.startswith("volatile_"):
            pat_order.append(k)
    print(f"\n  {'pattern':30s} {'total':>7s}  " + " ".join(f"{c:>10s}" for c in DOMAIN_ORDER))
    for name in pat_order:
        if pat_counts.get(name, 0) == 0: continue
        cells = " ".join(f"{pat_by_dom[name].get(c, 0):>10d}" for c in DOMAIN_ORDER)
        print(f"  {name:30s} {pat_counts[name]:>7d}  {cells}")


def print_flapping_table(results):
    if len(results) < 2: return
    all_ids = set(results[0]["by_doc"].keys())
    for r in results[1:]:
        all_ids &= set(r["by_doc"].keys())
    flapping = []
    for d in sorted(all_ids):
        pat = tuple(r["by_doc"][d][0] for r in results)
        if len(set(pat)) > 1:
            flapping.append((d, pat))
    if not flapping: return
    print("\n" + "=" * 100)
    print(f"FLAPPING SAMPLES — per-question detail ({len(flapping)} total)")
    print("=" * 100)
    name_w = max(4, max(len(r["name"]) for r in results))
    head = " ".join(f"{r['name'][:name_w]:>{name_w}s}" for r in results)
    print(f"\n  {'doc_id':>6s}  {'domain':10s}  {'subdomain':32s}  {head}  pattern")
    flapping.sort(key=lambda x: (
        results[0]["by_doc"][x[0]][1], results[0]["by_doc"][x[0]][2], x[0]))
    for d, pat in flapping:
        _, dom, sub, _, _, _ = results[0]["by_doc"][d]
        marks = " ".join(f"{('✓' if v else '✗'):>{name_w}s}" for v in pat)
        print(f"  {d:>6d}  {dom[:10]:10s}  {sub[:32]:32s}  {marks}  {_classify_pattern(pat)}")


def write_csv(results, path):
    import csv
    all_ids = sorted(set(results[0]["by_doc"].keys()))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["doc_id", "domain", "subdomain", "target"]
        for r in results:
            header.append(r["name"])                 # 1/0 correctness
            header.append(f"{r['name']}_via_boxed")  # 1/0 used \\boxed{{X}}
        header.append("pattern")
        w.writerow(header)
        for d in all_ids:
            _, dom, sub, target, _, _ = results[0]["by_doc"][d]
            row = [d, dom, sub, target]
            pat = []
            for r in results:
                tup = r["by_doc"][d]
                row.append(1 if tup[0] else 0)
                row.append(1 if tup[5] else 0)
                pat.append(tup[0])
            row.append(_classify_pattern(tuple(pat)))
            w.writerow(row)
    print(f"\n[csv] wrote {len(all_ids)} rows to {path}")


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
    p.add_argument("--no-flapping-table", action="store_true")
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
        if not args.no_flapping_table:
            print_flapping_table(results)
        if args.csv:
            write_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()
