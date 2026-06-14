#!/usr/bin/env python3
"""
math500 difficulty (level) × topic (subject) breakdown across checkpoints.

For each labeled samples_*.jsonl input, computes accuracy by:
  - level (1..5)
  - subject (Algebra / Geometry / etc.)
  - level × subject (heatmap form)

Then prints cross-ckpt deltas so you can see WHERE the gains land — e.g.
'base_math beats base mostly on Level-5 Intermediate Algebra'.

Uses the `exact_match` field as-is (already includes sympy regrade if the file
was patched by regrade_math_jsonl.py). Optionally rescues plain-text
hidden_correct cases via --rescue-hidden-text (shells out to the same grader
the audit script uses).

Usage:
  python breakdown_math_eval.py base=…jsonl base_math=…jsonl final_search=…jsonl

  # also count plain-text hidden_correct (solid only, ignores fragile)
  python breakdown_math_eval.py base=…jsonl base_math=…jsonl --rescue-hidden-text
"""
import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_MATH_UTILS_PATH = "/data1/hhzhang/slime/slime/rollout/rm_hub/math_utils.py"


# ---------- response / boxed plumbing (mirrors classify_math_failures.py) ----------

def get_response(s):
    fr = s.get("filtered_resps")
    if isinstance(fr, list) and fr:
        x = fr[0]
        if isinstance(x, str): return x
        if isinstance(x, list) and x and isinstance(x[0], str): return x[0]
    return ""


def strip_think(t):
    return t.split("</think>")[-1] if "</think>" in t else t


def first_boxed(text):
    idx = text.find("\\boxed")
    if idx < 0:
        idx = text.find("\\fbox")
        if idx < 0:
            return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


# ---------- optional plain-text rescue ----------

_CAND_RE = re.compile("|".join(f"({p})" for p in [
    r"\\frac\{-?\d+\}\{-?\d+\}",
    r"\\sqrt\{-?\d+\}",
    r"\\d?frac\{[^{}]+\}\{[^{}]+\}",
    r"-?\d+\s*/\s*-?\d+",
    r"-?\d+\.\d+",
    r"-?\d+",
]))


def _is_fragile(target):
    t = target.strip().strip("$").strip()
    return t in {"0","1","2","3","4","5","6","7","8","9","-1","10"}


def _load_grader(path):
    if not Path(path).exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_g", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    except Exception as e:
        print(f"[warn] can't load grader: {e}", file=sys.stderr)
        return None


def _solid_text_rescue(grader, response, target):
    """True iff response has no \\boxed but a candidate in tail is sympy-equiv to target,
    AND target isn't single-digit (those are fragile and skipped here)."""
    if grader is None or _is_fragile(target):
        return False
    txt = strip_think(response)
    if first_boxed(txt) is not None:
        return False
    tail = txt[-1500:] if len(txt) > 1500 else txt
    cands, seen = [], set()
    for m in _CAND_RE.finditer(tail):
        s = m.group(0).strip()
        if s and s not in seen:
            seen.add(s); cands.append(s)
    cands.reverse()
    t = target
    if "\\boxed" in t:
        t = grader.extract_boxed_answer(t) or t
    for c in cands[:30]:
        try:
            if grader.grade_answer_mathd(c, t) or grader.grade_answer_sympy(c, t):
                return True
        except Exception:
            try:
                if grader.grade_answer_mathd(c, t):
                    return True
            except Exception:
                pass
    return False


# ---------- per-file breakdown ----------

def breakdown(path, name, grader=None, rescue=False):
    by_level = defaultdict(lambda: [0, 0])         # level -> [correct, total]
    by_subject = defaultdict(lambda: [0, 0])
    by_cell = defaultdict(lambda: [0, 0])          # (level, subject) -> [c, t]
    by_doc = {}                                    # doc_id -> (correct, level, subject, target)
    n_rescued = 0

    with open(path) as f:
        for line in f:
            s = json.loads(line)
            doc = s.get("doc", {})
            level = doc.get("level", "?")
            subject = doc.get("subject", "?")
            correct = (s.get("exact_match") == 1.0)
            if not correct and rescue:
                if _solid_text_rescue(grader, get_response(s), s.get("target", "")):
                    correct = True
                    n_rescued += 1
            by_level[level][1] += 1
            by_subject[subject][1] += 1
            by_cell[(level, subject)][1] += 1
            if correct:
                by_level[level][0] += 1
                by_subject[subject][0] += 1
                by_cell[(level, subject)][0] += 1
            by_doc[s["doc_id"]] = (correct, level, subject, s.get("target", ""))

    return dict(name=name, path=str(path),
                by_level=dict(by_level),
                by_subject=dict(by_subject),
                by_cell=dict(by_cell),
                by_doc=by_doc,
                n_rescued=n_rescued,
                n=sum(v[1] for v in by_level.values()),
                correct=sum(v[0] for v in by_level.values()))


# ---------- printing ----------

def _acc(c, t):
    return f"{c/t:.4f}" if t else "  N/A"


def print_per_ckpt(r):
    print(f"\n=== {r['name']} ===")
    print(f"  file: {r['path']}")
    rescue_note = f"  (+{r['n_rescued']} rescued via plain-text hidden_correct)" if r["n_rescued"] else ""
    print(f"  overall: {r['correct']}/{r['n']} = {r['correct']/r['n']:.4f}{rescue_note}")
    print(f"\n  By level:")
    for lvl in sorted(r["by_level"], key=lambda x: (str(x))):
        c, t = r["by_level"][lvl]
        print(f"    Level {lvl}    {c:3d}/{t:3d}  =  {_acc(c, t)}")
    print(f"\n  By subject:")
    for subj, (c, t) in sorted(r["by_subject"].items(), key=lambda x: -x[1][1]):
        print(f"    {subj:25s} {c:3d}/{t:3d}  =  {_acc(c, t)}")


def print_compare(results):
    if len(results) < 2:
        return
    levels = sorted({l for r in results for l in r["by_level"]}, key=str)
    subjects = sorted({s for r in results for s in r["by_subject"]})

    print("\n" + "=" * 90)
    print("LEVEL ACROSS CKPTS")
    print("=" * 90)
    header = f"{'level':12s} " + " ".join(f"{r['name']:>14s}" for r in results) + f"  {'Δ first→last':>14s}"
    print(header)
    print("-" * len(header))
    for lvl in levels:
        cells = []
        accs = []
        for r in results:
            c, t = r["by_level"].get(lvl, (0, 0))
            accs.append(c / t if t else None)
            cells.append(f"{_acc(c, t):>9s} ({c:>2d}/{t:>3d})")
        delta = ""
        if accs[0] is not None and accs[-1] is not None:
            delta = f"{accs[-1] - accs[0]:+.4f}"
        print(f"  Level {str(lvl):4s}  " + " ".join(cells) + f"  {delta:>14s}")

    print("\n" + "=" * 90)
    print("SUBJECT ACROSS CKPTS")
    print("=" * 90)
    print(f"{'subject':28s} " + " ".join(f"{r['name']:>14s}" for r in results) + f"  {'Δ first→last':>14s}")
    print("-" * 90)
    for subj in subjects:
        cells = []
        accs = []
        for r in results:
            c, t = r["by_subject"].get(subj, (0, 0))
            accs.append(c / t if t else None)
            cells.append(f"{_acc(c, t):>9s} ({c:>2d}/{t:>3d})")
        delta = ""
        if accs[0] is not None and accs[-1] is not None:
            delta = f"{accs[-1] - accs[0]:+.4f}"
        print(f"  {subj:26s}" + " ".join(cells) + f"  {delta:>14s}")

    # pairwise deltas
    for i in range(len(results) - 1):
        a, b = results[i], results[i + 1]
        print(f"\n  --- {a['name']} → {b['name']} biggest movers (level × subject) ---")
        rows = []
        for lvl in levels:
            for subj in subjects:
                ca, ta = a["by_cell"].get((lvl, subj), (0, 0))
                cb, tb = b["by_cell"].get((lvl, subj), (0, 0))
                if ta < 5 or tb < 5:           # ignore noisy small cells
                    continue
                d = (cb / tb) - (ca / ta)
                rows.append((d, lvl, subj, (ca, ta), (cb, tb)))
        rows.sort(key=lambda x: x[0])
        if not rows:
            continue
        print(f"    biggest losses ({a['name']} > {b['name']}):")
        for d, lvl, subj, ab, bb in rows[:5]:
            if d >= 0:
                continue
            print(f"      L{lvl} {subj:24s}  {ab[0]}/{ab[1]} ({ab[0]/ab[1]:.3f})  →  "
                  f"{bb[0]}/{bb[1]} ({bb[0]/bb[1]:.3f})   Δ={d:+.4f}")
        print(f"    biggest gains ({b['name']} > {a['name']}):")
        for d, lvl, subj, ab, bb in reversed(rows[-5:]):
            if d <= 0:
                continue
            print(f"      L{lvl} {subj:24s}  {ab[0]}/{ab[1]} ({ab[0]/ab[1]:.3f})  →  "
                  f"{bb[0]}/{bb[1]} ({bb[0]/bb[1]:.3f})   Δ={d:+.4f}")


def _bin_by_level_subject(doc_ids, by_doc):
    """Return (level_counter, subject_counter) for the given doc_ids."""
    lvl = defaultdict(int)
    subj = defaultdict(int)
    for d in doc_ids:
        _, level, subject, _ = by_doc[d]
        lvl[level] += 1
        subj[subject] += 1
    return dict(lvl), dict(subj)


def _fmt_counter(d, key_fmt=str):
    if not d:
        return "—"
    return ", ".join(f"{key_fmt(k)}:{v}" for k, v in sorted(d.items(), key=lambda x: str(x[0])))


def _matrix_table(doc_ids, by_doc, all_levels, all_subjects):
    """Return a 2D dict[level][subject] -> count."""
    m = {l: {s: 0 for s in all_subjects} for l in all_levels}
    for d in doc_ids:
        _, level, subject, _ = by_doc[d]
        if level in m and subject in m[level]:
            m[level][subject] += 1
    return m


def _print_matrix(label, mat, all_levels, all_subjects):
    print(f"\n  {label}")
    print(f"    {'level':6s}" + "".join(f"{s[:9]:>10s}" for s in all_subjects) + f"  {'total':>7s}")
    grand_total = 0
    col_totals = {s: 0 for s in all_subjects}
    for l in all_levels:
        row_total = 0
        cells = []
        for s in all_subjects:
            v = mat[l][s]
            cells.append(f"{v:>10d}" if v else f"{'·':>10s}")
            row_total += v
            col_totals[s] += v
        grand_total += row_total
        print(f"    L{str(l):4s}" + "".join(cells) + f"  {row_total:>7d}")
    print(f"    {'total':6s}"
          + "".join(f"{col_totals[s]:>10d}" for s in all_subjects)
          + f"  {grand_total:>7d}")


def print_fixed_broken_matrices(results):
    """For each consecutive pair, print fixed and broken as level × subject matrices."""
    if len(results) < 2:
        return
    print("\n" + "=" * 90)
    print("FIXED / BROKEN MATRICES (level × subject)")
    print("=" * 90)

    all_ids = set(results[0]["by_doc"].keys())
    for r in results[1:]:
        all_ids &= set(r["by_doc"].keys())
    all_ids = sorted(all_ids)
    all_levels = sorted({results[0]["by_doc"][d][1] for d in all_ids}, key=str)
    all_subjects = sorted({results[0]["by_doc"][d][2] for d in all_ids})

    pairs = [(results[i], results[i + 1]) for i in range(len(results) - 1)]
    if len(results) > 2:
        pairs.append((results[0], results[-1]))

    for a, b in pairs:
        fixed = [d for d in all_ids if not a["by_doc"][d][0] and b["by_doc"][d][0]]
        broken = [d for d in all_ids if a["by_doc"][d][0] and not b["by_doc"][d][0]]
        print(f"\n--- {a['name']} → {b['name']}    fixed={len(fixed)}  broken={len(broken)}  net={len(fixed)-len(broken):+d} ---")
        _print_matrix(f"FIXED ({a['name']} ✗ → {b['name']} ✓):",
                      _matrix_table(fixed, a["by_doc"], all_levels, all_subjects),
                      all_levels, all_subjects)
        _print_matrix(f"BROKEN ({a['name']} ✓ → {b['name']} ✗):",
                      _matrix_table(broken, a["by_doc"], all_levels, all_subjects),
                      all_levels, all_subjects)


def _classify_pattern(pattern: tuple) -> str:
    """Given a tuple of bool correctness across ckpts (ordered base → … → last),
    return a label describing the trajectory.

    Generalized for n=2/3/4. For n=4 the ckpts are assumed to be:
        (base, base_math, final_search, tau2)
    Patterns that match a "monotonic fix" or "monotonic break" get a clean
    name; everything else is "volatile_<bitstring>"."""
    n = len(pattern)
    if len(set(pattern)) == 1:
        return "stable"
    if n == 2:
        a, b = pattern
        return "fixed" if (not a and b) else "broken"
    if n == 3:
        a, b, c = pattern
        if not a and b and c: return "fixed_at_bm"        # ✗✓✓
        if not a and not b and c: return "fixed_at_fs"    # ✗✗✓
        if a and not b and not c: return "broken_at_bm"   # ✓✗✗
        if a and b and not c: return "broken_at_fs"       # ✓✓✗
        if a and not b and c: return "regressed_then_recovered"  # ✓✗✓
        if not a and b and not c: return "fixed_then_broken"     # ✗✓✗
    if n == 4:
        # Monotonic patterns
        if pattern == (False, True, True, True):    return "fixed_at_bm"
        if pattern == (False, False, True, True):   return "fixed_at_fs"
        if pattern == (False, False, False, True):  return "fixed_at_tau2"
        if pattern == (True, False, False, False):  return "broken_at_bm"
        if pattern == (True, True, False, False):   return "broken_at_fs"
        if pattern == (True, True, True, False):    return "broken_at_tau2"
        # Everything else gets a bitstring suffix so it's still grep-able
        return "volatile_" + "".join("1" if x else "0" for x in pattern)
    return "other"


def print_flapping_table(results, max_problem_chars=70):
    """Per-doc-id table for samples whose correctness varies across ckpts."""
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
    print(f"FLAPPING SAMPLES — per-question detail ({len(flapping)} samples)")
    print("=" * 100)

    pat_order = [
        "fixed_at_bm", "fixed_at_fs", "fixed_at_tau2",
        "broken_at_bm", "broken_at_fs", "broken_at_tau2",
        "regressed_then_recovered", "fixed_then_broken", "fixed", "broken",
    ]
    grouped = defaultdict(list)
    for d, pat in flapping:
        grouped[_classify_pattern(pat)].append((d, pat))

    name_w = max(4, max(len(r["name"]) for r in results))
    header_marks = " ".join(f"{r['name'][:name_w]:>{name_w}s}" for r in results)
    print(f"\n  {'doc_id':>6s}  {'L':>1s}  {'subject':24s}  {'target':30s}  {header_marks}  pattern")
    print("  " + "-" * (8 + 4 + 26 + 32 + (name_w + 1) * len(results) + 2 + 28))

    # Append any volatile_<bitstring> buckets that aren't in the canonical
    # pat_order list (only relevant for n=4 mode), preserving alphabetical sort
    # so the output is deterministic.
    extra = sorted(k for k in grouped if k not in pat_order and k != "stable")
    for pat_name in pat_order + extra:
        rows = grouped.get(pat_name, [])
        if not rows:
            continue
        rows.sort(key=lambda x: (results[0]["by_doc"][x[0]][1], results[0]["by_doc"][x[0]][2], x[0]))
        for d, pat in rows:
            _, lvl, subj, target = results[0]["by_doc"][d]
            target_s = (target[:max_problem_chars - 3] + "...") if len(target) > max_problem_chars else target
            marks = " ".join(f"{('✓' if v else '✗'):>{name_w}s}" for v in pat)
            print(f"  {d:>6d}  {str(lvl):>1s}  {subj[:24]:24s}  {target_s[:30]:30s}  {marks}  {pat_name}")
        print()  # blank between groups


def write_csv(results, csv_path: Path):
    """Dump a per-doc-id CSV with per-ckpt correctness for spreadsheet analysis."""
    import csv
    all_ids = sorted(set(results[0]["by_doc"].keys()))

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["doc_id", "level", "subject", "target"]
        for r in results:
            header.append(r["name"])
        header.append("pattern")
        w.writerow(header)

        for d in all_ids:
            _, lvl, subj, target = results[0]["by_doc"][d]
            row = [d, lvl, subj, target]
            pat = []
            for r in results:
                v = r["by_doc"][d][0]
                row.append(1 if v else 0)
                pat.append(v)
            row.append(_classify_pattern(tuple(pat)))
            w.writerow(row)
    print(f"\n[csv] wrote {len(all_ids)} rows to {csv_path}")


def print_transitions(results, max_list=25):
    """For each consecutive pair of ckpts (and overall first→last), list
    fixed (was wrong, now correct) and broken (was correct, now wrong),
    plus level/subject breakdown of each."""
    if len(results) < 2:
        return
    print("\n" + "=" * 90)
    print("TRANSITION ANALYSIS (per-question fixed / broken)")
    print("=" * 90)

    # Use the doc-id intersection across all runs (should be 500 anyway)
    all_ids = set(results[0]["by_doc"].keys())
    for r in results[1:]:
        all_ids &= set(r["by_doc"].keys())
    all_ids = sorted(all_ids)

    persistent_correct = [d for d in all_ids if all(r["by_doc"][d][0] for r in results)]
    persistent_wrong   = [d for d in all_ids if all(not r["by_doc"][d][0] for r in results)]
    flapping = [d for d in all_ids if d not in persistent_correct and d not in persistent_wrong]

    print(f"  total doc_ids:           {len(all_ids)}")
    print(f"  persistent correct (all ✓): {len(persistent_correct)}")
    print(f"  persistent wrong   (all ✗): {len(persistent_wrong)}")
    print(f"  flapping (varies):          {len(flapping)}")

    persistent_wrong_lvl, persistent_wrong_subj = _bin_by_level_subject(persistent_wrong, results[0]["by_doc"])
    print(f"\n  persistent-wrong by level:    {_fmt_counter(persistent_wrong_lvl, lambda l: f'L{l}')}")
    print(f"  persistent-wrong by subject:  {_fmt_counter(persistent_wrong_subj)}")

    pairs = [(results[i], results[i + 1]) for i in range(len(results) - 1)]
    if len(results) > 2:
        pairs.append((results[0], results[-1]))   # overall first → last

    for a, b in pairs:
        fixed = [d for d in all_ids if not a["by_doc"][d][0] and b["by_doc"][d][0]]
        broken = [d for d in all_ids if a["by_doc"][d][0] and not b["by_doc"][d][0]]
        net = len(fixed) - len(broken)
        print(f"\n  --- {a['name']}  →  {b['name']}    "
              f"fixed={len(fixed)}  broken={len(broken)}  net={net:+d} ---")

        if fixed:
            lvl, subj = _bin_by_level_subject(fixed, a["by_doc"])
            print(f"    fixed by level:   {_fmt_counter(lvl, lambda l: f'L{l}')}")
            print(f"    fixed by subject: {_fmt_counter(subj)}")
            print(f"    fixed doc_ids:    {fixed[:max_list]}"
                  f"{'  +' + str(len(fixed) - max_list) + ' more' if len(fixed) > max_list else ''}")
        if broken:
            lvl, subj = _bin_by_level_subject(broken, a["by_doc"])
            print(f"    broken by level:   {_fmt_counter(lvl, lambda l: f'L{l}')}")
            print(f"    broken by subject: {_fmt_counter(subj)}")
            print(f"    broken doc_ids:    {broken[:max_list]}"
                  f"{'  +' + str(len(broken) - max_list) + ' more' if len(broken) > max_list else ''}")


def print_cell_matrix(results):
    """Compact level × subject heatmap printed once per ckpt."""
    levels = sorted({l for r in results for l in r["by_level"]}, key=str)
    subjects = sorted({s for r in results for s in r["by_subject"]})
    print("\n" + "=" * 90)
    print("LEVEL × SUBJECT HEATMAP (acc per ckpt)")
    print("=" * 90)
    for r in results:
        print(f"\n  --- {r['name']} ---")
        col_w = max(10, max(len(s) for s in subjects) + 1)
        head = f"  {'level':6s}" + "".join(f"{s[:9]:>10s}" for s in subjects)
        print(head)
        for lvl in levels:
            row = [f"  L{str(lvl):4s}"]
            for subj in subjects:
                c, t = r["by_cell"].get((lvl, subj), (0, 0))
                cell = f"{c}/{t}" if t else "—"
                acc = c / t if t else None
                if acc is None:
                    row.append(f"{'—':>10s}")
                else:
                    row.append(f"{acc:>5.2f}{cell:>5s}")
            print("".join(row))


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
    p.add_argument("inputs", nargs="+",
                   help="Labeled samples_*.jsonl paths. Use NAME=PATH.")
    p.add_argument("--rescue-hidden-text", action="store_true",
                   help="Also count plain-text hidden_correct (solid; skips fragile single-digit targets).")
    p.add_argument("--math-utils-path", default=DEFAULT_MATH_UTILS_PATH)
    p.add_argument("--csv", default=None,
                   help="Path to write per-doc-id CSV (doc_id, level, subject, target, "
                        "<correct flag per ckpt>, pattern).")
    p.add_argument("--no-flapping-table", action="store_true",
                   help="Skip the per-question flapping table (it's long).")
    args = p.parse_args()

    grader = _load_grader(args.math_utils_path) if args.rescue_hidden_text else None

    results = []
    for inp in args.inputs:
        name, path = parse_input(inp)
        if not path.exists():
            print(f"[skip] {path} does not exist", file=sys.stderr); continue
        r = breakdown(path, name, grader=grader, rescue=args.rescue_hidden_text)
        results.append(r)
        print_per_ckpt(r)

    if len(results) > 1:
        print_compare(results)
        print_transitions(results)
        print_fixed_broken_matrices(results)
        if not args.no_flapping_table:
            print_flapping_table(results)
        print_cell_matrix(results)
        if args.csv:
            write_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()
