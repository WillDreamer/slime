#!/usr/bin/env python3
"""
Analyze and compare math evaluation results from lm-eval-harness style sample jsonl files.

Designed for math500_slime / similar tasks where each line is a sample dict with:
  doc_id, doc.{problem,subject,level,answer}, target, filtered_resps, exact_match

Usage:
  # Single run
  python analyze_math_eval.py path/to/samples_math500_slime_*.jsonl

  # Compare multiple checkpoints (label each)
  python analyze_math_eval.py \
      ckpt100=/path/to/ckpt100/samples_math500_slime_*.jsonl \
      ckpt200=/path/to/ckpt200/samples_math500_slime_*.jsonl

  # Dump per-error-category details for inspection
  python analyze_math_eval.py run.jsonl --dump-wrongs out_dir/

  # Show worst-N hardest problems (failed by most checkpoints)
  python analyze_math_eval.py *.jsonl --hardest 20

  # Re-grade with slime's mathd+sympy graders (catches LaTeX-equivalence errors
  # the harness misses, e.g. 0.09 vs \\frac{9}{100}, C vs \\text{(C)}, 1/5 vs \\frac{1}{5}).
  # Also uses FIRST \\boxed{} for robustness against post-answer rambling/loops.
  python analyze_math_eval.py *.jsonl --regrade
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


# -------- answer extraction (mirrors math500_slime process_results) --------

def strip_think(text: str) -> str:
    return text.split("</think>")[-1] if "</think>" in text else text


def extract_boxed(text: str):
    """Last \\boxed{...} with balanced braces. Returns None if missing/unbalanced."""
    idx = text.rfind("\\boxed")
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


def get_response(sample) -> str:
    fr = sample.get("filtered_resps")
    if isinstance(fr, list) and fr:
        x = fr[0]
        if isinstance(x, str):
            return x
        if isinstance(x, list) and x and isinstance(x[0], str):
            return x[0]
    rs = sample.get("resps")
    if isinstance(rs, list) and rs and isinstance(rs[0], list) and rs[0]:
        return rs[0][0] if isinstance(rs[0][0], str) else ""
    return ""


def first_boxed(text: str):
    """First \\boxed{...} with balanced braces. More robust than last for samples
    where the model rambles/loops after giving the answer (which produces stray
    empty \\boxed{} patterns later in the response)."""
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


# -------- regrade with slime's stricter math_utils (mathd + sympy) --------

DEFAULT_MATH_UTILS_PATH = "/data1/hhzhang/slime/slime/rollout/rm_hub/math_utils.py"


def _load_regrader(math_utils_path: str):
    """Lazy-load slime/rollout/rm_hub/math_utils.py via importlib (bypasses
    slime/__init__.py + rm_hub/__init__.py which pull in aiohttp etc.).
    Returns the module, or None if missing/unloadable."""
    if not Path(math_utils_path).exists():
        print(f"[warn] --regrade: math_utils.py not found at {math_utils_path}", file=sys.stderr)
        return None
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("_slime_math_for_regrade", math_utils_path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        # Sanity check required attrs
        for fn in ("grade_answer_mathd", "grade_answer_sympy", "extract_boxed_answer"):
            if not hasattr(m, fn):
                print(f"[warn] --regrade: math_utils missing {fn}", file=sys.stderr)
                return None
        return m
    except Exception as e:
        print(f"[warn] --regrade: failed to import math_utils ({e.__class__.__name__}: {e}); "
              "make sure sympy + pylatexenc are installed in this env.", file=sys.stderr)
        return None


def regrade_in_place(samples, regrader) -> tuple[int, int]:
    """For each sample harness called WRONG, try to rescue with slime's
    grader (first_boxed + mathd-or-sympy). Mutates sample['exact_match'] = 1.0
    on rescue. Returns (n_harness_correct, n_rescued)."""
    n_harness_correct = 0
    n_rescued = 0
    for s in samples:
        harness_ok = (s.get("exact_match") == 1.0)
        if harness_ok:
            n_harness_correct += 1
            continue
        extracted = first_boxed(strip_think(get_response(s)))
        if extracted is None:
            continue
        target = s.get("target", "")
        if "\\boxed" in target:
            target = regrader.extract_boxed_answer(target) or target
        try:
            ok = (regrader.grade_answer_mathd(extracted, target)
                  or regrader.grade_answer_sympy(extracted, target))
        except Exception:
            ok = regrader.grade_answer_mathd(extracted, target)
        if ok:
            s["exact_match"] = 1.0
            n_rescued += 1
    return n_harness_correct, n_rescued


# -------- error categorization --------

# Why a sample is wrong:
#   correct            : graded right
#   no_boxed           : response has no \boxed{} at all (format failure)
#   truncated_no_boxed : response is very long AND has no \boxed{} (likely hit max_gen_toks)
#   wrong_truncated    : has \boxed{} but response is very long (likely repetition before answer)
#   wrong_answer       : extracted a clean \boxed{} but it's wrong (genuine reasoning error)

def categorize(sample, long_thresh: int):
    if sample.get("exact_match") == 1.0:
        return "correct", None
    txt = get_response(sample)
    truncated = len(txt) > long_thresh
    extracted = extract_boxed(strip_think(txt))
    if extracted is None:
        return ("truncated_no_boxed" if truncated else "no_boxed"), None
    if truncated:
        return "wrong_truncated", extracted
    return "wrong_answer", extracted


CATEGORY_ORDER = ["correct", "no_boxed", "truncated_no_boxed", "wrong_truncated", "wrong_answer"]


# -------- per-run summary --------

def summarize(path: Path, name: str, long_thresh: int, regrader=None):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))

    # When regrading, mutate exact_match in place — all downstream logic stays the same
    n_harness_correct = sum(1 for s in samples if s.get("exact_match") == 1.0)
    n_rescued = 0
    if regrader is not None:
        _, n_rescued = regrade_in_place(samples, regrader)

    by_doc = {}
    cats = Counter()
    subj_stats = defaultdict(lambda: [0, 0])
    lvl_stats = defaultdict(lambda: [0, 0])

    for s in samples:
        cat, extracted = categorize(s, long_thresh)
        cats[cat] += 1
        doc = s.get("doc", {})
        subj = doc.get("subject", "?")
        lvl = doc.get("level", "?")
        is_correct = (cat == "correct")
        subj_stats[subj][1] += 1
        lvl_stats[lvl][1] += 1
        if is_correct:
            subj_stats[subj][0] += 1
            lvl_stats[lvl][0] += 1
        by_doc[s["doc_id"]] = {
            "doc_id": s["doc_id"],
            "subject": subj,
            "level": lvl,
            "target": s.get("target", ""),
            "correct": is_correct,
            "category": cat,
            "extracted": extracted,
            "resp_len": len(get_response(s)),
            "problem": doc.get("problem", ""),
            "response": get_response(s),
        }

    return {
        "name": name,
        "path": str(path),
        "n": len(samples),
        "correct": cats["correct"],
        "n_harness_correct": n_harness_correct,
        "n_rescued": n_rescued,
        "regraded": regrader is not None,
        "categories": cats,
        "subj": dict(subj_stats),
        "lvl": dict(lvl_stats),
        "by_doc": by_doc,
    }


# -------- printing --------

def print_summary(s):
    print(f"\n=== {s['name']} ===")
    print(f"file:     {s['path']}")
    n, c = s["n"], s["correct"]
    if s.get("regraded"):
        h = s["n_harness_correct"]
        r = s["n_rescued"]
        print(f"accuracy (harness original): {h}/{n} = {h/n:.4f}")
        print(f"accuracy (slime regrade):    {c}/{n} = {c/n:.4f}   (+{r} rescued)")
    else:
        print(f"accuracy: {c}/{n} = {c/n:.4f}")
    cats = s["categories"]
    nwrong = n - c
    print(f"\nerror breakdown ({nwrong} wrong total):")
    for k in CATEGORY_ORDER[1:]:
        v = cats.get(k, 0)
        pct = v / nwrong * 100 if nwrong else 0
        print(f"  {k:20s} {v:4d}  ({pct:5.1f}% of errors)")

    print("\nby subject:")
    for subj, (cc, t) in sorted(s["subj"].items(), key=lambda x: -x[1][1]):
        print(f"  {subj:25s} {cc:3d}/{t:3d} = {cc/t:.4f}")

    print("\nby level:")
    for lvl, (cc, t) in sorted(s["lvl"].items()):
        print(f"  Level {lvl}                  {cc:3d}/{t:3d} = {cc/t:.4f}")


def print_compare(summaries):
    if len(summaries) < 2:
        return
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON")
    print("=" * 80)

    # headline
    header = f"{'metric':25s} " + " ".join(f"{s['name']:>14s}" for s in summaries)
    print(header)
    print("-" * len(header))
    if any(s.get("regraded") for s in summaries):
        print(f"{'accuracy (harness)':25s} " +
              " ".join(f"{s['n_harness_correct']/s['n']:>14.4f}" for s in summaries))
        print(f"{'accuracy (regraded)':25s} " +
              " ".join(f"{s['correct']/s['n']:>14.4f}" for s in summaries))
        print(f"{'rescued by regrade':25s} " +
              " ".join(f"{('+' + str(s['n_rescued'])):>14s}" for s in summaries))
    else:
        print(f"{'accuracy':25s} " + " ".join(f"{s['correct']/s['n']:>14.4f}" for s in summaries))
    for cat in CATEGORY_ORDER[1:]:
        row = [f"{cat:25s}"]
        for s in summaries:
            v = s["categories"].get(cat, 0)
            n = s["n"]
            row.append(f"{v:>5d} ({v/n*100:>4.1f}%)")
        print(" ".join(row))

    # alignment
    all_ids = sorted(summaries[0]["by_doc"].keys())
    for s in summaries[1:]:
        if sorted(s["by_doc"].keys()) != all_ids:
            print("\n[warn] doc_id sets differ across runs; using intersection")
            ids = set(all_ids)
            for ss in summaries[1:]:
                ids &= set(ss["by_doc"].keys())
            all_ids = sorted(ids)
            break

    persistent_wrong = [d for d in all_ids if all(s["by_doc"][d]["correct"] is False for s in summaries)]
    persistent_right = [d for d in all_ids if all(s["by_doc"][d]["correct"] is True for s in summaries)]
    flapping = [d for d in all_ids if d not in persistent_wrong and d not in persistent_right]

    print(f"\npersistent correct (all {len(summaries)} runs):  {len(persistent_right)}")
    print(f"persistent wrong   (all {len(summaries)} runs):  {len(persistent_wrong)}")
    print(f"flapping           (varies across runs):     {len(flapping)}")

    # pairwise diffs
    for i in range(len(summaries) - 1):
        a, b = summaries[i], summaries[i + 1]
        fixed = [d for d in all_ids if not a["by_doc"][d]["correct"] and b["by_doc"][d]["correct"]]
        broken = [d for d in all_ids if a["by_doc"][d]["correct"] and not b["by_doc"][d]["correct"]]
        print(f"\n  {a['name']}  ->  {b['name']}")
        print(f"    fixed:  {len(fixed):3d}   doc_ids: {fixed[:25]}{'...' if len(fixed)>25 else ''}")
        print(f"    broken: {len(broken):3d}   doc_ids: {broken[:25]}{'...' if len(broken)>25 else ''}")

        # error-category transition for the broken ones (was correct -> now what?)
        if broken:
            cat_count = Counter(b["by_doc"][d]["category"] for d in broken)
            print(f"    broken-into categories: {dict(cat_count)}")


def print_hardest(summaries, top_n: int):
    if not summaries:
        return
    all_ids = sorted(summaries[0]["by_doc"].keys())
    fail_count = {}
    for d in all_ids:
        fail_count[d] = sum(1 for s in summaries if not s["by_doc"].get(d, {}).get("correct", True))
    hardest = sorted(fail_count.items(), key=lambda x: -x[1])[:top_n]
    print("\n" + "=" * 80)
    print(f"HARDEST {top_n} PROBLEMS (failed by most runs)")
    print("=" * 80)
    for d, fc in hardest:
        if fc == 0:
            continue
        ref = summaries[0]["by_doc"][d]
        cats = [s["by_doc"].get(d, {}).get("category", "?") for s in summaries]
        print(f"  doc_id={d:3d}  fails={fc}/{len(summaries)}  "
              f"{ref['subject']:22s} L{ref['level']}  target={str(ref['target'])[:30]!r}  "
              f"cats={cats}")


# -------- dump wrong examples for human inspection --------

def dump_wrongs(summary, out_dir: Path, max_resp_chars: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_cat = defaultdict(list)
    for d, w in summary["by_doc"].items():
        if not w["correct"]:
            by_cat[w["category"]].append(w)

    index_path = out_dir / f"{summary['name']}_index.txt"
    with open(index_path, "w") as ix:
        ix.write(f"# {summary['name']}\n")
        ix.write(f"# accuracy: {summary['correct']}/{summary['n']}\n\n")
        for cat in CATEGORY_ORDER[1:]:
            items = sorted(by_cat.get(cat, []), key=lambda x: x["doc_id"])
            ix.write(f"## {cat}: {len(items)}\n")
            for w in items:
                ix.write(f"  doc_id={w['doc_id']:3d}  L{w['level']}  {w['subject']:22s}  "
                         f"target={str(w['target'])[:40]!r}  extracted={str(w['extracted'])[:40]!r}\n")
            ix.write("\n")

        for cat in CATEGORY_ORDER[1:]:
            items = sorted(by_cat.get(cat, []), key=lambda x: x["doc_id"])
            if not items:
                continue
            cat_path = out_dir / f"{summary['name']}_{cat}.txt"
            with open(cat_path, "w") as f:
                for w in items:
                    f.write("=" * 80 + "\n")
                    f.write(f"doc_id={w['doc_id']}  subject={w['subject']}  level={w['level']}\n")
                    f.write(f"category: {w['category']}\n")
                    f.write(f"target:    {w['target']}\n")
                    f.write(f"extracted: {w['extracted']}\n")
                    f.write(f"resp_len:  {w['resp_len']}\n")
                    f.write(f"\n--- problem ---\n{w['problem']}\n")
                    resp = w["response"]
                    if len(resp) > max_resp_chars:
                        f.write(f"\n--- response (truncated to {max_resp_chars}/{len(resp)} chars) ---\n")
                        f.write(resp[:max_resp_chars])
                        f.write("\n... [TRUNCATED] ...\n")
                        f.write(resp[-500:])
                    else:
                        f.write("\n--- response ---\n")
                        f.write(resp)
                    f.write("\n\n")
    print(f"\n[dumped wrongs to {out_dir}/]")


# -------- CLI --------

def parse_input(arg: str):
    """Accepts 'NAME=PATH' or just 'PATH'. Returns (name, Path)."""
    if "=" in arg and not arg.startswith("/") and not arg.startswith("."):
        name, path = arg.split("=", 1)
    else:
        path = arg
        name = Path(path).stem
    return name, Path(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="One or more sample jsonl paths. Use NAME=PATH to label.")
    p.add_argument("--long-thresh", type=int, default=30000,
                   help="Char count above which a response is treated as 'truncated/repeating' (default 30000)")
    p.add_argument("--dump-wrongs", metavar="DIR", default=None,
                   help="Write per-category wrong-answer dumps under DIR for inspection.")
    p.add_argument("--max-resp-chars", type=int, default=8000,
                   help="When dumping, truncate each response to this many chars (default 8000)")
    p.add_argument("--hardest", type=int, default=0,
                   help="Print the N problems failed by the most runs.")
    p.add_argument("--regrade", action="store_true",
                   help="Re-grade with slime's mathd+sympy graders on first \\boxed{} (catches "
                        "LaTeX-equivalence cases harness misses, e.g. 0.09 vs \\frac{9}{100}).")
    p.add_argument("--math-utils-path", default=DEFAULT_MATH_UTILS_PATH,
                   help=f"Path to slime/rollout/rm_hub/math_utils.py for --regrade "
                        f"(default: {DEFAULT_MATH_UTILS_PATH})")
    args = p.parse_args()

    regrader = None
    if args.regrade:
        regrader = _load_regrader(args.math_utils_path)
        if regrader is None:
            print("[error] --regrade requested but graders unavailable; aborting.", file=sys.stderr)
            sys.exit(1)

    summaries = []
    for arg in args.inputs:
        name, path = parse_input(arg)
        if not path.exists():
            print(f"[error] {path} does not exist", file=sys.stderr)
            sys.exit(1)
        s = summarize(path, name, args.long_thresh, regrader=regrader)
        summaries.append(s)
        print_summary(s)

    if len(summaries) > 1:
        print_compare(summaries)

    if args.hardest:
        print_hardest(summaries, args.hardest)

    if args.dump_wrongs:
        out_dir = Path(args.dump_wrongs)
        for s in summaries:
            dump_wrongs(s, out_dir, args.max_resp_chars)


if __name__ == "__main__":
    main()
