#!/usr/bin/env python3
"""
Re-grade an existing mmlu_pro_slime samples_*.jsonl using the patched
3-tier extraction (strict 'the answer is (X)' → '\\boxed{X}' → bare A-J).

Updates each sample's `exact_match` field in place, plus writes two NEW
audit fields per sample:
  extraction_method  : 'strict' / 'boxed' / 'bare' / 'no_match'
  extracted_letter   : the A-J letter actually picked (or null)

Then patches the matching `results_<timestamp>.json` aggregate score.

Usage:
  python regrade_mmlu_pro_jsonl.py path/to/samples_mmlu_pro_slime_*.jsonl
  python regrade_mmlu_pro_jsonl.py path1.jsonl path2.jsonl --no-backup
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

# Use the patched utils (with boxed extraction tier)
TASKS_DIR = "/data1/hhzhang/slime/eval/slime_tasks"
sys.path.insert(0, TASKS_DIR)
import utils  # noqa: E402


def get_response(s):
    fr = s.get("filtered_resps")
    if isinstance(fr, list) and fr:
        x = fr[0]
        if isinstance(x, str): return x
        if isinstance(x, list) and x and isinstance(x[0], str): return x[0]
    return ""


def regrade_jsonl(path, backup):
    samples = [json.loads(l) for l in open(path)]
    n = len(samples)
    rescued = 0          # was wrong, now correct (boxed rescue)
    demoted = 0          # was correct, now wrong (rare)
    method_count = Counter()
    method_correct = Counter()
    n_correct_after = 0

    for s in samples:
        old = s.get("exact_match", 0.0)
        post_think = utils._strip_think(get_response(s))
        letter, method = utils.extract_mmlu_pro_answer(post_think)
        target = str(s.get("doc", {}).get("answer", "")).strip().upper()
        is_correct = bool(letter and target and letter == target)

        new = float(is_correct)
        if old != new:
            (rescued if new == 1.0 else demoted).__iadd__ if False else None
            if new == 1.0:
                rescued += 1
            else:
                demoted += 1
        if new == 1.0:
            n_correct_after += 1

        s["exact_match"] = new
        s["extraction_method"] = method
        s["extracted_letter"] = letter

        method_count[method] += 1
        if is_correct:
            method_correct[method] += 1

    if backup:
        backup_path = path.with_suffix(path.suffix + ".original")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"  [backup] {backup_path.name}")

    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    return dict(n=n, n_correct_after=n_correct_after,
                rescued=rescued, demoted=demoted,
                method_count=method_count, method_correct=method_correct)


def update_results_json(path, n_correct, n, backup):
    name = path.name
    if not name.startswith("samples_"):
        return None
    rest = name[len("samples_"):]
    parts = rest.rsplit("_", 1)
    if len(parts) != 2:
        return None
    task_name, ts_with_ext = parts
    ts = ts_with_ext.rsplit(".", 1)[0]
    results_json = path.parent / f"results_{ts}.json"
    if not results_json.exists():
        return None

    data = json.load(open(results_json))
    new_acc = n_correct / n
    if "results" in data and task_name in data["results"]:
        old_acc = data["results"][task_name].get("exact_match,none")
        data["results"][task_name]["exact_match,none"] = new_acc
        data["results"][task_name]["regraded_with_boxed_extraction"] = True
        if backup:
            bp = results_json.with_suffix(".json.original")
            if not bp.exists():
                shutil.copy2(results_json, bp)
        with open(results_json, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return dict(path=results_json, old=old_acc, new=new_acc)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="One or more samples_mmlu_pro_slime_*.jsonl paths.")
    p.add_argument("--no-backup", action="store_true",
                   help="Don't write *.original backup files.")
    args = p.parse_args()

    backup = not args.no_backup
    print(f"Using PATCHED utils from {TASKS_DIR}/utils.py "
          f"(MMLU-Pro 3-tier: strict → boxed → bare)\n")

    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            print(f"[skip] {path}", file=sys.stderr); continue
        print(f"=== {path.name} ===")
        st = regrade_jsonl(path, backup=backup)
        print(f"  total: {st['n']}    correct after regrade: "
              f"{st['n_correct_after']}/{st['n']} = {st['n_correct_after']/st['n']:.4f}")
        print(f"  rescued (was wrong, now correct): +{st['rescued']}")
        if st["demoted"]:
            print(f"  demoted (was correct, now wrong): -{st['demoted']}")
        print(f"  extraction-method breakdown:")
        for method in ("strict", "boxed", "bare", "no_match"):
            cnt = st["method_count"].get(method, 0)
            corr = st["method_correct"].get(method, 0)
            pct = cnt / st["n"] * 100 if st["n"] else 0
            acc = corr / cnt if cnt else 0
            print(f"    {method:10s} {cnt:5d} ({pct:5.1f}%)  "
                  f"of those, {corr:5d} correct ({acc:.4f})")

        upd = update_results_json(path, st["n_correct_after"], st["n"], backup=backup)
        if upd:
            print(f"  results json: {upd['path'].name}    "
                  f"{upd['old']:.4f}  →  {upd['new']:.4f}")
        else:
            print("  [warn] no matching results_*.json")
        print()


if __name__ == "__main__":
    main()
