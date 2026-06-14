#!/usr/bin/env python3
"""
Re-grade an existing math500_slime / aime*_slime samples_*.jsonl using the
PATCHED utils.py (with sympy fallback) — equivalent to re-running lm-harness
on the same model responses, but instant.

Updates each sample's `exact_match` field, recomputes the aggregate accuracy,
and updates the matching `results_*.json` file's score.

Usage:
  python regrade_math_jsonl.py path/to/samples_math500_slime_<timestamp>.jsonl

  # Multiple files at once
  python regrade_math_jsonl.py path/to/*/samples_math500_slime_*.jsonl

  # Don't write a .original backup (default: backup is made)
  python regrade_math_jsonl.py file.jsonl --no-backup
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

# Use the patched utils with sympy fallback
TASKS_DIR = "/data1/hhzhang/slime/eval/slime_tasks"
sys.path.insert(0, TASKS_DIR)
import utils  # noqa: E402


def get_response(sample: dict) -> str:
    fr = sample.get("filtered_resps")
    if isinstance(fr, list) and fr:
        x = fr[0]
        if isinstance(x, str):
            return x
        if isinstance(x, list) and x and isinstance(x[0], str):
            return x[0]
    return ""


def regrade_jsonl(jsonl_path: Path, backup: bool):
    samples = [json.loads(line) for line in open(jsonl_path)]

    n = len(samples)
    n_changed_to_correct = 0
    n_changed_to_wrong = 0
    n_correct_after = 0

    for s in samples:
        old = s.get("exact_match", 0.0)
        response = get_response(s)
        new_metric = utils.process_results_math(s["doc"], [response])
        new = new_metric["exact_match"]
        s["exact_match"] = new
        # also update the metrics list if present
        if isinstance(s.get("metrics"), list):
            s["metrics"] = ["exact_match"]  # keep simple; harness writes ['exact_match']
        if old != new:
            if new == 1.0:
                n_changed_to_correct += 1
            else:
                n_changed_to_wrong += 1
        if new == 1.0:
            n_correct_after += 1

    if backup:
        backup_path = jsonl_path.with_suffix(jsonl_path.suffix + ".original")
        if not backup_path.exists():
            shutil.copy2(jsonl_path, backup_path)
            print(f"  [backup] {backup_path.name}")

    with open(jsonl_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    return dict(n=n, n_correct_after=n_correct_after,
                rescued=n_changed_to_correct, demoted=n_changed_to_wrong)


def update_results_json(jsonl_path: Path, new_correct: int, n: int, backup: bool):
    """Find the corresponding results_<timestamp>.json and patch its accuracy field."""
    # samples_<task>_<timestamp>.jsonl  →  results_<timestamp>.json
    name = jsonl_path.name
    # strip 'samples_' prefix and '_<task>_' middle to get '<timestamp>.jsonl'
    prefix = "samples_"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    # rest looks like: math500_slime_2026-04-09T15-48-35.906420.jsonl
    # find the last underscore before the timestamp (timestamps start with 4 digits)
    parts = rest.rsplit("_", 1)
    if len(parts) != 2:
        return None
    task_name, ts_with_ext = parts
    ts = ts_with_ext.rsplit(".", 1)[0]  # remove .jsonl
    results_json = jsonl_path.parent / f"results_{ts}.json"
    if not results_json.exists():
        return None

    data = json.load(open(results_json))
    new_acc = new_correct / n
    # update the score in results.results.<task>.exact_match,none
    if "results" in data and task_name in data["results"]:
        old_acc = data["results"][task_name].get("exact_match,none")
        data["results"][task_name]["exact_match,none"] = new_acc
        # bootstrap stderr is hard to recompute exactly without resampling;
        # leave the original stderr but mark the result as regraded.
        data["results"][task_name]["regraded_with_sympy"] = True

        if backup:
            backup_path = results_json.with_suffix(".json.original")
            if not backup_path.exists():
                shutil.copy2(results_json, backup_path)

        with open(results_json, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return dict(path=results_json, old=old_acc, new=new_acc)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="One or more samples_<task>_<timestamp>.jsonl paths.")
    p.add_argument("--no-backup", action="store_true",
                   help="Don't write *.original backup files (default: keep originals).")
    args = p.parse_args()

    backup = not args.no_backup
    print(f"Using PATCHED utils from {TASKS_DIR}/utils.py (sympy fallback enabled)\n")

    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            print(f"[skip] {path} does not exist", file=sys.stderr)
            continue
        print(f"=== {path.name} ===")
        st = regrade_jsonl(path, backup=backup)
        print(f"  total: {st['n']}    correct after regrade: {st['n_correct_after']}/{st['n']} = {st['n_correct_after']/st['n']:.4f}")
        print(f"  rescued (was wrong, now correct): +{st['rescued']}")
        if st['demoted']:
            print(f"  demoted (was correct, now wrong): -{st['demoted']}")
        upd = update_results_json(path, st["n_correct_after"], st["n"], backup=backup)
        if upd:
            print(f"  results json:  {upd['path'].name}    {upd['old']:.4f}  →  {upd['new']:.4f}")
        else:
            print("  [warn] could not locate matching results_*.json (skipped score patch)")
        print()


if __name__ == "__main__":
    main()
