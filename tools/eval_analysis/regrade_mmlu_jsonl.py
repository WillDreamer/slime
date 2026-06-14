#!/usr/bin/env python3
"""
Re-grade an existing mmlu_slime samples_*.jsonl using the patched extraction
(adds \\boxed{X} and "the answer is X" fallbacks beyond the original
"Answer: X" strict regex).

Each sample gets two NEW fields written into the jsonl for audit:
  extraction_method  — which pattern matched: 'answer_colon' / 'boxed' /
                       'the_answer_is' / 'no_match'
  extracted_letter   — the A-D letter the extractor picked (or null)

Plus the existing `exact_match` is updated, and the corresponding
results_<timestamp>.json's score is patched in place.

Usage:
  python regrade_mmlu_jsonl.py path/to/samples_mmlu_slime_<ts>.jsonl
  python regrade_mmlu_jsonl.py path/.../samples_mmlu_slime_*.jsonl --no-backup
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

# Use the patched utils (with broadened MMLU extraction)
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


def correct_letter_from_doc(doc) -> str | None:
    idx = doc.get("answer", -1)
    try:
        idx = int(idx)
    except (ValueError, TypeError):
        return None
    return ["A", "B", "C", "D"][idx] if 0 <= idx <= 3 else None


def regrade_jsonl(jsonl_path: Path, backup: bool):
    samples = [json.loads(line) for line in open(jsonl_path)]
    n = len(samples)

    rescued = 0       # was wrong, now correct
    demoted = 0       # was correct, now wrong (sanity check; should be ~0)
    method_count = {"answer_colon": 0, "boxed": 0, "the_answer_is": 0, "no_match": 0}
    method_correct = {"answer_colon": 0, "boxed": 0, "the_answer_is": 0, "no_match": 0}
    n_correct_after = 0

    for s in samples:
        old = s.get("exact_match", 0.0)
        post_think = utils._strip_think(get_response(s))
        letter, method = utils._extract_mmlu_answer(post_think)
        gt = correct_letter_from_doc(s.get("doc", {}))
        is_correct = bool(letter and gt and letter == gt)

        new = float(is_correct)
        if old != new:
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
        backup_path = jsonl_path.with_suffix(jsonl_path.suffix + ".original")
        if not backup_path.exists():
            shutil.copy2(jsonl_path, backup_path)
            print(f"  [backup] {backup_path.name}")

    with open(jsonl_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    return dict(n=n, n_correct_after=n_correct_after,
                rescued=rescued, demoted=demoted,
                method_count=method_count, method_correct=method_correct)


def update_results_json(jsonl_path: Path, n_correct: int, n: int, backup: bool):
    """Find sibling results_<timestamp>.json and patch its score."""
    name = jsonl_path.name
    if not name.startswith("samples_"):
        return None
    rest = name[len("samples_"):]
    parts = rest.rsplit("_", 1)
    if len(parts) != 2:
        return None
    task_name, ts_with_ext = parts
    ts = ts_with_ext.rsplit(".", 1)[0]
    results_json = jsonl_path.parent / f"results_{ts}.json"
    if not results_json.exists():
        return None

    data = json.load(open(results_json))
    new_acc = n_correct / n
    if "results" in data and task_name in data["results"]:
        old_acc = data["results"][task_name].get("exact_match,none")
        data["results"][task_name]["exact_match,none"] = new_acc
        data["results"][task_name]["regraded_with_broader_extraction"] = True

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
    p.add_argument("inputs", nargs="+",
                   help="One or more samples_mmlu_slime_<timestamp>.jsonl paths.")
    p.add_argument("--no-backup", action="store_true",
                   help="Don't write *.original backup files (default: keep originals).")
    args = p.parse_args()

    backup = not args.no_backup
    print(f"Using PATCHED utils from {TASKS_DIR}/utils.py "
          f"(MMLU broadened extraction with \\boxed/the-answer-is fallbacks)\n")

    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            print(f"[skip] {path} does not exist", file=sys.stderr)
            continue
        print(f"=== {path.name} ===")
        st = regrade_jsonl(path, backup=backup)
        print(f"  total: {st['n']}    correct after regrade: "
              f"{st['n_correct_after']}/{st['n']} = {st['n_correct_after']/st['n']:.4f}")
        print(f"  rescued (was wrong, now correct): +{st['rescued']}")
        if st["demoted"]:
            print(f"  demoted (was correct, now wrong): -{st['demoted']}")
        print(f"  extraction-method breakdown:")
        for method in ("answer_colon", "boxed", "the_answer_is", "no_match"):
            cnt = st["method_count"][method]
            corr = st["method_correct"][method]
            pct = cnt / st["n"] * 100 if st["n"] else 0
            acc = corr / cnt if cnt else 0
            print(f"    {method:18s} {cnt:5d} ({pct:5.1f}%)  "
                  f"of those, {corr:5d} correct ({acc:.4f})")

        upd = update_results_json(path, st["n_correct_after"], st["n"], backup=backup)
        if upd:
            print(f"  results json: {upd['path'].name}    "
                  f"{upd['old']:.4f}  →  {upd['new']:.4f}")
        else:
            print("  [warn] could not locate matching results_*.json (skipped score patch)")
        print()


if __name__ == "__main__":
    main()
