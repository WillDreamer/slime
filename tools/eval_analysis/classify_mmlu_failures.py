#!/usr/bin/env python3
"""
Audit a mmlu_slime samples_*.jsonl for hidden corrects, hidden failures, and
the WHY behind every wrong answer. Same three-layer taxonomy as
classify_math_failures.py, adapted for multiple-choice (A/B/C/D).

Three layers:

  Layer 1 — extraction verdict
    answer_colon / boxed / the_answer_is  : extractor found a letter
    no_match                               : nothing matched (the audit pile)

  Layer 2 — for no_match samples, try harder:
    hidden_correct_letter  : aggressive pattern finds correct letter in tail
    hidden_failure         : aggressive pattern finds a wrong letter (committed)
    no_letter_found        : no letter present anywhere identifiable

  Layer 3 — among samples that are STILL wrong after layers 1+2, why?
    extracted_wrong       : letter was extracted but is wrong (genuine MCQ error)
    hidden_failure        : carried up from layer 2
    no_letter:
      reach_max_function_call : 'Reach max function call limit' marker
      python_block_unfinished : stuck in ```python``` block
      repetition_loop         : tail is a repeating substring
      truncated_max_tokens    : long, no looping pattern, just cut
      gave_up_short           : short response that trails off
      medium_unconverged      : medium length, never picks a letter

Usage:
  python classify_mmlu_failures.py samples_mmlu_slime_*.jsonl
  python classify_mmlu_failures.py base=…jsonl base_math=…jsonl final_search=…jsonl
  python classify_mmlu_failures.py samples.jsonl --dump-dir audit/
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------- response / letter plumbing ----------

def get_response(s) -> str:
    fr = s.get("filtered_resps")
    if isinstance(fr, list) and fr:
        x = fr[0]
        if isinstance(x, str): return x
        if isinstance(x, list) and x and isinstance(x[0], str): return x[0]
    return ""


def strip_think(t):
    return t.split("</think>")[-1] if "</think>" in t else t


def correct_letter(doc) -> str | None:
    idx = doc.get("answer", -1)
    try:
        idx = int(idx)
    except (ValueError, TypeError):
        return None
    return ["A", "B", "C", "D"][idx] if 0 <= idx <= 3 else None


# ---------- aggressive letter extraction for no_match samples ----------

# Patterns more permissive than the regrade script's set. Each (regex, name).
# Listed roughly most→least specific. Capture group must contain a single A-D letter.
_AGGRESSIVE_PATTERNS = [
    (re.compile(r"(?i)\b(?:correct|right)\s+(?:option|choice|answer)\s+is\s+\(?([A-D])\)?"), "correct_X_is"),
    (re.compile(r"(?i)\b(?:option|choice)\s+\(?([A-D])\)?\s+is\s+(?:correct|right)"),         "X_is_correct"),
    (re.compile(r"(?i)\bchoose\s+\(?([A-D])\)?\b"),                                            "choose_X"),
    (re.compile(r"(?i)\bgo\s+with\s+\(?([A-D])\)?\b"),                                         "go_with_X"),
    (re.compile(r"(?i)\b(?:option|choice|answer)\s*[:.]?\s*\(?([A-D])\)?\b"),                  "option_X"),
    (re.compile(r"\(([A-D])\)\s*\.?\s*$", re.MULTILINE),                                       "paren_X_end"),
    (re.compile(r"\b([A-D])\)\s*$", re.MULTILINE),                                             "X_paren_end"),
    (re.compile(r"^\s*([A-D])\s*\.?\s*$", re.MULTILINE),                                       "bare_X"),
]


def aggressive_letter(text: str, tail_chars: int = 800) -> tuple[str | None, str | None]:
    """Return (letter, pattern_name) using the LAST match across all patterns."""
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    best_pos = -1
    best_letter = None
    best_pattern = None
    for rx, name in _AGGRESSIVE_PATTERNS:
        for m in rx.finditer(tail):
            if m.start() > best_pos:
                best_pos = m.start()
                best_letter = m.group(1).upper()
                best_pattern = name
    return best_letter, best_pattern


# ---------- failure subtype detection (for genuine no_match wrongs) ----------

REACH_MAX_RE = re.compile(r"Reach\s+max\s+function\s+call\s+limit", re.IGNORECASE)
PYTHON_FENCE_RE = re.compile(r"```(?:python|output|tool_call|sympy)", re.IGNORECASE)


def has_repetition_loop(text, tail_chars=4000, window=60, min_repeats=8):
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    if len(tail) < window * min_repeats:
        return False
    step = max(1, window // 3)
    seen = Counter()
    for i in range(0, len(tail) - window, step):
        seen[tail[i:i + window]] += 1
    if not seen:
        return False
    most_common, count = seen.most_common(1)[0]
    if count < min_repeats:
        return False
    if most_common.strip() == "" or len(most_common.strip()) < window * 0.3:
        return False
    return True


def python_block_unfinished(text):
    if not PYTHON_FENCE_RE.search(text):
        return False
    if len(re.findall(r"```", text)) % 2 == 1:
        return True
    tail = text[-2000:]
    if PYTHON_FENCE_RE.search(tail):
        return True
    return False


def classify_no_match_subtype(text, truncation_thresh):
    if REACH_MAX_RE.search(text):
        return "reach_max_function_call"
    if python_block_unfinished(text):
        return "python_block_unfinished"
    if has_repetition_loop(text):
        return "repetition_loop"
    n = len(text)
    if n > truncation_thresh:
        return "truncated_max_tokens"
    if n < 1500:
        return "gave_up_short"
    return "medium_unconverged"


# ---------- per-sample audit ----------

def audit_sample(s, truncation_thresh):
    method = s.get("extraction_method", "?")
    extracted = s.get("extracted_letter")
    target_letter = correct_letter(s.get("doc", {}))
    txt_full = get_response(s)
    txt = strip_think(txt_full)

    layer1 = method  # answer_colon / boxed / the_answer_is / no_match

    # Layer 2 — only for no_match
    hidden_correct = False
    hidden_failure = False
    aggr_letter = None
    aggr_pattern = None
    if method == "no_match":
        aggr_letter, aggr_pattern = aggressive_letter(txt)
        if aggr_letter and target_letter:
            if aggr_letter == target_letter:
                hidden_correct = True
            else:
                hidden_failure = True

    # Final correctness
    if method != "no_match":
        final_correct = (extracted == target_letter)
    else:
        final_correct = hidden_correct  # rescued via aggressive extraction

    # Layer 3 subtype
    subtype = None
    if not final_correct:
        if method == "no_match":
            if hidden_failure:
                subtype = "hidden_failure"
            else:
                subtype = classify_no_match_subtype(txt_full, truncation_thresh)
        else:
            subtype = "extracted_wrong"

    return dict(
        doc_id=s.get("doc_id"),
        subject=s.get("doc", {}).get("subject"),
        layer1=layer1,
        extracted=extracted,
        target=target_letter,
        aggr_letter=aggr_letter,
        aggr_pattern=aggr_pattern,
        hidden_correct=hidden_correct,
        hidden_failure=hidden_failure,
        final_correct=final_correct,
        subtype=subtype,
        resp_len=len(txt_full),
    )


# ---------- analysis ----------

def analyze(path, name, truncation_thresh):
    samples = [json.loads(l) for l in open(path)]
    audits = [audit_sample(s, truncation_thresh) for s in samples]

    layer1 = Counter(a["layer1"] for a in audits)
    nm = [a for a in audits if a["layer1"] == "no_match"]
    hidden_correct = sum(1 for a in nm if a["hidden_correct"])
    hidden_failure = sum(1 for a in nm if a["hidden_failure"])
    no_letter = len(nm) - hidden_correct - hidden_failure

    subtypes = Counter(a["subtype"] for a in audits if a["subtype"])
    n = len(samples)
    n_orig_correct = sum(1 for s in samples if s.get("exact_match") == 1.0)
    n_after_rescue = n_orig_correct + hidden_correct

    return dict(
        name=name, path=str(path), n=n, audits=audits,
        layer1=layer1,
        hidden_correct=hidden_correct,
        hidden_failure=hidden_failure,
        no_letter=no_letter,
        subtypes=subtypes,
        n_orig_correct=n_orig_correct,
        n_after_rescue=n_after_rescue,
    )


# ---------- printing ----------

def _pct(v, n):
    return f"{v/n*100:5.1f}%" if n else "  N/A"


SUBTYPE_ORDER = [
    "extracted_wrong",
    "hidden_failure",
    "reach_max_function_call",
    "python_block_unfinished",
    "repetition_loop",
    "truncated_max_tokens",
    "medium_unconverged",
    "gave_up_short",
]


def print_summary(r):
    n = r["n"]
    print(f"\n=== {r['name']} ===")
    print(f"  file: {r['path']}")
    print(f"  total: {n}")
    print(f"\n  Accuracy variants:")
    print(f"    extractor original             : {r['n_orig_correct']}/{n} = {r['n_orig_correct']/n:.4f}")
    print(f"    + aggressive letter rescue     : {r['n_after_rescue']}/{n} = {r['n_after_rescue']/n:.4f}  "
          f"(+{r['hidden_correct']})")

    print(f"\n  Layer 1 — extraction method distribution:")
    for k in ("answer_colon", "the_answer_is", "boxed", "no_match"):
        v = r["layer1"].get(k, 0)
        print(f"    {k:18s} {v:5d}  ({_pct(v, n)})")

    nm = r["layer1"].get("no_match", 0)
    print(f"\n  Layer 2 — within no_match ({nm} samples):")
    print(f"    hidden_correct_letter   {r['hidden_correct']:4d}  ({_pct(r['hidden_correct'], nm)} of no_match)  ← rescued")
    print(f"    hidden_failure (wrong)  {r['hidden_failure']:4d}  ({_pct(r['hidden_failure'], nm)} of no_match)")
    print(f"    no letter found        {r['no_letter']:4d}  ({_pct(r['no_letter'], nm)} of no_match)")

    print(f"\n  Layer 3 — surviving wrongs by subtype:")
    n_wrong = sum(r["subtypes"].values())
    for st in SUBTYPE_ORDER:
        v = r["subtypes"].get(st, 0)
        if v == 0:
            continue
        print(f"    {st:28s} {v:5d}  ({_pct(v, n_wrong)} of wrongs)")


def print_compare(results):
    if len(results) < 2:
        return
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON")
    print("=" * 80)
    header = f"{'metric':32s} " + " ".join(f"{r['name']:>14s}" for r in results)
    print(header)
    print("-" * len(header))

    def row(label, fmt):
        print(f"{label:32s} " + " ".join(fmt(r) for r in results))

    row("acc (extractor)",        lambda r: f"{r['n_orig_correct']/r['n']:>14.4f}")
    row("acc (+ aggressive)",     lambda r: f"{r['n_after_rescue']/r['n']:>14.4f}")
    row("  hidden_correct rescue",lambda r: f"{r['hidden_correct']:>14d}")
    row("  hidden_failure",       lambda r: f"{r['hidden_failure']:>14d}")
    row("  no_match total",       lambda r: f"{r['layer1'].get('no_match',0):>14d}")

    print(f"\n{'no_match subtypes':32s} " + " ".join(f"{r['name']:>14s}" for r in results))
    for st in SUBTYPE_ORDER:
        if st == "extracted_wrong":   # not a no_match thing
            continue
        cells = " ".join(f"{r['subtypes'].get(st, 0):>14d}" for r in results)
        print(f"  {st:30s} {cells}")


# ---------- example dumps ----------

def dump_examples(r, out_dir, max_chars):
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = defaultdict(list)
    for a in r["audits"]:
        if a["hidden_correct"]:
            buckets["hidden_correct_letter"].append(a)
        if a["hidden_failure"]:
            buckets["hidden_failure"].append(a)
        if a["subtype"] and a["subtype"] not in {"extracted_wrong", "hidden_failure"}:
            buckets[f"no_letter_{a['subtype']}"].append(a)

    samples_by_id = {s["doc_id"]: s for s in
                     (json.loads(l) for l in open(r["path"]))}
    for bucket, audits in buckets.items():
        if not audits:
            continue
        f_path = out_dir / f"{r['name']}__{bucket}.txt"
        # cap dump at 50 per bucket so the audit folder doesn't explode
        if len(audits) > 50:
            audits = audits[:50]
        with open(f_path, "w") as f:
            f.write(f"# {r['name']} — bucket: {bucket}\n\n")
            for a in audits:
                s = samples_by_id.get(a["doc_id"], {})
                doc = s.get("doc", {})
                txt = get_response(s)
                f.write("=" * 80 + "\n")
                f.write(f"doc_id={a['doc_id']}  subject={a['subject']}  "
                        f"target={a['target']}  resp_len={len(txt)}\n")
                f.write(f"layer1={a['layer1']}  subtype={a['subtype']}\n")
                if a["aggr_letter"]:
                    f.write(f"aggressive: letter={a['aggr_letter']} "
                            f"pattern={a['aggr_pattern']}\n")
                f.write(f"\n--- question ---\n{doc.get('question', '')}\n")
                f.write(f"\nchoices: {doc.get('choices')}\n")
                if len(txt) > max_chars:
                    f.write(f"\n--- response (head, {max_chars}/{len(txt)} chars) ---\n")
                    f.write(txt[:max_chars])
                    f.write(f"\n\n--- response (tail, last 1500 chars) ---\n")
                    f.write(txt[-1500:])
                else:
                    f.write(f"\n--- response ({len(txt)} chars) ---\n")
                    f.write(txt)
                f.write("\n\n")
        print(f"  [dump] {f_path.name}  ({len(audits)} samples)")


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
                   help="Labeled samples_mmlu_slime_*.jsonl paths.")
    p.add_argument("--truncation-thresh", type=int, default=30000,
                   help="Char count above which a no_match response is treated as truncated.")
    p.add_argument("--dump-dir", default=None)
    p.add_argument("--max-chars", type=int, default=4000)
    args = p.parse_args()

    results = []
    for inp in args.inputs:
        name, path = parse_input(inp)
        if not path.exists():
            print(f"[skip] {path} does not exist", file=sys.stderr); continue
        r = analyze(path, name, args.truncation_thresh)
        results.append(r)
        print_summary(r)

    if len(results) > 1:
        print_compare(results)

    if args.dump_dir:
        out = Path(args.dump_dir)
        print(f"\ndumping examples to {out}/ ...")
        for r in results:
            dump_examples(r, out, args.max_chars)


if __name__ == "__main__":
    main()
