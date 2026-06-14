#!/usr/bin/env python3
"""
Audit a mmlu_pro_slime samples_*.jsonl for hidden corrects, hidden failures,
and the WHY behind every wrong answer. Same three-layer taxonomy as
classify_mmlu_failures.py, adapted for MMLU-Pro (10-way A-J multiple choice
with two-tier extraction: strict 'the answer is (X)' + bare-letter fallback).

Three layers:

  Layer 1 — extraction path (mirrors slime_tasks/utils.extract_mmlu_pro_answer)
    strict_the_answer_is : 'the answer is (X)' matched
    boxed                : '\\boxed{X}' matched (math-RL format leak)
    bare_letter_fallback : neither strict nor boxed matched; last \\b[A-J]\\b wins
    no_letter            : nothing matched (real format failure)

  Layer 2 — for `no_letter` samples, try harder:
    hidden_correct_letter : aggressive pattern finds correct letter in tail
    hidden_failure        : aggressive pattern finds a wrong letter (committed)
    no_letter_found       : no letter present at all

  Layer 3 — among samples that are STILL wrong, why?
    extracted_wrong       : letter was extracted but is wrong (real MCQ error)
    fallback_wrong        : strict missed → fell back → fallback was wrong
                           (more suspicious than extracted_wrong; the model
                            may have been close but chose a different letter)
    hidden_failure        : (carried up from layer 2)
    no_letter:
      reach_max_function_call / python_block_unfinished / repetition_loop /
      truncated_max_tokens / gave_up_short / medium_unconverged

Usage:
  python classify_mmlu_pro_failures.py samples_mmlu_pro_slime_*.jsonl
  python classify_mmlu_pro_failures.py base=…jsonl base_math=…jsonl final_search=…jsonl
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------- response & target plumbing ----------

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
    a = doc.get("answer")
    return str(a).strip().upper() if a else None


# ---------- live extraction (mirrors slime_tasks/utils.process_results_mmlu_pro) ----------

_STRICT_RE = re.compile(r"(?i)the\s+answer\s+is\s+\(?([A-J])\)?")
_BOXED_RE = re.compile(r"\\boxed\{\(?([A-J])\)?\}")
_BARE_RE = re.compile(r"\b([A-J])\b")


def extract_letter(post_think: str) -> tuple[str | None, str]:
    """Mirror slime_tasks/utils.extract_mmlu_pro_answer. Returns (letter, method).
    Method is 'strict_the_answer_is' / 'boxed' / 'bare_letter_fallback' / 'no_letter'."""
    m = _STRICT_RE.findall(post_think)
    if m:
        return m[-1].upper(), "strict_the_answer_is"
    m = _BOXED_RE.findall(post_think)
    if m:
        return m[-1].upper(), "boxed"
    cands = _BARE_RE.findall(post_think.upper())
    if cands:
        return cands[-1], "bare_letter_fallback"
    return None, "no_letter"


# ---------- aggressive recovery for no_letter samples ----------

# Looser patterns than _BARE_RE that look for letter inside parens / after "answer:" / etc.
# Used when bare-letter fallback ALSO fails (i.e. text has zero A-J alpha char).
_AGGRESSIVE_PATTERNS = [
    (re.compile(r"(?i)\b(?:correct|right)\s+(?:option|choice|answer)\s+is\s+\(?([A-J])\)?"), "correct_X_is"),
    (re.compile(r"(?i)\bchoose\s+\(?([A-J])\)?\b"),                                            "choose_X"),
    (re.compile(r"(?i)\b(?:option|choice)\s*[:.]?\s*\(?([A-J])\)?\b"),                         "option_X"),
    (re.compile(r"\(([A-J])\)\s*\.?\s*$", re.MULTILINE),                                       "paren_X_end"),
    (re.compile(r"^\s*\(?([A-J])\)?\s*\.?\s*$", re.MULTILINE),                                 "bare_X_line"),
]


def aggressive_letter(text: str, tail_chars: int = 800) -> tuple[str | None, str | None]:
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    best_pos, best_letter, best_pattern = -1, None, None
    for rx, name in _AGGRESSIVE_PATTERNS:
        for m in rx.finditer(tail):
            if m.start() > best_pos:
                best_pos = m.start()
                best_letter = m.group(1).upper()
                best_pattern = name
    return best_letter, best_pattern


# ---------- failure subtype (genuine no_letter wrongs) ----------

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
    if PYTHON_FENCE_RE.search(text[-2000:]):
        return True
    return False


def classify_no_letter_subtype(text, truncation_thresh):
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
    target = correct_letter(s.get("doc", {}))
    txt_full = get_response(s)
    txt = strip_think(txt_full)
    extracted, method = extract_letter(txt)

    layer1 = method
    hidden_correct, hidden_failure = False, False
    aggr_letter, aggr_pattern = None, None

    if method == "no_letter":
        aggr_letter, aggr_pattern = aggressive_letter(txt)
        if aggr_letter and target:
            if aggr_letter == target:
                hidden_correct = True
            else:
                hidden_failure = True

    # Final correctness uses live extraction (matches what extractor would produce
    # if you re-graded today). For 'no_letter', allow Layer-2 rescue.
    if method == "no_letter":
        final_correct = hidden_correct
    else:
        final_correct = (extracted == target)

    subtype = None
    if not final_correct:
        if method == "no_letter":
            if hidden_failure:
                subtype = "hidden_failure"
            else:
                subtype = classify_no_letter_subtype(txt_full, truncation_thresh)
        elif method == "bare_letter_fallback":
            subtype = "fallback_wrong"
        elif method == "boxed":
            subtype = "boxed_wrong"
        else:
            subtype = "extracted_wrong"

    return dict(
        doc_id=s.get("doc_id"),
        category=s.get("doc", {}).get("category"),
        layer1=layer1,
        extracted=extracted,
        target=target,
        aggr_letter=aggr_letter,
        aggr_pattern=aggr_pattern,
        hidden_correct=hidden_correct,
        hidden_failure=hidden_failure,
        final_correct=final_correct,
        subtype=subtype,
        resp_len=len(txt_full),
        original_exact_match=s.get("exact_match"),
    )


def analyze(path, name, truncation_thresh):
    samples = [json.loads(l) for l in open(path)]
    audits = [audit_sample(s, truncation_thresh) for s in samples]

    layer1 = Counter(a["layer1"] for a in audits)
    nm = [a for a in audits if a["layer1"] == "no_letter"]
    hidden_correct = sum(1 for a in nm if a["hidden_correct"])
    hidden_failure = sum(1 for a in nm if a["hidden_failure"])
    no_letter_genuine = len(nm) - hidden_correct - hidden_failure

    subtypes = Counter(a["subtype"] for a in audits if a["subtype"])
    n = len(samples)
    n_orig_correct = sum(1 for s in samples if s.get("exact_match") == 1.0)
    n_relive = sum(1 for a in audits if a["final_correct"])

    return dict(
        name=name, path=str(path), n=n, audits=audits,
        layer1=layer1,
        hidden_correct=hidden_correct,
        hidden_failure=hidden_failure,
        no_letter_genuine=no_letter_genuine,
        subtypes=subtypes,
        n_orig_correct=n_orig_correct,
        n_relive=n_relive,
    )


# ---------- printing ----------

def _pct(v, n):
    return f"{v/n*100:5.1f}%" if n else "  N/A"


SUBTYPE_ORDER = [
    "extracted_wrong",
    "boxed_wrong",
    "fallback_wrong",
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
    print(f"    original exact_match (lm-eval)  : {r['n_orig_correct']}/{n} = {r['n_orig_correct']/n:.4f}")
    print(f"    re-live with current extractor  : {r['n_relive']}/{n} = {r['n_relive']/n:.4f}")

    print(f"\n  Layer 1 — extraction path:")
    for k in ("strict_the_answer_is", "boxed", "bare_letter_fallback", "no_letter"):
        v = r["layer1"].get(k, 0)
        print(f"    {k:24s} {v:5d}  ({_pct(v, n)})")

    nm = r["layer1"].get("no_letter", 0)
    print(f"\n  Layer 2 — within no_letter ({nm} samples):")
    print(f"    hidden_correct (rescued)     {r['hidden_correct']:4d}  ({_pct(r['hidden_correct'], nm)} of no_letter)")
    print(f"    hidden_failure (committed wrong) {r['hidden_failure']:4d}  ({_pct(r['hidden_failure'], nm)} of no_letter)")
    print(f"    no letter found              {r['no_letter_genuine']:4d}  ({_pct(r['no_letter_genuine'], nm)} of no_letter)")

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
    print(f"{'metric':32s} " + " ".join(f"{r['name']:>14s}" for r in results))
    print("-" * 80)

    def row(label, fmt):
        print(f"{label:32s} " + " ".join(fmt(r) for r in results))

    row("acc (original exact_match)",  lambda r: f"{r['n_orig_correct']/r['n']:>14.4f}")
    row("acc (re-live extractor)",     lambda r: f"{r['n_relive']/r['n']:>14.4f}")
    row("  hidden_correct rescue",     lambda r: f"{r['hidden_correct']:>14d}")
    row("  hidden_failure",            lambda r: f"{r['hidden_failure']:>14d}")
    row("  strict_the_answer_is used", lambda r: f"{r['layer1'].get('strict_the_answer_is', 0):>14d}")
    row("  boxed used",                lambda r: f"{r['layer1'].get('boxed', 0):>14d}")
    row("  bare_letter_fallback used", lambda r: f"{r['layer1'].get('bare_letter_fallback', 0):>14d}")
    row("  no_letter total",           lambda r: f"{r['layer1'].get('no_letter', 0):>14d}")

    print(f"\n{'failure subtypes':32s} " + " ".join(f"{r['name']:>14s}" for r in results))
    for st in SUBTYPE_ORDER:
        cells = " ".join(f"{r['subtypes'].get(st, 0):>14d}" for r in results)
        print(f"  {st:30s} {cells}")


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
    p.add_argument("--truncation-thresh", type=int, default=30000)
    args = p.parse_args()

    results = []
    for inp in args.inputs:
        name, path = parse_input(inp)
        if not path.exists():
            print(f"[skip] {path}", file=sys.stderr); continue
        r = analyze(path, name, args.truncation_thresh)
        results.append(r)
        print_summary(r)

    if len(results) > 1:
        print_compare(results)


if __name__ == "__main__":
    main()
