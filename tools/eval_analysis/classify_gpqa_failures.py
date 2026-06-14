#!/usr/bin/env python3
"""
Audit a gpqa_slime samples_*.jsonl for hidden corrects, hidden failures, and
the WHY behind every wrong answer. Mirrors classify_mmlu_pro_failures.py but
adapted for GPQA's more elaborate extractor.

GPQA's current extractor (eval/slime_tasks/utils.py::_extract_letter_from_response)
runs five passes in order:
  1. pattern_answer    : (answer|option|choice) (is|:) X
  2. pattern_X_correct : X is correct
  3. pattern_final     : final (answer|option) (is|:) X
  4. boxed             : \\boxed{X}                  ← promoted to a first-class
                                                      path so the bucket
                                                      accounting is honest
  5. bare_letter       : last \\b[A-Z]\\b in valid_letters set
                         (only fires when even the box is absent)

Three layers (same as other classify scripts):

  Layer 1 — extraction path
    pattern_answer     : matched first regex
    pattern_X_correct  : matched second regex
    pattern_final      : matched third regex
    boxed              : \\boxed{X} matched
    bare_letter        : fell through to bare-letter fallback
    no_match           : no letter found at all

  Layer 2 — no_match audit:
    hidden_correct_letter : aggressive scan finds correct letter in tail
    hidden_failure        : aggressive scan finds a wrong letter committed
    no_letter_found       : truly nothing

  Layer 3 — surviving wrongs by subtype:
    extracted_wrong       : a strict pattern matched but the letter was wrong
    bare_wrong            : bare-letter fallback was wrong (sketchier signal)
    boxed_disagreement    : \\boxed{X} exists in tail but extractor returned a
                            DIFFERENT letter (potential bug worth investigating)
    hidden_failure        : carried up from layer 2
    no_letter:
      reach_max_function_call / python_block_unfinished / repetition_loop /
      truncated_max_tokens / gave_up_short / medium_unconverged
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


VALID_LETTERS = {"A", "B", "C", "D"}


def get_response(s):
    fr = s.get("filtered_resps")
    if isinstance(fr, list) and fr:
        x = fr[0]
        if isinstance(x, str): return x
        if isinstance(x, list) and x and isinstance(x[0], str): return x[0]
    return ""


def strip_think(t):
    return t.split("</think>")[-1] if "</think>" in t else t


def correct_letter(doc):
    raw = str(doc.get("answer", "")).upper()
    m = re.search(r"([A-D])", raw)
    return m.group(1) if m else None


# Mirror of utils._extract_letter_from_response, but returns the path name too.
_PATTERNS = [
    (re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])", re.IGNORECASE), "pattern_answer"),
    (re.compile(r"([A-Z])\s*(?:is\s*(?:the)?\s*correct)", re.IGNORECASE),           "pattern_X_correct"),
    (re.compile(r"final\s*(?:answer|option)\s*(?:is|:)?\s*([A-Z])", re.IGNORECASE), "pattern_final"),
]
_BARE_RE = re.compile(r"\b([A-Z])\b")
_BOXED_RE = re.compile(r"\\boxed\{\(?([A-Z])\)?\}")


def extract_letter(response: str) -> tuple[str | None, str]:
    """Faithful mirror of utils._extract_letter_from_response, returning the path.
    Order: 3 strict patterns → \\boxed{X} → bare-letter → no_match."""
    text = strip_think(response or "")
    for rx, name in _PATTERNS:
        m = rx.search(text)
        if m:
            letter = m.group(1).upper()
            if letter in VALID_LETTERS:
                return letter, name
    boxed = _BOXED_RE.findall(text)
    for letter in reversed(boxed):
        if letter.upper() in VALID_LETTERS:
            return letter.upper(), "boxed"
    cands = _BARE_RE.findall(text)
    for letter in reversed(cands):
        if letter.upper() in VALID_LETTERS:
            return letter.upper(), "bare_letter"
    return None, "no_match"


def boxed_letter(text: str) -> str | None:
    """Last \\boxed{X} where X in A-D. Tells us what a boxed-aware grader would pick."""
    matches = _BOXED_RE.findall(text)
    for m in reversed(matches):
        if m.upper() in VALID_LETTERS:
            return m.upper()
    return None


# Aggressive recovery for no_match samples
_AGGRESSIVE_PATTERNS = [
    (re.compile(r"(?i)\b(?:correct|right)\s+(?:option|choice|answer)\s+is\s+\(?([A-D])\)?"), "correct_X_is"),
    (re.compile(r"(?i)\bchoose\s+\(?([A-D])\)?\b"),                                            "choose_X"),
    (re.compile(r"\\boxed\{\(?([A-D])\)?\}"),                                                  "boxed"),
    (re.compile(r"\(([A-D])\)\s*\.?\s*$", re.MULTILINE),                                       "paren_X_end"),
]


def aggressive_letter(text):
    tail = text[-1000:] if len(text) > 1000 else text
    best_pos, best_letter, best_pat = -1, None, None
    for rx, name in _AGGRESSIVE_PATTERNS:
        for m in rx.finditer(tail):
            if m.start() > best_pos:
                best_pos, best_letter, best_pat = m.start(), m.group(1).upper(), name
    return best_letter, best_pat


# Failure subtypes for genuine no_match
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


def audit_sample(s, truncation_thresh):
    target = correct_letter(s.get("doc", {}))
    txt_full = get_response(s)
    txt = strip_think(txt_full)
    extracted, method = extract_letter(txt_full)
    boxed = boxed_letter(txt)

    layer1 = method
    hidden_correct, hidden_failure = False, False
    aggr_letter, aggr_pattern = None, None

    if method == "no_match":
        aggr_letter, aggr_pattern = aggressive_letter(txt)
        if aggr_letter and target:
            if aggr_letter == target:
                hidden_correct = True
            else:
                hidden_failure = True

    final_correct = (extracted == target) if extracted else hidden_correct

    # via_boxed: did the extracted letter come from a \\boxed{} in the response?
    # True iff there's a \\boxed{X} where X equals the extractor's chosen letter.
    via_boxed = bool(boxed and extracted and boxed == extracted)

    # Boxed disagreement: a \\boxed{X} exists in the tail but extractor returned
    # a different letter from a strict pattern. This is the "fixable bug" zone.
    boxed_disagreement = bool(
        boxed and extracted and boxed != extracted and method != "bare_letter"
    )

    subtype = None
    if not final_correct:
        if method == "no_match":
            subtype = "hidden_failure" if hidden_failure else classify_no_match_subtype(txt_full, truncation_thresh)
        elif boxed_disagreement and boxed == target:
            # extractor said wrong letter, but boxed says correct letter ←
            # would be rescued by adding boxed extraction
            subtype = "rescuable_via_boxed"
        elif method == "bare_letter":
            subtype = "bare_wrong"
        elif method == "boxed":
            subtype = "boxed_wrong"
        else:
            subtype = "extracted_wrong"

    return dict(
        doc_id=s.get("doc_id"),
        domain=s.get("doc", {}).get("High-level domain"),
        subdomain=s.get("doc", {}).get("Subdomain"),
        layer1=layer1,
        extracted=extracted,
        boxed=boxed,
        target=target,
        via_boxed=via_boxed,
        boxed_disagreement=boxed_disagreement,
        aggr_letter=aggr_letter,
        aggr_pattern=aggr_pattern,
        hidden_correct=hidden_correct,
        hidden_failure=hidden_failure,
        final_correct=final_correct,
        subtype=subtype,
        resp_len=len(txt_full),
    )


def analyze(path, name, truncation_thresh):
    samples = [json.loads(l) for l in open(path)]
    audits = [audit_sample(s, truncation_thresh) for s in samples]

    layer1 = Counter(a["layer1"] for a in audits)
    nm = [a for a in audits if a["layer1"] == "no_match"]
    hidden_correct = sum(1 for a in nm if a["hidden_correct"])
    hidden_failure = sum(1 for a in nm if a["hidden_failure"])
    no_letter_genuine = len(nm) - hidden_correct - hidden_failure

    subtypes = Counter(a["subtype"] for a in audits if a["subtype"])
    n = len(samples)
    n_orig = sum(1 for s in samples if s.get("exact_match") == 1.0)
    n_relive = sum(1 for a in audits if a["final_correct"])
    n_boxed_disagree = sum(1 for a in audits if a["boxed_disagreement"])
    n_rescuable = sum(1 for a in audits if a["subtype"] == "rescuable_via_boxed")

    # via_boxed accounting — how many samples' extracted letter came from a
    # \\boxed{X}, broken down by which Layer-1 path picked it up.
    via_boxed_by_layer1 = Counter()
    via_boxed_correct = 0
    for a in audits:
        if a["via_boxed"]:
            via_boxed_by_layer1[a["layer1"]] += 1
            if a["final_correct"]:
                via_boxed_correct += 1
    n_via_boxed = sum(via_boxed_by_layer1.values())

    return dict(
        name=name, path=str(path), n=n, audits=audits,
        layer1=layer1,
        hidden_correct=hidden_correct,
        hidden_failure=hidden_failure,
        no_letter_genuine=no_letter_genuine,
        subtypes=subtypes,
        n_orig=n_orig, n_relive=n_relive,
        n_boxed_disagree=n_boxed_disagree,
        n_rescuable=n_rescuable,
        n_via_boxed=n_via_boxed,
        via_boxed_by_layer1=via_boxed_by_layer1,
        via_boxed_correct=via_boxed_correct,
    )


def _pct(v, n):
    return f"{v/n*100:5.1f}%" if n else "  N/A"


SUBTYPE_ORDER = [
    "extracted_wrong", "boxed_wrong", "bare_wrong", "rescuable_via_boxed",
    "hidden_failure",
    "reach_max_function_call", "python_block_unfinished",
    "repetition_loop", "truncated_max_tokens",
    "medium_unconverged", "gave_up_short",
]


def print_summary(r):
    n = r["n"]
    print(f"\n=== {r['name']} ===")
    print(f"  file: {r['path']}")
    print(f"  total: {n}")
    print(f"\n  Accuracy variants:")
    print(f"    original exact_match     : {r['n_orig']}/{n} = {r['n_orig']/n:.4f}")
    print(f"    re-live extractor        : {r['n_relive']}/{n} = {r['n_relive']/n:.4f}")

    print(f"\n  Layer 1 — extraction path:")
    for k in ("pattern_answer", "pattern_X_correct", "pattern_final", "boxed", "bare_letter", "no_match"):
        v = r["layer1"].get(k, 0)
        print(f"    {k:24s} {v:4d}  ({_pct(v, n)})")

    nm = r["layer1"].get("no_match", 0)
    if nm:
        print(f"\n  Layer 2 — within no_match ({nm} samples):")
        print(f"    hidden_correct  {r['hidden_correct']:3d}  ({_pct(r['hidden_correct'], nm)} of no_match)")
        print(f"    hidden_failure  {r['hidden_failure']:3d}  ({_pct(r['hidden_failure'], nm)} of no_match)")
        print(f"    no letter found {r['no_letter_genuine']:3d}  ({_pct(r['no_letter_genuine'], nm)} of no_match)")

    print(f"\n  Layer 3 — wrongs by subtype:")
    n_wrong = sum(r["subtypes"].values())
    for st in SUBTYPE_ORDER:
        v = r["subtypes"].get(st, 0)
        if v == 0: continue
        print(f"    {st:24s} {v:4d}  ({_pct(v, n_wrong)} of wrongs)")

    print(f"\n  Boxed-aware audit:")
    n_vb = r["n_via_boxed"]
    print(f"    samples whose extracted letter came from \\boxed{{X}}: "
          f"{n_vb}/{n}  ({_pct(n_vb, n)})")
    print(f"      of those, correct: {r['via_boxed_correct']}/{n_vb}  "
          f"({(r['via_boxed_correct']/n_vb if n_vb else 0):.4f})")
    print(f"      via_boxed by Layer-1 path:")
    for k in ("pattern_answer", "pattern_X_correct", "pattern_final", "boxed", "bare_letter"):
        v = r["via_boxed_by_layer1"].get(k, 0)
        if v:
            total_in_path = r["layer1"].get(k, 0)
            print(f"        {k:24s} {v:3d} / {total_in_path}  "
                  f"({(v/total_in_path if total_in_path else 0)*100:.1f}% of that path)")
    print(f"    boxed-vs-extractor disagreement (potential bug): {r['n_boxed_disagree']}")
    print(f"      of those, rescuable_via_boxed (boxed = target): {r['n_rescuable']}")


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

    row("acc (original)",                lambda r: f"{r['n_orig']/r['n']:>14.4f}")
    row("acc (re-live)",                 lambda r: f"{r['n_relive']/r['n']:>14.4f}")
    row("  bare_letter used",            lambda r: f"{r['layer1'].get('bare_letter', 0):>14d}")
    row("  no_match total",              lambda r: f"{r['layer1'].get('no_match', 0):>14d}")
    row("  via_boxed total",             lambda r: f"{r['n_via_boxed']:>14d}")
    row("  via_boxed acc",               lambda r: f"{(r['via_boxed_correct']/r['n_via_boxed'] if r['n_via_boxed'] else 0):>14.4f}")
    row("  boxed disagreements",         lambda r: f"{r['n_boxed_disagree']:>14d}")
    row("  rescuable via boxed (target)",lambda r: f"{r['n_rescuable']:>14d}")

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
