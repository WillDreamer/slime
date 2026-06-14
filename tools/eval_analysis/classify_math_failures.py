#!/usr/bin/env python3
"""
Audit a math500 / aime samples_*.jsonl for hidden corrects, hidden failures,
and the WHY behind every remaining wrong answer. Mirrors the manual analysis
we did across base / base_math / final_search.

Three layers:

  Layer 1 — boxed verdict (harness vs slime grader)
    agree_correct      : harness ✓ AND slime ✓
    hidden_correct_box : harness ✗ but slime ✓ on first \\boxed{}
                         (LaTeX-equivalence rescue, e.g. 0.09 vs \\frac{9}{100})
    boxed_demoted      : harness ✓ but slime ✗ on first \\boxed{}
                         (rare — string-norm got lucky)
    agree_wrong        : both wrong on the boxed value

  Layer 2 — for the no_boxed subset (no \\boxed{} at all):
    hidden_correct_text: response prose contains a candidate sympy-equivalent
                         to target. Single-digit / 0 targets flagged 'fragile'
                         since they false-positive easily.
    hidden_failure     : response committed to a final answer in prose
                         ('the answer is X', 'Therefore X', etc.) but the
                         committed value is WRONG. This is a real reasoning
                         error that just happens to skip the \\boxed wrapper.

  Layer 3 — among samples that are STILL wrong after layers 1+2, why?
    wrong_answer        : has \\boxed{...} but it really is wrong
    hidden_failure      : (carried up from layer 2)
    no_boxed_no_commit  : never settled on a final answer in prose either —
                          drilled down further into:
      reach_max_function_call : hallucinated TIR trace ('Reach max function call limit')
      python_block_unfinished : stuck mid ```python ... ``` / ```output``` block
      repetition_loop         : tail of response is a repeating substring
      truncated_max_tokens    : long, no looping pattern, just cut off
      gave_up_short           : short output that trails off without a box
      medium_unconverged      : medium-length, just never lands on a final answer

Usage:
  python classify_math_failures.py samples_math500_slime_*.jsonl

  # compare ckpts
  python classify_math_failures.py base=…jsonl base_math=…jsonl final_search=…jsonl

  # dump examples per category for human review
  python classify_math_failures.py samples.jsonl --dump-dir audit_out/

  # disable the slime grader (fall back to original exact_match field only)
  python classify_math_failures.py samples.jsonl --no-regrade
"""
import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MATH_UTILS_PATH = "/data1/hhzhang/slime/slime/rollout/rm_hub/math_utils.py"


# ---------- response extraction ----------

def get_response(sample) -> str:
    fr = sample.get("filtered_resps")
    if isinstance(fr, list) and fr:
        x = fr[0]
        if isinstance(x, str):
            return x
        if isinstance(x, list) and x and isinstance(x[0], str):
            return x[0]
    return ""


def strip_think(text: str) -> str:
    return text.split("</think>")[-1] if "</think>" in text else text


def first_boxed(text: str):
    """First balanced \\boxed{...}/\\fbox{...}; None if missing/unbalanced."""
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


# ---------- slime grader loader (mathd + sympy) ----------

def load_grader(math_utils_path: str):
    if not Path(math_utils_path).exists():
        print(f"[warn] math_utils.py not found at {math_utils_path}", file=sys.stderr)
        return None
    try:
        spec = importlib.util.spec_from_file_location("_slime_math", math_utils_path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for fn in ("grade_answer_mathd", "grade_answer_sympy", "extract_boxed_answer"):
            if not hasattr(m, fn):
                print(f"[warn] math_utils missing {fn}", file=sys.stderr)
                return None
        return m
    except Exception as e:
        print(f"[warn] failed to load math_utils ({e.__class__.__name__}: {e}); "
              "ensure sympy + pylatexenc are installed.", file=sys.stderr)
        return None


def slime_grade(grader, candidate: str | None, target: str) -> bool:
    if candidate is None or target is None:
        return False
    if "\\boxed" in target:
        target = grader.extract_boxed_answer(target) or target
    try:
        return grader.grade_answer_mathd(candidate, target) or grader.grade_answer_sympy(candidate, target)
    except Exception:
        try:
            return grader.grade_answer_mathd(candidate, target)
        except Exception:
            return False


# ---------- candidate extraction for hidden_correct_text ----------

# Patterns ordered roughly from most-specific to least-specific so that
# duplicates from broader patterns (e.g. "3" inside "3/2") get deduped.
_CAND_PATTERNS = [
    r"\\frac\{-?\d+\}\{-?\d+\}",                           # \frac{a}{b}
    r"\\sqrt\{-?\d+\}",                                    # \sqrt{n}
    r"\\d?frac\{[^{}]+\}\{[^{}]+\}",                       # generic frac
    r"-?\d+\s*/\s*-?\d+",                                  # a/b
    r"-?\d+\.\d+",                                         # decimal
    r"-?\d+",                                              # integer
]
_CAND_RE = re.compile("|".join(f"({p})" for p in _CAND_PATTERNS))


def extract_candidates(text: str, tail_chars: int = 1500, max_cands: int = 30) -> list[str]:
    """Pull answer-shaped tokens from the last `tail_chars`. Most-recent first
    (we walk the tail backward so earlier-in-text candidates are last)."""
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    cands: list[str] = []
    seen: set[str] = set()
    for m in _CAND_RE.finditer(tail):
        s = m.group(0).strip()
        if s and s not in seen:
            seen.add(s)
            cands.append(s)
    cands.reverse()  # last-occurring first; that's what the model "settled on"
    return cands[:max_cands]


def is_fragile_target(target: str) -> bool:
    """Single-digit / trivial targets false-positive easily on candidate match.
    Caller should treat hidden_correct_text on fragile targets as 'maybe'."""
    t = target.strip().strip("$").strip()
    if t in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
        return True
    if t in {"-1", "10"}:
        return True
    return False


# ---------- answer-commitment detection (for hidden_failure) ----------

# These markers indicate the model THINKS it landed on a final answer, even
# without a \\boxed{}. Used to distinguish 'committed wrong answer in prose'
# (hidden_failure) from 'never converged / gave up' (no_commit subtypes).
_ANSWER_COMMITMENT_RE = re.compile(
    r"(?i)(?:"
    r"(?:final\s+)?answer\s+is\b"
    r"|answer\s*[:=]"
    r"|\btherefore[,\s]"
    r"|\bthus[,\s]"
    r"|\bhence[,\s]"
    r"|\bso\s+the\s+answer"
    r"|\bwe\s+(?:get|have|conclude|obtain|find)\b"
    r"|\bequals\s+(?:to\s+)?"
    r"|\bresult\s+is\b"
    r")"
)


def has_answer_commitment(text: str, tail_chars: int = 1500) -> bool:
    """True if the tail contains language signaling the model committed to a
    final answer (even though it didn't produce a \\boxed{})."""
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    return bool(_ANSWER_COMMITMENT_RE.search(tail))


# ---------- failure subtype detection (for genuine no_boxed errors) ----------

REACH_MAX_RE = re.compile(r"Reach\s+max\s+function\s+call\s+limit", re.IGNORECASE)
PYTHON_FENCE_RE = re.compile(r"```(?:python|output|tool_call|sympy)", re.IGNORECASE)


def has_repetition_loop(text: str, tail_chars: int = 4000, window: int = 60,
                        min_repeats: int = 8) -> bool:
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    if len(tail) < window * min_repeats:
        return False
    step = max(1, window // 3)
    seen: Counter[str] = Counter()
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


def python_block_unfinished(text: str) -> bool:
    if not PYTHON_FENCE_RE.search(text):
        return False
    fences = re.findall(r"```", text)
    if len(fences) % 2 == 1:
        return True
    tail = text[-2000:]
    if PYTHON_FENCE_RE.search(tail) and "\\boxed" not in tail:
        return True
    return False


def classify_no_boxed_subtype(text: str, truncation_thresh: int) -> str:
    if REACH_MAX_RE.search(text):
        return "reach_max_function_call"
    if python_block_unfinished(text):
        return "python_block_unfinished"
    if has_repetition_loop(text):
        return "repetition_loop"
    n = len(text)
    if n > truncation_thresh:
        return "truncated_max_tokens"
    if n < 3000:
        return "gave_up_short"
    return "medium_unconverged"


# ---------- per-sample audit ----------

def audit_sample(s: dict, grader, truncation_thresh: int) -> dict:
    """Returns a dict with the verdicts. Mutates nothing."""
    harness_ok = s.get("exact_match") == 1.0
    target = s.get("target", "")
    txt_full = get_response(s)
    txt = strip_think(txt_full)
    box = first_boxed(txt)

    slime_ok = None
    if grader is not None and box is not None:
        slime_ok = slime_grade(grader, box, target)

    # Layer 1: boxed verdict (the boxed_demoted bucket replaces what was
    # previously called hidden_failure to keep that name reserved for
    # no_boxed wrong-prose-answer cases).
    if box is None:
        layer1 = "no_boxed"
    elif slime_ok is None:
        layer1 = "agree_correct" if harness_ok else "agree_wrong"
    elif harness_ok and slime_ok:
        layer1 = "agree_correct"
    elif (not harness_ok) and slime_ok:
        layer1 = "hidden_correct_box"
    elif harness_ok and (not slime_ok):
        layer1 = "boxed_demoted"
    else:
        layer1 = "agree_wrong"

    # Layer 2: only for no_boxed samples
    #   hidden_correct_text -> rescued
    #   hidden_failure      -> committed to a wrong final answer in prose
    hidden_text = None        # 'solid' / 'fragile' when text-rescue triggers
    matching_cand = None
    is_hidden_failure = False
    committed = False
    if layer1 == "no_boxed":
        if grader is not None:
            for cand in extract_candidates(txt):
                if slime_grade(grader, cand, target):
                    hidden_text = "fragile" if is_fragile_target(target) else "solid"
                    matching_cand = cand
                    break
        if hidden_text is None:
            committed = has_answer_commitment(txt)
            if committed:
                is_hidden_failure = True

    # Final correctness for the surviving pile
    if layer1 in {"agree_correct", "hidden_correct_box"}:
        final_correct = True
    elif layer1 == "no_boxed" and hidden_text == "solid":
        final_correct = True
    else:
        final_correct = False

    # Layer 3: WHY each surviving wrong is wrong
    subtype = None
    if not final_correct:
        if layer1 in {"agree_wrong", "boxed_demoted"}:
            subtype = "wrong_answer"
        elif is_hidden_failure:
            subtype = "hidden_failure"
        elif hidden_text == "fragile":
            # ambiguous match — keep separate so it doesn't pollute either bin
            subtype = "fragile_text_match"
        else:
            subtype = classify_no_boxed_subtype(txt_full, truncation_thresh)

    return dict(
        doc_id=s.get("doc_id"),
        harness_ok=harness_ok,
        slime_ok=slime_ok,
        box=box,
        target=target,
        layer1=layer1,
        hidden_text=hidden_text,
        matching_cand=matching_cand,
        is_hidden_failure=is_hidden_failure,
        committed=committed,
        final_correct=final_correct,
        subtype=subtype,
        resp_len=len(txt_full),
    )


# ---------- per-file analysis ----------

def analyze(path: Path, name: str, grader, truncation_thresh: int) -> dict:
    samples = [json.loads(l) for l in open(path)]
    audits = [audit_sample(s, grader, truncation_thresh) for s in samples]

    layer1 = Counter(a["layer1"] for a in audits)

    nb_audits = [a for a in audits if a["layer1"] == "no_boxed"]
    hidden_text_solid = sum(1 for a in nb_audits if a["hidden_text"] == "solid")
    hidden_text_fragile = sum(1 for a in nb_audits if a["hidden_text"] == "fragile")
    hidden_failure = sum(1 for a in nb_audits if a["is_hidden_failure"])
    no_commit_count = len(nb_audits) - hidden_text_solid - hidden_text_fragile - hidden_failure

    subtypes = Counter(a["subtype"] for a in audits if a["subtype"])

    n = len(samples)
    n_harness = sum(1 for a in audits if a["harness_ok"])
    n_slime_box = (n_harness
                   + layer1.get("hidden_correct_box", 0)
                   - layer1.get("boxed_demoted", 0))
    n_after_text_rescue = n_slime_box + hidden_text_solid

    return dict(
        name=name, path=str(path), n=n, audits=audits,
        layer1=layer1,
        hidden_text_solid=hidden_text_solid,
        hidden_text_fragile=hidden_text_fragile,
        hidden_failure=hidden_failure,
        no_commit_count=no_commit_count,
        subtypes=subtypes,
        n_harness=n_harness,
        n_slime_box=n_slime_box,
        n_after_text_rescue=n_after_text_rescue,
        regraded=grader is not None,
    )


# ---------- printing ----------

def _pct(v, n):
    return f"{v/n*100:5.1f}%" if n else "  N/A"


SUBTYPE_ORDER = [
    "wrong_answer",
    "hidden_failure",
    "fragile_text_match",
    "reach_max_function_call",
    "python_block_unfinished",
    "repetition_loop",
    "truncated_max_tokens",
    "medium_unconverged",
    "gave_up_short",
]


def print_summary(r: dict):
    n = r["n"]
    print(f"\n=== {r['name']} ===")
    print(f"  file: {r['path']}")
    print(f"  total: {n}")
    print(f"\n  Accuracy variants:")
    print(f"    harness original              : {r['n_harness']}/{n} = {r['n_harness']/n:.4f}")
    if r["regraded"]:
        print(f"    after slime regrade (boxed)   : {r['n_slime_box']}/{n} = {r['n_slime_box']/n:.4f}  "
              f"({'+' + str(r['layer1'].get('hidden_correct_box',0))} hidden_correct_box, "
              f"{'-' + str(r['layer1'].get('hidden_failure',0))} hidden_failure)")
        print(f"    + plain-text hidden_correct   : {r['n_after_text_rescue']}/{n} = "
              f"{r['n_after_text_rescue']/n:.4f}  "
              f"(+{r['hidden_text_solid']} solid, +{r['hidden_text_fragile']} fragile/maybe)")

    if r["regraded"]:
        print(f"\n  Layer 1 — boxed verdict (harness vs slime grader):")
        for k in ("agree_correct", "hidden_correct_box", "boxed_demoted", "agree_wrong", "no_boxed"):
            v = r["layer1"].get(k, 0)
            print(f"    {k:22s} {v:4d}  ({_pct(v, n)})")
        print(f"\n  Layer 2 — within no_boxed ({r['layer1'].get('no_boxed', 0)} samples):")
        nb = r["layer1"].get("no_boxed", 0)
        print(f"    hidden_correct_text (solid)    {r['hidden_text_solid']:4d}  ({_pct(r['hidden_text_solid'], nb)} of no_boxed)  ← rescued")
        print(f"    hidden_correct_text (fragile)  {r['hidden_text_fragile']:4d}  ({_pct(r['hidden_text_fragile'], nb)} of no_boxed)  ← needs human verify")
        print(f"    hidden_failure (committed wrong){r['hidden_failure']:4d}  ({_pct(r['hidden_failure'], nb)} of no_boxed)  ← prose answer is wrong")
        print(f"    no commitment (didn't settle)  {r['no_commit_count']:4d}  ({_pct(r['no_commit_count'], nb)} of no_boxed)")

    print(f"\n  Layer 3 — surviving wrongs by subtype:")
    n_wrong = sum(r["subtypes"].values())
    for st in SUBTYPE_ORDER:
        v = r["subtypes"].get(st, 0)
        if v == 0:
            continue
        print(f"    {st:28s} {v:4d}  ({_pct(v, n_wrong)} of wrongs)")


def print_compare(results: list[dict]):
    if len(results) < 2:
        return
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON")
    print("=" * 80)
    header = f"{'metric':32s} " + " ".join(f"{r['name']:>14s}" for r in results)
    print(header)
    print("-" * len(header))

    def row(label, fmt):
        cells = " ".join(fmt(r) for r in results)
        print(f"{label:32s} {cells}")

    row("acc (harness)", lambda r: f"{r['n_harness']/r['n']:>14.4f}")
    if any(r["regraded"] for r in results):
        row("acc (slime regrade boxed)", lambda r: f"{r['n_slime_box']/r['n']:>14.4f}")
        row("acc (+ text rescue)", lambda r: f"{r['n_after_text_rescue']/r['n']:>14.4f}")
        row("  hidden_correct_box", lambda r: f"{r['layer1'].get('hidden_correct_box', 0):>14d}")
        row("  boxed_demoted", lambda r: f"{r['layer1'].get('boxed_demoted', 0):>14d}")
        row("  hidden_correct_text solid", lambda r: f"{r['hidden_text_solid']:>14d}")
        row("  hidden_correct_text fragile", lambda r: f"{r['hidden_text_fragile']:>14d}")
        row("  hidden_failure (no_boxed)", lambda r: f"{r['hidden_failure']:>14d}")

    print(f"\n{'no_boxed subtypes':32s} " + " ".join(f"{r['name']:>14s}" for r in results))
    for st in SUBTYPE_ORDER:
        cells = " ".join(f"{r['subtypes'].get(st, 0):>14d}" for r in results)
        print(f"  {st:30s} {cells}")


# ---------- example dumps ----------

def dump_examples(r: dict, out_dir: Path, max_chars: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for a in r["audits"]:
        if a["layer1"] == "hidden_correct_box":
            buckets["hidden_correct_box"].append(a)
        if a["layer1"] == "boxed_demoted":
            buckets["boxed_demoted"].append(a)
        if a["hidden_text"] == "solid":
            buckets["hidden_correct_text_solid"].append(a)
        if a["hidden_text"] == "fragile":
            buckets["hidden_correct_text_fragile"].append(a)
        if a["is_hidden_failure"]:
            buckets["hidden_failure"].append(a)
        if a["subtype"] and a["subtype"] not in {"wrong_answer", "hidden_failure", "fragile_text_match"}:
            buckets[f"no_commit_{a['subtype']}"].append(a)

    samples_by_id = {s["doc_id"]: s for s in
                     (json.loads(l) for l in open(r["path"]))}

    for bucket, audits in buckets.items():
        if not audits:
            continue
        f_path = out_dir / f"{r['name']}__{bucket}.txt"
        with open(f_path, "w") as f:
            f.write(f"# {r['name']} — bucket: {bucket} ({len(audits)} samples)\n\n")
            for a in audits:
                s = samples_by_id.get(a["doc_id"], {})
                doc = s.get("doc", {})
                txt = get_response(s)
                f.write("=" * 80 + "\n")
                f.write(f"doc_id={a['doc_id']}  subject={doc.get('subject')}  "
                        f"level={doc.get('level')}  resp_len={len(txt)}\n")
                f.write(f"target:  {a['target']!r}\n")
                if a["box"] is not None:
                    f.write(f"boxed:   {a['box']!r}\n")
                f.write(f"harness_ok={a['harness_ok']}  slime_ok={a['slime_ok']}\n")
                if a["matching_cand"] is not None:
                    f.write(f"matching_candidate: {a['matching_cand']!r}  "
                            f"(hidden_text={a['hidden_text']})\n")
                f.write(f"layer1={a['layer1']}  subtype={a['subtype']}\n\n")
                f.write(f"--- problem ---\n{doc.get('problem', '')}\n\n")
                if len(txt) > max_chars:
                    f.write(f"--- response (head, {max_chars}/{len(txt)} chars) ---\n")
                    f.write(txt[:max_chars])
                    f.write("\n\n--- response (tail, last 1500 chars) ---\n")
                    f.write(txt[-1500:])
                else:
                    f.write(f"--- response ({len(txt)} chars) ---\n")
                    f.write(txt)
                f.write("\n\n")
        print(f"  [dump] {f_path.name}  ({len(audits)} samples)")


# ---------- CLI ----------

def parse_input(arg: str) -> tuple[str, Path]:
    if "=" in arg and not arg.startswith(("/", ".")):
        name, path = arg.split("=", 1)
    else:
        path = arg
        name = Path(path).stem
    return name, Path(path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="One or more samples_*.jsonl paths. Use NAME=PATH to label.")
    p.add_argument("--no-regrade", action="store_true",
                   help="Skip slime grader (mathd+sympy). Disables hidden_correct_box, "
                        "hidden_failure, and hidden_correct_text detection.")
    p.add_argument("--math-utils-path", default=DEFAULT_MATH_UTILS_PATH,
                   help=f"Path to slime/rollout/rm_hub/math_utils.py "
                        f"(default {DEFAULT_MATH_UTILS_PATH})")
    p.add_argument("--truncation-thresh", type=int, default=30000,
                   help="Char count above which a no_boxed response is treated as "
                        "max_tokens-truncated (default 30000).")
    p.add_argument("--dump-dir", default=None,
                   help="Write per-bucket example dumps under this directory.")
    p.add_argument("--max-chars", type=int, default=6000,
                   help="When dumping, head limit per response (default 6000).")
    args = p.parse_args()

    grader = None if args.no_regrade else load_grader(args.math_utils_path)
    if grader is None and not args.no_regrade:
        print("[warn] running without slime grader; "
              "hidden_correct_box / hidden_failure / hidden_correct_text won't be detected.",
              file=sys.stderr)

    results = []
    for inp in args.inputs:
        name, path = parse_input(inp)
        if not path.exists():
            print(f"[skip] {path} does not exist", file=sys.stderr)
            continue
        r = analyze(path, name, grader, args.truncation_thresh)
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
