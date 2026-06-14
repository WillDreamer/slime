"""
Parsing utilities that exactly mirror slime's reward logic.

Math: slime/rollout/rm_hub/math_utils.py  (deepscaler reward)
GPQA: slime/rollout/rm_hub/gpqa.py
MC (MMLU): extracts "Answer: X" from the text after </think>
"""

import re
import string
from typing import Dict, List


# Lazy-import slime's grade_answer_sympy so math grading matches the training
# reward (slime/rollout/rm_hub/deepscaler.py and math_utils.py:grade_answer_verl
# both do `grade_answer_mathd OR grade_answer_sympy`). Without this fallback,
# this harness misses LaTeX-equivalence cases like 0.09 vs \frac{9}{100},
# C vs \text{(C)}, 1/5 vs \frac{1}{5} — ~13 / 500 false-negatives per ckpt.
def _load_grade_answer_sympy():
    import importlib.util as _imp
    path = "/data1/hhzhang/slime/slime/rollout/rm_hub/math_utils.py"
    try:
        spec = _imp.spec_from_file_location("_slime_math_utils_for_eval", path)
        m = _imp.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.grade_answer_sympy
    except Exception:
        return lambda a, b: False  # graceful fallback if sympy/pylatexenc missing

_grade_sympy = _load_grade_answer_sympy()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    """Remove <think>...</think> block, returning only the post-think text.
    Mirrors deepscaler.py: response.split('</think>')[-1]
    """
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text


# ---------------------------------------------------------------------------
# Math parsing  (mirrors math_utils.py / deepscaler.py)
# ---------------------------------------------------------------------------

def _last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx: right_brace_idx + 1]


def _remove_boxed(s: str):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left): -1]
    except Exception:
        return None


def extract_boxed_answer(solution: str):
    s = _last_boxed_only_string(solution)
    if s is None:
        return None
    return _remove_boxed(s)


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string):
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("\\%", "").replace(r"\%", "")
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            string = splits[0]
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    # a/b -> \frac{a}{b}
    parts = string.split("/")
    if len(parts) == 2:
        try:
            a, b = int(parts[0]), int(parts[1])
            if string == f"{a}/{b}":
                string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        except Exception:
            pass
    return string


def mathd_normalize_answer(answer):
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def grade_answer_mathd(given: str, truth: str) -> bool:
    return mathd_normalize_answer(given) == mathd_normalize_answer(truth)


def is_equiv(str1, str2) -> bool:
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


def process_results_math(doc: dict, results: List[str]) -> Dict[str, float]:
    """
    Extract \\boxed{} from after </think>.
    Mirrors deepscaler.py: extract_answer(response.split('</think>')[-1])
    Grading: grade_answer_mathd (from math_utils.py).
    """
    response = results[0]
    model_solution = _strip_think(response)
    model_answer = extract_boxed_answer(model_solution)

    # Determine ground truth key
    target = None
    for k in ("answer", "Answer", "solution", "target"):
        if k in doc:
            target = str(doc[k])
            break

    if model_answer is None or target is None:
        return {"exact_match": 0.0}

    # Remove \boxed from target if present
    if "\\boxed" in target:
        target = extract_boxed_answer(target) or target

    correct = (
        grade_answer_mathd(model_answer, target)
        or is_equiv(model_answer, target)
        or _grade_sympy(model_answer, target)
    )
    return {"exact_match": float(correct)}


# ---------------------------------------------------------------------------
# GPQA / multiple-choice parsing  (mirrors gpqa.py)
# ---------------------------------------------------------------------------

_VALID_LETTERS = list(string.ascii_uppercase[:8])


_GPQA_BOXED_RE = re.compile(r"\\boxed\{\(?([A-Z])\)?\}")


def _extract_letter_from_response(response: str, valid_letters) -> str | None:
    """Mirrors slime/rollout/rm_hub/gpqa.py::_extract_letter_from_response,
    extended with an explicit \\boxed{X} extraction step before the bare-letter
    fallback (math-RL'd models often emit \\boxed{X} on GPQA — promoting it to a
    first-class path makes the bucket accounting honest, and bare-letter only
    fires when neither strict patterns nor a box are present)."""
    if not response:
        return None
    text = _strip_think(response)
    valid_letters = {l.upper() for l in valid_letters}
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])",
        r"([A-Z])\s*(?:is\s*(?:the)?\s*correct)",
        r"final\s*(?:answer|option)\s*(?:is|:)?\s*([A-Z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter
    # \boxed{X} — last match wins (consistent with bare-letter "last A-Z" semantics).
    boxed = _GPQA_BOXED_RE.findall(text)
    for letter in reversed(boxed):
        if letter.upper() in valid_letters:
            return letter.upper()
    candidates = re.findall(r"\b([A-Z])\b", text)
    for letter in reversed(candidates):
        if letter.upper() in valid_letters:
            return letter.upper()
    return None


def process_results_gpqa(doc: dict, results: List[str]) -> Dict[str, float]:
    """
    Extract answer letter from model response after </think>.
    Mirrors slime/rollout/rm_hub/gpqa.py::compute_gpqa_reward
    doc['answer'] is "(B)" format set by process_docs_gpqa (copied from lm_eval gpqa utils).
    """
    response = results[0]
    valid_letters = list(string.ascii_uppercase[:4])  # A B C D

    # answer is "(B)" after process_docs_gpqa
    answer_raw = doc.get("answer", "")
    m = re.search(r"([A-D])", str(answer_raw).upper())
    correct_letter = m.group(1) if m else None

    extracted = _extract_letter_from_response(response, valid_letters)
    if extracted and correct_letter:
        return {"exact_match": float(extracted == correct_letter)}
    return {"exact_match": 0.0}


# ---------------------------------------------------------------------------
# GPQA process_docs — mirrors lm_eval gpqa cot_n_shot/utils.py exactly
# ---------------------------------------------------------------------------

import random
import datasets as _datasets


def process_docs_gpqa(dataset: _datasets.Dataset) -> _datasets.Dataset:
    def _preprocess(text):
        if text is None:
            return " "
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("  ", " ")
        return text

    def _process_doc(doc):
        choices = [
            _preprocess(doc["Incorrect Answer 1"]),
            _preprocess(doc["Incorrect Answer 2"]),
            _preprocess(doc["Incorrect Answer 3"]),
            _preprocess(doc["Correct Answer"]),
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(_preprocess(doc["Correct Answer"]))
        return {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": choices,
            "answer": f"({chr(65 + correct_answer_index)})",
        }

    return dataset.map(_process_doc)


# ---------------------------------------------------------------------------
# MMLU multiple-choice  (mirrors simple-evals common.py)
# Prompt asks: "The last line of your response should be: 'Answer: $LETTER'"
# Parser: regex on text after </think>
# ---------------------------------------------------------------------------

# Models trained with math RL (e.g. base_math) often emit \boxed{X} format
# even on non-math MCQ tasks. Add fallbacks so we don't underrate ~6% of MMLU.
_MMLU_PATTERNS = [
    (re.compile(r"(?i)Answer\s*:\s*\$?\s*([A-D])\$?"),                                "answer_colon"),
    (re.compile(r"\\boxed\{([A-D])\}"),                                               "boxed"),
    (re.compile(r"(?i)the\s+(?:correct\s+)?answer\s+is.{0,30}?\(?([A-D])\)?"),        "the_answer_is"),
]


def _extract_mmlu_answer(post_think: str):
    """Run all MMLU patterns, take the LAST hit (rightmost in response).
    Returns (letter or None, method tag)."""
    candidates = []  # (position, letter, method)
    for pat, name in _MMLU_PATTERNS:
        for m in pat.finditer(post_think):
            candidates.append((m.start(), m.group(1).upper(), name))
    if not candidates:
        return None, "no_match"
    candidates.sort(key=lambda x: x[0])
    _, letter, method = candidates[-1]
    return letter, method


def process_results_mmlu(doc: dict, results: List[str]) -> Dict[str, float]:
    """
    Extract MCQ answer from after </think>. Tries multiple patterns
    (Answer: X, \\boxed{X}, the answer is X) and takes the LAST match.
    """
    response = results[0]
    post_think = _strip_think(response)
    extracted, _ = _extract_mmlu_answer(post_think)
    if extracted is None:
        return {"exact_match": 0.0}

    answer_idx = doc.get("answer", -1)
    correct_letter = ["A", "B", "C", "D"][int(answer_idx)] if 0 <= int(answer_idx) <= 3 else None
    if correct_letter is None:
        return {"exact_match": 0.0}

    return {"exact_match": float(extracted == correct_letter)}


# ---------------------------------------------------------------------------
# MMLU doc_to_text  (matches simple-evals format exactly)
# ---------------------------------------------------------------------------

_MMLU_TEMPLATE = (
    "Answer the following multiple choice question. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
    "Think step by step before answering.\n\n"
    "{question}\n\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}"
)


def doc_to_text_mmlu(doc: dict) -> str:
    return _MMLU_TEMPLATE.format(
        question=doc["question"],
        A=doc["choices"][0],
        B=doc["choices"][1],
        C=doc["choices"][2],
        D=doc["choices"][3],
    )


# ---------------------------------------------------------------------------
# Math doc_to_text
# ---------------------------------------------------------------------------

_MATH_TEMPLATE = (
    "Solve the following problem. "
    "Show your reasoning, then put your final answer in \\boxed{{}}.\n\n"
    "{problem}"
)


def doc_to_text_aime24(doc: dict) -> str:
    return _MATH_TEMPLATE.format(problem=doc["Problem"])


def doc_to_text_aime25(doc: dict) -> str:
    return _MATH_TEMPLATE.format(problem=doc["problem"])


def doc_to_text_math500(doc: dict) -> str:
    return _MATH_TEMPLATE.format(problem=doc["problem"])


# ---------------------------------------------------------------------------
# MMLU-Pro zero-shot CoT  (10 choices A-J)
# Prompt: "Think step by step... the answer is (X)"
# Parsing: extract "the answer is (X)" after </think>
# ---------------------------------------------------------------------------

_MMLU_PRO_CHOICES = list("ABCDEFGHIJ")

_MMLU_PRO_TEMPLATE = (
    "Answer the following multiple choice question. "
    "Think step by step, then finish your answer with "
    "\"the answer is (X)\" where X is the correct letter choice.\n\n"
    "Question:\n{question}\n"
    "Options:\n{options}"
)

_ANSWER_PATTERN_PRO = re.compile(
    r"(?i)the\s+answer\s+is\s+\(?([A-J])\)?",
)
# math-RL'd models often emit \boxed{X} instead of the strict template.
# This pattern catches them before the bare-letter fallback so we don't
# accidentally pick up a stray A-J letter mentioned AFTER the box.
_BOXED_PATTERN_PRO = re.compile(r"\\boxed\{\(?([A-J])\)?\}")


def doc_to_text_mmlu_pro(doc: dict) -> str:
    options_text = ""
    for i, opt in enumerate(doc["options"]):
        if i >= len(_MMLU_PRO_CHOICES):
            break
        options_text += f"{_MMLU_PRO_CHOICES[i]}. {opt.strip()}\n"
    return _MMLU_PRO_TEMPLATE.format(
        question=doc["question"].strip(),
        options=options_text,
    )


def extract_mmlu_pro_answer(post_think: str) -> tuple[str | None, str]:
    """Three-tier MMLU-Pro letter extraction. Returns (letter, method).
    Method is one of: 'strict' / 'boxed' / 'bare' / 'no_match'.

      strict : 'the answer is (X)'  — official template
      boxed  : '\\boxed{X}'         — math-RL'd format leak
      bare   : last \\b[A-J]\\b     — last-resort prose scrape
    """
    m = _ANSWER_PATTERN_PRO.findall(post_think)
    if m:
        return m[-1].upper(), "strict"
    m = _BOXED_PATTERN_PRO.findall(post_think)
    if m:
        return m[-1].upper(), "boxed"
    cands = re.findall(r"\b([A-J])\b", post_think.upper())
    if cands:
        return cands[-1], "bare"
    return None, "no_match"


def process_results_mmlu_pro(doc: dict, results: List[str]) -> Dict[str, float]:
    """
    Three-tier extraction (strict 'the answer is (X)' → '\\boxed{X}' → bare A-J).
    Ground truth: doc['answer'] is the letter string e.g. 'B'.
    """
    post_think = _strip_think(results[0])
    extracted, _ = extract_mmlu_pro_answer(post_think)
    correct = str(doc.get("answer", "")).strip().upper()
    if not extracted or not correct:
        return {"exact_match": 0.0}
    return {"exact_match": float(extracted == correct)}
