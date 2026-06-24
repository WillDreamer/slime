"""Standalone Search-R1 evaluation against a served model + local retriever.

Faithfully replicates the rollout loop in
slime_scai/examples/search-r1/generate_with_search_tools_qwen_sft.py — the
*tool-call* variant. The model emits actions as

    <tool_call>
    {"name": "search", "arguments": {"query": "..."}}
    </tool_call>

dense retrieval turns each query into a <tool_response>...</tool_response> user
turn, and <answer>...</answer> terminates. Scoring uses the Search-R1 EM scorer
(qa_em_format.compute_score_em, format_score=0 -> pure exact-match), whose tag
set was likewise switched to tool_call / tool_response.

This is NOT a drop-in for the original <search>/<information> harness: a model
trained with the tool-call template must be evaluated with this one, or every
rollout mis-parses. See the module-level NOTES block for the exact list of
behavioural differences relative to the old <search> harness.

Data: Search-R1 test parquet(s) — rows carry a chat 'prompt', 'data_source',
and 'reward_model'.ground_truth.target (gold answers).

Usage:
  python eval_search.py --base-url http://127.0.0.1:8400 \
      --retriever-url http://127.0.0.1:8500/retrieve \
      --tokenizer <hf ckpt> --data test.parquet --output results.json

# =============================================================================
# NOTES — where this diverges from the old <search>/<information> harness, i.e.
# everything that had to change *besides* the literal tag rename. Each item
# mirrors generate_with_search_tools_qwen_sft.py:
#
#  1. Chat template is rendered WITH tools=[SEARCH_TOOL_DESC]. The Qwen tool
#     template injects a <tools> system block the old harness never had.
#  2. The dataset prompts still carry the OLD `<search>` instruction text;
#     clean_instruction_in_messages() rewrites it to TOOL_CALLING_INSTRUCTION
#     before templating (otherwise the prompt tells the model to use <search>).
#  3. Retrieved passages are injected as a separate <|im_start|>user turn wrapped
#     in <tool_response>...</tool_response> (then a fresh assistant header) — NOT
#     concatenated into the running assistant string as <information> was.
#  4. Stop strings are ["</tool_call>", "</answer>"] (was ["</search>", ...]).
#  5. Action parsing reads tool_call JSON (with a bare-JSON fallback) instead of
#     a <search>/<answer> regex.
#  6. No INVALID_ACTION_OBS is injected on an unparseable action — the rollout
#     simply re-prompts the assistant, matching the training loop.
#  7. A context window (context_window_k=2) compresses older <tool_response>
#     blocks, exactly like MessageContextWindowManager. Only used by the LEGACY
#     --context-format; the default 'chat' format renders all results in full.
#
# CONTEXT FORMAT (--context-format, default 'chat'):
#   The 8B Search/Tau checkpoints were trained with
#   generate_with_search_tools_qwen_sft_no_drift.generate, which delegates ALL
#   turn-wrapping to tokenizer.apply_chat_template — every assistant turn is
#   closed with <|im_end|> and each tool result is a proper <|im_start|>user
#   turn. The original string-concat path (now --context-format legacy) omitted
#   the <|im_end|> between assistant and the tool_response, shifting every
#   observation token by 1 after turn 1; the model fell off-distribution on
#   multi-turn / hard questions and stopped emitting <tool_call>/<answer>.
#   'chat' (render_chat) reproduces the no_drift construction byte-for-byte.
#   Use 'legacy' only for the 30B checkpoints trained on the non-no_drift fn.
#
# FORCED FINAL ANSWER (--force-final-answer, default on):
#   Neither the eval nor the training rollout forces an answer (training only
#   shaped it via reward). A search-happy model (e.g. the Tau-SFT one) that
#   spends all max_turns searching, or one that rambles to the length cap, ends
#   with no <answer> and is floored to EM 0. After the loop, one constrained
#   turn (stop=</answer>, in-format nudge) lets it conclude. The summary reports
#   answered_rate / answered_only_em / forced_answer_rate so the floor is visible.
#
# TOKEN BUDGET (--max-response-tokens, default 4096):
#   Total response tokens across ALL turns, matching training
#   --rollout-max-response-len 4096. Per-turn max_new_tokens is clamped to the
#   remaining budget so the model can't monologue past what it saw in training.
#
# Knob to watch: --max-turns defaults to 5 (training rollout depth, matches
# SEARCH_R1_CONFIGS['max_turns']=5).
# =============================================================================
"""

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict

import aiohttp
import pandas as pd
from transformers import AutoTokenizer

from qa_em_format import compute_score_em


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True, help="sglang server root (no /v1)")
    p.add_argument("--retriever-url", required=True, help=".../retrieve endpoint")
    p.add_argument("--tokenizer", required=True, help="HF checkpoint for chat template")
    p.add_argument("--data", required=True, help="test parquet path")
    p.add_argument("--output", required=True)
    p.add_argument("--max-turns", type=int, default=5, help="generation rounds (matches the training rollout SEARCH_R1_CONFIGS['max_turns']=5)")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--context-window-k", type=int, default=2,
                   help="keep the last K <tool_response> blocks full; compress older ones")
    p.add_argument("--max-docs-compressed", type=int, default=1)
    p.add_argument("--max-chars-per-doc-compressed", type=int, default=1000)
    p.add_argument("--n-per-dataset", type=int, default=0, help="subsample per data_source (0 = all)")
    p.add_argument("--limit", type=int, default=0, help="global cap (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trajectory-output", default=None,
                   help="if set, write per-question full trajectory (prompt + multi-turn "
                        "tool_call/tool_response/answer response + gold) to this jsonl path")
    # ---- train/eval alignment knobs (see module docstring NOTE) --------------
    p.add_argument("--context-format", choices=["chat", "legacy"], default="chat",
                   help="how the multi-turn context is rebuilt each turn. 'chat' "
                        "re-renders the whole conversation via tokenizer.apply_chat_template "
                        "(assistant turns closed with <|im_end|>, tool results as proper user "
                        "turns) — byte-aligned with generate_with_search_tools_qwen_sft_no_drift, "
                        "which the 8B checkpoints were trained with. 'legacy' is the old "
                        "string-concat path (no <|im_end|> between assistant and tool_response) "
                        "matching generate_with_search_tools_qwen_sft (the 30B training).")
    p.add_argument("--force-final-answer", dest="force_final_answer",
                   action="store_true", default=True,
                   help="if the rollout ends (turn budget / length) without an <answer>, do one "
                        "extra constrained generation (stop=</answer>) so a search-happy model "
                        "still produces a final answer instead of being floored to EM 0 (default on)")
    p.add_argument("--no-force-final-answer", dest="force_final_answer", action="store_false")
    p.add_argument("--max-response-tokens", type=int, default=4096,
                   help="total response-token budget across ALL turns (matches training "
                        "--rollout-max-response-len 4096). Per-turn max_new_tokens is clamped to "
                        "the remaining budget. 0 = unlimited (per-turn cap only).")
    return p.parse_args()


# ---- tool definition + instruction rewriting --------------------------------
# Mirrors generate_with_search_tools_qwen_sft.py so the served (tool-call-trained)
# model sees exactly the prompt it was trained/rolled-out with.

SEARCH_TOOL_DESC = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information using a search engine",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
}


OLD_INSTRUCTION_SUBSTRING = (
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>."
)

OLD_NEW_INSTRUCTION_PREFIX = (
    "You must conduct reasoning inside <think> and </think> first every time you get new information.\n"
    "After reasoning, if you find you lack some knowledge, you may call a function to help.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:"
)

TOOL_CALLING_INSTRUCTION = (
    "First conduct reasoning in english every time you try to get new information.\n"
    "After reasoning, if you find you lack some knowledge, you may call a function to help.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    + json.dumps(SEARCH_TOOL_DESC, indent=2)
    + "\n</tools>\n\n"
    "For each function call, return a JSON object with function name and arguments "
    "within <tool_call></tool_call> XML tags. For example:\n"
    "<tool_call>\n"
    '{"name": "search", "arguments": {"query": "your search query"}}\n'
    "</tool_call>\n"
    "The search results will be returned in <tool_response></tool_response> tags.\n"
    "You can search as many times as you want.\n"
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>. \n"
)


def clean_instruction_in_messages(messages: list[dict]) -> list[dict]:
    """Swap the old <search> instruction for TOOL_CALLING_INSTRUCTION (Fix #2).

    The Search-R1 parquets predate the tool-call format, so their prompts still
    instruct the model to use <search>/<information>. Rewrite that text — or, if
    no known instruction is present, prepend the tool-calling instruction to the
    first user message — exactly as the training rollout does.
    """
    for msg in messages:
        content = msg["content"]
        if OLD_INSTRUCTION_SUBSTRING in content:
            msg["content"] = content.replace(OLD_INSTRUCTION_SUBSTRING, TOOL_CALLING_INSTRUCTION)
            return messages
        if OLD_NEW_INSTRUCTION_PREFIX in content:
            pattern = (
                r"You must conduct reasoning inside <think>.*?"
                r"For example, <answer> Beijing </answer>\."
            )
            msg["content"] = re.sub(pattern, TOOL_CALLING_INSTRUCTION, content, flags=re.DOTALL)
            return messages

    for msg in messages:
        if msg["role"] == "user":
            msg["content"] = TOOL_CALLING_INSTRUCTION + "\n\n" + msg["content"]
            break
    return messages


# ---- replicated from generate_with_search_tools_qwen_sft.py ------------------
def _passages2string(retrieval_result):
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


def postprocess_responses(resp: str) -> str:
    """Trim anything the model emitted past its first complete action."""
    if "</tool_call>" in resp:
        return resp.split("</tool_call>")[0] + "</tool_call>"
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    # Bare-JSON fallback: the model emitted a tool call without the wrapping tags.
    bare_pattern = r'\{[^{}]*"name"\s*:\s*"search"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
    match = re.search(bare_pattern, resp)
    if match:
        return resp[: match.end()]
    return resp


def _extract_search_query(data) -> tuple[str | None, str]:
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        return None, ""
    if data.get("name") != "search":
        return None, ""
    arguments = data.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:  # noqa: BLE001
            return None, ""
    if not isinstance(arguments, dict):
        return None, ""
    query = arguments.get("query", "")
    if not isinstance(query, str):
        return None, ""
    return "search", query.strip()


def _parse_tool_call_block(block: str):
    match = re.search(r"<tool_call>(.*?)</tool_call>", block, re.DOTALL)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group(1).strip())
    except Exception:  # noqa: BLE001
        return None, ""
    return _extract_search_query(data)


def _parse_bare_json_tool_call(text: str):
    """Fallback: bare {"name": "search", "arguments": {...}} without <tool_call> tags."""
    pattern = r'\{[^{}]*"name"\s*:\s*"search"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
    match = re.search(pattern, text)
    if not match:
        pattern = r'\{[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*"name"\s*:\s*"search"[^{}]*\}'
        match = re.search(pattern, text)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group(0))
    except Exception:  # noqa: BLE001
        return None, ""
    return _extract_search_query(data)


def postprocess_predictions(prediction: str):
    action, content = _parse_tool_call_block(prediction)
    if action is not None:
        return action, content

    action, content = _parse_bare_json_tool_call(prediction)
    if action is not None:
        return action, content

    match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if match:
        return "answer", match.group(1).strip()

    return None, ""


# ---- context window (mirror of MessageContextWindowManager) ------------------
def compress_search_result(text: str, max_docs: int, max_chars_per_doc: int) -> str:
    doc_pattern = r"(Doc \d+\(Title: [^)]*\))\s*(.*?)(?=Doc \d+\(Title:|$)"
    docs = re.findall(doc_pattern, text, re.DOTALL)
    if not docs:
        return text
    parts = []
    for header, body in docs[:max_docs]:
        body = body.strip()
        if len(body) > max_chars_per_doc:
            body = body[:max_chars_per_doc] + "..."
        parts.append(f"{header} {body}")
    return "\n".join(parts)


def _format_tool_response(search_result: str) -> str:
    # NOTE: this matches MessageContextWindowManager._format_tool_response
    # exactly, including the fact that the assistant turn is NOT closed with
    # <|im_end|> before the <|im_start|>user tool_response turn. The served
    # model produced its rollouts this way, so the eval must reproduce it.
    return (
        f"\n<|im_start|>user\n<tool_response>\n{search_result}\n</tool_response><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_context(prompt_text: str, turns: list[dict], args, add_final_generation_prompt: bool = True) -> str:
    """Rebuild the served context from the accumulated turns.

    turns: list of {"text": <assistant output>, "search_result": <str|None>}.
    Older <tool_response> blocks (beyond the last context_window_k searches) are
    compressed, exactly like MessageContextWindowManager.
    """
    num_search_turns = sum(1 for t in turns if t["search_result"] is not None)
    context = prompt_text
    search_count = 0
    for i, turn in enumerate(turns):
        is_last = i == len(turns) - 1
        context += turn["text"]
        if turn["search_result"] is not None:
            search_count += 1
            within_window = search_count > num_search_turns - args.context_window_k
            sr = (
                turn["search_result"]
                if within_window
                else compress_search_result(
                    turn["search_result"], args.max_docs_compressed, args.max_chars_per_doc_compressed
                )
            )
            context += _format_tool_response(sr)
        elif not is_last or add_final_generation_prompt:
            context += "\n<|im_start|>assistant\n"
    return context


# Forced-answer nudge injected as a final user turn when the rollout would
# otherwise end without an <answer>. Deliberately contains NO literal
# <answer>/</answer> tags: extract_solution() counts <answer> occurrences and
# returns the LAST one only when >=2 exist (the instruction's "<answer> Beijing
# </answer>" example is the baseline), so injecting tags here would pollute that
# count. The trained model emits the <answer>...</answer> wrapper on its own.
FORCE_ANSWER_NUDGE = (
    "You have used all available searches and cannot search further. "
    "Based on the information gathered so far, give your final answer now."
)


def render_chat(tokenizer, tools, base_messages, turns, add_generation_prompt=True,
                final_user=None):
    """No-drift context construction: re-render the WHOLE conversation through
    tokenizer.apply_chat_template every turn (mirrors
    generate_with_search_tools_qwen_sft_no_drift.ChatTemplateConversationManager
    with strip_think=False / SEARCH_R1_STRIP_THINK=0).

    Each accumulated turn becomes an assistant message (verbatim model text,
    think included) followed — when it issued a search — by a user message whose
    content is the already-wrapped <tool_response>...</tool_response>. This is
    byte-identical to the trained rollout, so the model stays in-distribution and
    keeps emitting <tool_call>/<answer> on multi-turn / hard questions. Unlike the
    legacy path there is no rolling compression: search results render in full.

    final_user, if given, is appended as a trailing user turn (used by the
    forced-answer step) before the generation prompt.
    """
    msgs = [dict(m) for m in base_messages]
    for t in turns:
        msgs.append({"role": "assistant", "content": t["text"]})
        if t["search_result"] is not None:
            msgs.append({
                "role": "user",
                "content": f"<tool_response>\n{t['search_result']}\n</tool_response>",
            })
    if final_user is not None:
        msgs.append({"role": "user", "content": final_user})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_generation_prompt, tools=tools
    )


async def retrieve(session, sem, url, query, topk):
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    async with sem:
        for attempt in range(5):
            try:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return _passages2string(data["result"][0])
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    raise
                await asyncio.sleep(2**attempt)


async def _generate(session, gen_sem, args, context, stop, max_new):
    """POST to sglang /generate with retries. Returns the json output or None."""
    payload = {
        "text": context,
        "sampling_params": {
            "max_new_tokens": max_new,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stop": stop,
            "no_stop_trim": True,  # keep the matched stop tag in the output
        },
    }
    for _attempt in range(5):
        try:
            async with gen_sem:
                async with session.post(f"{args.base_url}/generate", json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:  # noqa: BLE001 — transient server disconnect / 5xx
            if _attempt == 4:
                print(f"[gen] sample failed after retries: {e}", file=sys.stderr)
                return None
            await asyncio.sleep(1.5 * (_attempt + 1))
    return None


async def run_sample(session, gen_sem, ret_sem, args, base_messages, tokenizer, tools, prompt_text):
    """One multi-turn tool-call rollout.

    Returns a dict: solution_str (full conversation text scored by the EM scorer),
    response (trajectory suffix), answered (bool), forced (bool), n_turns (int).
    """
    turns: list[dict] = []
    used_tokens = 0

    def build_ctx(add_gen=True, final_user=None):
        if args.context_format == "chat":
            return render_chat(tokenizer, tools, base_messages, turns, add_gen, final_user)
        # legacy string-concat path (drift-y; matches the non-no_drift training)
        return build_context(prompt_text, turns, args, add_final_generation_prompt=add_gen)

    def remaining():
        if args.max_response_tokens and args.max_response_tokens > 0:
            return max(0, args.max_response_tokens - used_tokens)
        return args.max_new_tokens

    answered = False
    for _turn in range(args.max_turns):
        budget = min(args.max_new_tokens, remaining())
        if budget <= 0:
            break
        output = await _generate(session, gen_sem, args, build_ctx(add_gen=True),
                                 ["</tool_call>", "</answer>"], budget)
        if output is None:
            break

        cur = postprocess_responses(output["text"])
        used_tokens += len(tokenizer(cur, add_special_tokens=False)["input_ids"])
        finish = output["meta_info"]["finish_reason"]["type"]

        action, content = postprocess_predictions(cur)
        if action == "answer":
            turns.append({"text": cur, "search_result": None})
            answered = True
            break
        if finish == "length":
            # ran out of per-turn / total budget mid-thought; stop (forced-answer
            # step below gives it one constrained chance to conclude).
            turns.append({"text": cur, "search_result": None})
            break
        if action == "search":
            try:
                docs = await retrieve(session, ret_sem, args.retriever_url, content, args.topk)
            except Exception as e:  # noqa: BLE001
                print(f"[search] retrieval failed: {e}", file=sys.stderr)
                turns.append({"text": cur, "search_result": None})
                break
            turns.append({"text": cur, "search_result": docs.strip()})
        else:
            # Unparseable action: re-prompt the assistant (no INVALID_ACTION_OBS),
            # matching the training rollout's behaviour.
            turns.append({"text": cur, "search_result": None})

    # ---- forced final answer -------------------------------------------------
    # Neither the eval nor the training rollout *forces* an answer (training only
    # shaped it via reward), so a model that searches on its last turn or rambles
    # to the length cap ends with no <answer> and is floored to EM 0. Give it one
    # constrained turn that may only answer (stop=</answer>), nudged in-format.
    # Chat-format only — the legacy path can't append a clean trailing user turn.
    forced = False
    if (args.force_final_answer and not answered and args.context_format == "chat"
            and turns):
        # Dedicated budget independent of the (possibly exhausted) search budget —
        # a model that spent all its tokens searching must still get to conclude.
        ans_budget = min(args.max_new_tokens, 1024)
        output = await _generate(
            session, gen_sem, args,
            build_ctx(add_gen=True, final_user=FORCE_ANSWER_NUDGE),
            ["</answer>"], ans_budget,
        )
        if output is not None:
            cur = output["text"]
            if "</answer>" in cur:  # trim trailing only at the answer close
                cur = cur.split("</answer>")[0] + "</answer>"
            turns.append({"text": cur, "search_result": None})
            forced = True
            if "<answer>" in cur:
                answered = True

    solution_str = build_ctx(add_gen=False)
    response = solution_str[len(prompt_text):] if solution_str.startswith(prompt_text) else solution_str
    return {
        "solution_str": solution_str,
        "response": response,
        "answered": answered,
        "forced": forced,
        "n_turns": len(turns),
    }


async def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    if args.n_per_dataset > 0 and "data_source" in df.columns:
        # Explicit per-group sample (NOT groupby.apply — pandas 2.x drops the
        # grouping column there, which collapsed per_dataset to "unknown").
        parts = [
            g.sample(min(len(g), args.n_per_dataset), random_state=args.seed)
            for _, g in df.groupby("data_source")
        ]
        df = pd.concat(parts).reset_index(drop=True)
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"[search] {len(df)} questions from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tools = [SEARCH_TOOL_DESC]
    base_messages_list = [
        clean_instruction_in_messages([dict(m) for m in row["prompt"]])
        for _, row in df.iterrows()
    ]
    prompts = [
        tokenizer.apply_chat_template(
            bm, tokenize=False, add_generation_prompt=True, tools=tools
        )
        for bm in base_messages_list
    ]

    gen_sem = asyncio.Semaphore(args.concurrency)
    ret_sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[run_sample(session, gen_sem, ret_sem, args, bm, tokenizer, tools, p)
              for bm, p in zip(base_messages_list, prompts)]
        )

    traj_f = open(args.trajectory_output, "w") if args.trajectory_output else None
    per_ds_scores, details = defaultdict(list), []
    n_answered = n_forced = 0
    answered_score_sum = 0.0
    for (_, row), prompt_text, res in zip(df.iterrows(), prompts, results):
        gt = row["reward_model"]["ground_truth"]
        gt = {k: (list(v) if hasattr(v, "tolist") or isinstance(v, (list, tuple)) else v) for k, v in dict(gt).items()}
        # Same call as the RL reward_func, but format_score=0 -> pure EM metric.
        score = compute_score_em(
            solution_str=res["solution_str"], ground_truth=gt, format_score=0
        )
        n_answered += int(res["answered"])
        n_forced += int(res["forced"])
        if res["answered"]:
            answered_score_sum += score
        ds = row.get("data_source", "unknown")
        per_ds_scores[ds].append(score)
        details.append({"id": row.get("id"), "data_source": ds, "score": score,
                        "answered": res["answered"], "forced": res["forced"]})
        if traj_f is not None:
            # Full trajectory: the multi-turn response string already contains the
            # interleaved <tool_call>/<tool_response>/<answer> turns verbatim.
            traj_f.write(json.dumps({
                "id": row.get("id"), "data_source": ds, "question": row.get("question"),
                "gold": gt.get("target"), "score": score,
                "answered": res["answered"], "forced": res["forced"], "n_turns": res["n_turns"],
                "prompt": prompt_text, "response": res["response"],
            }, ensure_ascii=False) + "\n")
    if traj_f is not None:
        traj_f.close()

    per_dataset = {
        ds: {"n": len(s), "em": sum(s) / len(s)} for ds, s in sorted(per_ds_scores.items())
    }
    all_scores = [s for v in per_ds_scores.values() for s in v]
    n = len(all_scores)
    # answered-only EM: capability signal with the turn-budget floor removed.
    answered_em = answered_score_sum / max(n_answered, 1)
    summary = {
        "benchmark": "Search-R1 EM (tool-call)",
        "data": args.data,
        "n_samples": n,
        "em_overall": sum(all_scores) / max(n, 1),
        "answered_rate": n_answered / max(n, 1),
        "answered_only_em": answered_em,
        "forced_answer_rate": n_forced / max(n, 1),
        "per_dataset": per_dataset,
        "max_turns": args.max_turns,
        "topk": args.topk,
        "temperature": args.temperature,
        "context_format": args.context_format,
        "force_final_answer": args.force_final_answer,
        "max_response_tokens": args.max_response_tokens,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
