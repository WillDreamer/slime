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
#     blocks, exactly like MessageContextWindowManager. With --max-turns <= 2
#     this never triggers, so the canonical eval is unaffected.
#
# Knob to watch: the training config uses max_turns=5; this harness defaults to
# --max-turns 2 (matches the RL *eval* cadence). Set --max-turns 5 to match the
# training rollout depth exactly.
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


async def run_sample(session, gen_sem, ret_sem, args, prompt_text):
    """One multi-turn tool-call rollout. Returns the accumulated response string."""
    turns: list[dict] = []
    for _turn in range(args.max_turns):
        context = build_context(prompt_text, turns, args, add_final_generation_prompt=True)
        payload = {
            "text": context,
            "sampling_params": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "stop": ["</tool_call>", "</answer>"],
                "no_stop_trim": True,  # keep the matched stop tag in the output
            },
        }
        async with gen_sem:
            async with session.post(f"{args.base_url}/generate", json=payload) as resp:
                resp.raise_for_status()
                output = await resp.json()

        cur = postprocess_responses(output["text"])

        if output["meta_info"]["finish_reason"]["type"] == "length":
            turns.append({"text": cur, "search_result": None})
            break

        action, content = postprocess_predictions(cur)
        if action == "answer":
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

    # The scored solution string is prompt_text + this response (the harness
    # rebuilds the no-trailing-prompt context, same as the training reward_func).
    full_context = build_context(prompt_text, turns, args, add_final_generation_prompt=False)
    return full_context[len(prompt_text):]


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
    prompts = [
        tokenizer.apply_chat_template(
            clean_instruction_in_messages([dict(m) for m in row["prompt"]]),
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
        )
        for _, row in df.iterrows()
    ]

    gen_sem = asyncio.Semaphore(args.concurrency)
    ret_sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        responses = await asyncio.gather(
            *[run_sample(session, gen_sem, ret_sem, args, p) for p in prompts]
        )

    traj_f = open(args.trajectory_output, "w") if args.trajectory_output else None
    per_ds_scores, details = defaultdict(list), []
    for (_, row), prompt_text, response in zip(df.iterrows(), prompts, responses):
        gt = row["reward_model"]["ground_truth"]
        gt = {k: (list(v) if hasattr(v, "tolist") or isinstance(v, (list, tuple)) else v) for k, v in dict(gt).items()}
        # Same call as the RL reward_func, but format_score=0 -> pure EM metric.
        score = compute_score_em(
            solution_str=prompt_text + response, ground_truth=gt, format_score=0
        )
        ds = row.get("data_source", "unknown")
        per_ds_scores[ds].append(score)
        details.append({"id": row.get("id"), "data_source": ds, "score": score})
        if traj_f is not None:
            # Full trajectory: the multi-turn response string already contains the
            # interleaved <tool_call>/<tool_response>/<answer> turns verbatim.
            traj_f.write(json.dumps({
                "id": row.get("id"), "data_source": ds, "question": row.get("question"),
                "gold": gt.get("target"), "score": score,
                "prompt": prompt_text, "response": response,
            }, ensure_ascii=False) + "\n")
    if traj_f is not None:
        traj_f.close()

    per_dataset = {
        ds: {"n": len(s), "em": sum(s) / len(s)} for ds, s in sorted(per_ds_scores.items())
    }
    all_scores = [s for v in per_ds_scores.values() for s in v]
    summary = {
        "benchmark": "Search-R1 EM (tool-call)",
        "data": args.data,
        "n_samples": len(all_scores),
        "em_overall": sum(all_scores) / max(len(all_scores), 1),
        "per_dataset": per_dataset,
        "max_turns": args.max_turns,
        "topk": args.topk,
        "temperature": args.temperature,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
