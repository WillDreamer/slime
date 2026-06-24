"""BrowseComp-Plus agent client using the SEARCH-TASK parsing format.

Drop-in alternative to the official search_agent/qwen_client.py. Everything in
the BrowseComp-Plus pipeline stays official — the dataset (decrypted queries),
the retriever (searcher/search_r1_server.py `/retrieve`), the judge
(scripts_evaluation/evaluate_with_openai.py) and its metrics (accuracy, retrieval
recall, citation precision/recall, calibration). The ONLY thing that changes is
how the agent talks to the model:

  * official qwen_client.py: native OpenAI function-calling via qwen-agent/MCP.
  * THIS client:             the raw tool-call TEXT format the search benchmark
                             uses (benchmarks/search/eval_search.py) and that the
                             willhx/*-Search checkpoints were trained on:

        <tool_call>
        {"name": "search", "arguments": {"query": "..."}}
        </tool_call>
        ... -> <tool_response> ... </tool_response> -> ... -> <answer> ... </answer>

It drives the served sglang model directly via `/generate` (same as eval_search.py)
and writes the SAME run_<ts>.json record the official judge reads:
  {metadata, query_id, tool_call_counts, status, retrieved_docids,
   result:[{type:"tool_call"|"output_text", tool_name, arguments, output}, ...]}
The judge takes the LAST result item (type=="output_text") as the graded response
and reads top-level retrieved_docids for recall, so both must be populated here.

The user query keeps the official QUERY_TEMPLATE_NO_GET_DOCUMENT (asks for inline
[docid] citations + Exact Answer + Confidence) so citation/calibration metrics
stay meaningful; only the interaction is rewrapped in the tool-call tags.

Usage:
  python browsecomp_toolcall_client.py \
      --query topics-qrels/queries.tsv \
      --model qwen-8b-base --model-server http://127.0.0.1:7000 \
      --retriever-url http://127.0.0.1:8601/retrieve \
      --tokenizer <hf ckpt> --output-dir runs/bm25/<ckpt_name>
"""

import argparse
import asyncio
import csv
import datetime
import json
import os
import re
import sys

import aiohttp
from transformers import AutoTokenizer

# --- official BrowseComp-Plus query template (search_agent/prompts.py) --------
QUERY_TEMPLATE_NO_GET_DOCUMENT = (
    "You are a deep research agent. You need to answer the given question by "
    "interacting with a search engine, using the search tool provided. Please "
    "perform reasoning and use the tool step by step, in an interleaved manner. "
    "You may use the search tool multiple times.\n\n"
    "Question: {Question}\n\n"
    "Your response should be in the following format:\n"
    "Explanation: {{your explanation for your final answer. For this explanation "
    "section only, you should cite your evidence documents inline by enclosing "
    "their docids in square brackets [] at the end of sentences. For example, [20].}}\n"
    "Exact Answer: {{your succinct, final answer}}\n"
    "Confidence: {{your confidence score between 0% and 100% for your answer}}"
)

# --- tool-call format (identical to benchmarks/search/eval_search.py) ---------
SEARCH_TOOL_DESC = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information using a search engine",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}

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


def parse_args():
    p = argparse.ArgumentParser(description="BrowseComp-Plus tool-call agent (search-task format).")
    p.add_argument("--query", default="topics-qrels/queries.tsv",
                   help="TSV (qid<tab>question) or a single question string")
    p.add_argument("--model", required=True, help="served-model name (for the run record metadata)")
    p.add_argument("--model-server", required=True, help="sglang server root (no /v1) for /generate")
    p.add_argument("--retriever-url", required=True, help="official search_r1_server.py /retrieve endpoint")
    p.add_argument("--tokenizer", required=True, help="HF checkpoint for the chat template")
    p.add_argument("--output-dir", default="runs/bm25/toolcall")
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="PER-TURN generation cap. Was 10000, which let the model "
                        "monologue one whole turn to the length cap without ever "
                        "emitting </tool_call>/</answer>; 2048 matches the training "
                        "per-turn budget so it commits to an action.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--context-window-k", type=int, default=2)
    p.add_argument("--max-docs-compressed", type=int, default=1)
    p.add_argument("--max-chars-per-doc-compressed", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0, help="cap number of queries (0 = all)")
    # ---- train/eval alignment knobs (see benchmarks/search/eval_search.py) ----
    p.add_argument("--context-format", choices=["chat", "legacy"], default="chat",
                   help="'chat' re-renders the conversation via apply_chat_template "
                        "(assistant turns closed with <|im_end|>, tool results as user "
                        "turns) — byte-aligned with the no_drift training rollout the 8B "
                        "checkpoints used. 'legacy' is the old drift-y string-concat path.")
    p.add_argument("--force-final-answer", dest="force_final_answer",
                   action="store_true", default=True,
                   help="if the rollout ends without an <answer>, do one constrained "
                        "generation (stop=</answer>) so the model still concludes (default on)")
    p.add_argument("--no-force-final-answer", dest="force_final_answer", action="store_false")
    p.add_argument("--max-response-tokens", type=int, default=4096,
                   help="total response-token budget across ALL turns (matches training "
                        "--rollout-max-response-len 4096). 0 = unlimited (per-turn cap only).")
    return p.parse_args()


# ---- parsing (verbatim from eval_search.py) ----------------------------------
def postprocess_responses(resp: str) -> str:
    if "</tool_call>" in resp:
        return resp.split("</tool_call>")[0] + "</tool_call>"
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    bare = r'\{[^{}]*"name"\s*:\s*"search"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
    m = re.search(bare, resp)
    return resp[: m.end()] if m else resp


def _extract_search_query(data):
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict) or data.get("name") != "search":
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
    return ("search", query.strip()) if isinstance(query, str) else (None, "")


def _parse_tool_call_block(block: str):
    m = re.search(r"<tool_call>(.*?)</tool_call>", block, re.DOTALL)
    if not m:
        return None, ""
    try:
        data = json.loads(m.group(1).strip())
    except Exception:  # noqa: BLE001
        return None, ""
    return _extract_search_query(data)


def _parse_bare_json_tool_call(text: str):
    pattern = r'\{[^{}]*"name"\s*:\s*"search"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
    m = re.search(pattern, text)
    if not m:
        pattern = r'\{[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*"name"\s*:\s*"search"[^{}]*\}'
        m = re.search(pattern, text)
    if not m:
        return None, ""
    try:
        data = json.loads(m.group(0))
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
    m = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if m:
        return "answer", m.group(1).strip()
    return None, ""


# ---- retrieval (official search_r1_server.py /retrieve: single query) --------
def _passages2string(retrieval_result):
    """Format official /retrieve docs and collect their docids."""
    formatted, docids = "", []
    for idx, item in enumerate(retrieval_result):
        doc = item.get("document", {})
        title = doc.get("title", "")
        text = doc.get("text", "")
        docid = str(item.get("docid"))
        # Show the docid so the model can cite it in <answer> (citation metrics).
        formatted += f"Doc {idx+1}(docid: {docid}, Title: {title}) {text}\n"
        docids.append(docid)
    return formatted, docids


async def retrieve(session, sem, url, query, topk):
    payload = {"query": query, "topk": topk}
    async with sem:
        for attempt in range(5):
            try:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return _passages2string(data["result"])
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    raise
                await asyncio.sleep(2**attempt)


# ---- context assembly (verbatim from eval_search.py) -------------------------
def compress_search_result(text, max_docs, max_chars_per_doc):
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
    return (
        f"\n<|im_start|>user\n<tool_response>\n{search_result}\n</tool_response><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_context(prompt_text, turns, args, add_final_generation_prompt=True):
    num_search_turns = sum(1 for t in turns if t["search_result"] is not None)
    context = prompt_text
    search_count = 0
    for i, turn in enumerate(turns):
        is_last = i == len(turns) - 1
        context += turn["text"]
        if turn["search_result"] is not None:
            search_count += 1
            within = search_count > num_search_turns - args.context_window_k
            sr = turn["search_result"] if within else compress_search_result(
                turn["search_result"], args.max_docs_compressed, args.max_chars_per_doc_compressed
            )
            context += _format_tool_response(sr)
        elif not is_last or add_final_generation_prompt:
            context += "\n<|im_start|>assistant\n"
    return context


# Forced-answer nudge: NO literal <answer>/</answer> tags (see eval_search.py).
FORCE_ANSWER_NUDGE = (
    "You have used all available searches and cannot search further. "
    "Based on the information gathered so far, give your final answer now."
)


def render_chat(tokenizer, tools, base_messages, turns, add_generation_prompt=True,
                final_user=None):
    """No-drift context construction (see benchmarks/search/eval_search.py.render_chat):
    re-render the whole conversation through apply_chat_template each turn so
    assistant turns are closed with <|im_end|> and tool results are proper user
    turns — byte-aligned with the no_drift rollout the 8B checkpoints trained on.
    Search results render in full (no compression)."""
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


async def _generate(session, gen_sem, args, context, stop, max_new):
    payload = {
        "text": context,
        "sampling_params": {
            "max_new_tokens": max_new,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stop": stop,
            "no_stop_trim": True,
        },
    }
    async with gen_sem:
        async with session.post(f"{args.model_server}/generate", json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


# ---- one rollout -------------------------------------------------------------
async def run_sample(session, gen_sem, ret_sem, args, base_messages, tokenizer, tools, prompt_text):
    """Returns (steps, final_output, status). steps build the official result[]."""
    turns, steps = [], []
    final_text, status = "", "incomplete"
    used_tokens = 0

    def build_ctx(add_gen=True, final_user=None):
        if args.context_format == "chat":
            return render_chat(tokenizer, tools, base_messages, turns, add_gen, final_user)
        return build_context(prompt_text, turns, args, add_final_generation_prompt=add_gen)

    def remaining():
        if args.max_response_tokens and args.max_response_tokens > 0:
            return max(0, args.max_response_tokens - used_tokens)
        return args.max_new_tokens

    answered = False
    for _ in range(args.max_turns):
        budget = min(args.max_new_tokens, remaining())
        if budget <= 0:
            break
        output = await _generate(session, gen_sem, args, build_ctx(add_gen=True),
                                 ["</tool_call>", "</answer>"], budget)
        cur = postprocess_responses(output["text"])
        used_tokens += len(tokenizer(cur, add_special_tokens=False)["input_ids"])
        finish = output["meta_info"]["finish_reason"]["type"]

        action, content = postprocess_predictions(cur)
        if action == "answer":
            turns.append({"text": cur, "search_result": None})
            final_text, status, answered = cur, "completed", True
            break
        if finish == "length":
            turns.append({"text": cur, "search_result": None})
            final_text = cur
            break
        if action == "search":
            try:
                docs, docids = await retrieve(session, ret_sem, args.retriever_url, content, args.topk)
            except Exception as e:  # noqa: BLE001
                print(f"[bcp-toolcall] retrieval failed: {e}", file=sys.stderr)
                turns.append({"text": cur, "search_result": None})
                final_text = cur
                break
            steps.append({"query": content, "output": docs.strip(), "docids": docids})
            turns.append({"text": cur, "search_result": docs.strip()})
        else:
            turns.append({"text": cur, "search_result": None})
            final_text = cur

    # ---- forced final answer (chat format only) ------------------------------
    if (args.force_final_answer and not answered and args.context_format == "chat"
            and turns):
        # Dedicated budget independent of the (possibly exhausted) search budget.
        ans_budget = min(args.max_new_tokens, 1024)
        output = await _generate(
            session, gen_sem, args,
            build_ctx(add_gen=True, final_user=FORCE_ANSWER_NUDGE),
            ["</answer>"], ans_budget,
        )
        cur = output["text"]
        if "</answer>" in cur:
            cur = cur.split("</answer>")[0] + "</answer>"
        turns.append({"text": cur, "search_result": None})
        final_text = cur
        if "<answer>" in cur:
            status = "completed"

    # final answer text = inner <answer>…</answer> if present, else last assistant text
    m = re.search(r"<answer>(.*?)</answer>", final_text, re.DOTALL)
    final_output = m.group(1).strip() if m else final_text.strip()
    return steps, final_output, status


def write_record(out_dir, args, qid, steps, final_output, status):
    """Emit the official run_<ts>.json record consumed by evaluate_with_openai.py."""
    result, tool_counts, retrieved = [], {}, set()
    for s in steps:
        result.append({
            "type": "tool_call", "tool_name": "search",
            "arguments": {"query": s["query"]}, "output": s["output"],
        })
        tool_counts["search"] = tool_counts.get("search", 0) + 1
        retrieved.update(d for d in s["docids"] if d and d != "None")
    # the judge grades result[-1] when its type == "output_text"
    result.append({"type": "output_text", "tool_name": None, "arguments": None, "output": final_output})

    record = {
        "metadata": {"model": args.model, "max_output_tokens": args.max_new_tokens, "output_dir": out_dir},
        "query_id": str(qid),
        "tool_call_counts": tool_counts,
        "status": status,
        "retrieved_docids": sorted(retrieved),
        "result": result,
    }
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    path = os.path.join(out_dir, f"run_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def load_queries(query_arg, limit):
    if query_arg.endswith(".tsv"):
        rows = []
        with open(query_arg, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 2:
                    rows.append((row[0].strip(), row[1].strip()))
    else:
        rows = [("1", query_arg.strip())]
    if limit > 0:
        rows = rows[:limit]
    return rows


def already_processed(out_dir):
    done = set()
    if not os.path.isdir(out_dir):
        return done
    for name in os.listdir(out_dir):
        if name.endswith(".json"):
            try:
                with open(os.path.join(out_dir, name), encoding="utf-8") as f:
                    qid = json.load(f).get("query_id")
                    if qid is not None:
                        done.add(str(qid))
            except Exception:  # noqa: BLE001
                continue
    return done


async def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    queries = load_queries(args.query, args.limit)
    done = already_processed(args.output_dir)
    remaining = [(qid, q) for qid, q in queries if qid not in done]
    print(f"[bcp-toolcall] {len(remaining)} queries to run ({len(done)} already done)")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Prompt in the tool-call format the model was trained on (matches
    # benchmarks/search/eval_search.py). The official QUERY_TEMPLATE's prose
    # "Explanation/Exact Answer/Confidence" block is dropped — it overrode the
    # tool-call behaviour and made the model narrate instead of emitting
    # <tool_call> (≈90% incomplete). One citation line is kept so the official
    # citation metrics still work.
    citation = (
        "Each search result is labelled with its docid. In your final <answer>, "
        "cite the documents that support it by docid in square brackets, e.g. [20].\n"
    )

    tools = [SEARCH_TOOL_DESC]

    def make_base_messages(question):
        user = TOOL_CALLING_INSTRUCTION + citation + "\nQuestion: " + question
        return [{"role": "user", "content": user}]

    gen_sem = asyncio.Semaphore(args.concurrency)
    ret_sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)

    async def one(qid, question):
        base_messages = make_base_messages(question)
        prompt = tokenizer.apply_chat_template(
            base_messages, tokenize=False, add_generation_prompt=True, tools=tools,
        )
        steps, final_output, status = await run_sample(
            session, gen_sem, ret_sem, args, base_messages, tokenizer, tools, prompt
        )
        path = write_record(args.output_dir, args, qid, steps, final_output, status)
        return qid, status, path

    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[one(qid, q) for qid, q in remaining], return_exceptions=True
        )
    ok = sum(1 for r in results if not isinstance(r, Exception))
    print(f"[bcp-toolcall] wrote {ok}/{len(remaining)} records to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
