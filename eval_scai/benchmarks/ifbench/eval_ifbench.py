"""Standalone IFBench evaluation against an OpenAI-compatible endpoint.

Data   : IFBench_eval.jsonl (300 prompts; allenai/IFBench held-out instruction set)
Scorer : slime.rollout.rm_hub.ifbench.compute_ifbench_reward — the exact scorer
         used as the eval reward during RL training (official IFBench checkers,
         strict mode, prompt-level all-instructions-followed accuracy).

Usage:
  python eval_ifbench.py --base-url http://127.0.0.1:8400/v1 --model <name> \
      --data IFBench_eval.jsonl --output results.json
"""

import argparse
import asyncio
import json
import sys
import time

import aiohttp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True, help="OpenAI-compatible base URL (.../v1)")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--data", required=True, help="IFBench_eval.jsonl path")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--slime-root", default="/xuanwu-tank/north/xw27/multi/slime_scai")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--trajectory-output", default=None,
                   help="if set, write per-prompt full prompt+response to this jsonl path")
    return p.parse_args()


async def query_one(session, sem, args, row, idx):
    messages = row["prompt"]
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(
                    f"{args.base_url}/chat/completions", json=payload
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    msg = data["choices"][0]["message"]
                    # Thinking models behind a reasoning parser put text in
                    # reasoning_content; prefer content, fall back to reasoning.
                    content = msg.get("content") or ""
                    if not content.strip():
                        content = msg.get("reasoning_content") or ""
                    return idx, content
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    print(f"[ifbench] sample {idx} failed: {e}", file=sys.stderr)
                    return idx, ""
                await asyncio.sleep(2**attempt)
    return idx, ""


def load_ifbench_scorer(slime_root):
    """Load slime's ifbench scorer by file path, bypassing the slime package
    __init__ chain (which imports ray — not available in this env)."""
    import importlib.util

    path = f"{slime_root}/slime/rollout/rm_hub/ifbench.py"
    spec = importlib.util.spec_from_file_location("ifbench_scorer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_ifbench_reward


async def main():
    args = parse_args()
    compute_ifbench_reward = load_ifbench_scorer(args.slime_root)

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[ifbench] {len(rows)} prompts, model={args.model}")

    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=1800)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[query_one(session, sem, args, row, i) for i, row in enumerate(rows)]
        )

    responses = {idx: text for idx, text in results}
    traj_f = open(args.trajectory_output, "w") if args.trajectory_output else None
    details, n_correct, n_empty = [], 0, 0
    for i, row in enumerate(rows):
        resp = responses.get(i, "")
        if not resp.strip():
            n_empty += 1
        score = compute_ifbench_reward(resp, row.get("label"), metadata=row.get("metadata"))
        n_correct += int(score > 0)
        details.append(
            {
                "idx": i,
                "record_id": row.get("metadata", {}).get("record_id"),
                "instruction_id_list": row.get("metadata", {}).get("instruction_id_list"),
                "score": score,
                "response_len": len(resp),
            }
        )
        if traj_f is not None:
            traj_f.write(json.dumps({
                "idx": i,
                "record_id": row.get("metadata", {}).get("record_id"),
                "instruction_id_list": row.get("metadata", {}).get("instruction_id_list"),
                "score": score,
                "prompt": row["prompt"], "response": resp,
            }, ensure_ascii=False) + "\n")
    if traj_f is not None:
        traj_f.close()

    accuracy = n_correct / max(len(rows), 1)
    summary = {
        "benchmark": "IFBench (strict, prompt-level)",
        "model": args.model,
        "n_samples": len(rows),
        "n_correct": n_correct,
        "n_empty_responses": n_empty,
        "accuracy": accuracy,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
