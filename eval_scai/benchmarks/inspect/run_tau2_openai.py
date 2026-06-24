"""Run inspect_evals tau2_* against already-running OpenAI-compatible servers.

Why not run_inspect_api.py: that uses inspect's *sglang* provider, which (in
inspect_ai 0.3.x) tries to LAUNCH its own sglang server instead of connecting to
an existing one (observed: "Existing SGLang server not found. Starting new
server ..."). The openai provider connects cleanly to a given base_url per model,
which is what we want for tau2's two distinct endpoints:

  agent  = served checkpoint  @ --base-url        (e.g. :7000/v1)
  user   = GLM user simulator @ --user-base-url   (e.g. :7006/v1, role="user")

The GLM server is launched with --reasoning-parser glm45, so its reasoning is
split server-side and the customer message content stays clean. The agent
servers run no reasoning parser, so <think> stays in content for the grader.

Usage:
  run_tau2_openai.py --task inspect_evals/tau2_retail \
     --model-name qwen-8b-base --base-url http://127.0.0.1:7000/v1 \
     --user-model-name GLM-4.7-Flash --user-base-url http://127.0.0.1:7006/v1 \
     --log-dir <dir> --temperature 0.7 --message-limit 50 [--limit N]
"""

import argparse

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, get_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--user-model-name", required=True)
    p.add_argument("--user-base-url", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-connections", type=int, default=16)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--message-limit", type=int, default=50)
    args = p.parse_args()

    # responses_api=False: force the Chat Completions endpoint. inspect's openai
    # provider otherwise auto-probes /responses and (wrongly) decides sglang
    # supports it, then every tool request 500s. max_tokens caps per-turn
    # generation (these checkpoints ramble to the context limit otherwise).
    agent = get_model(
        f"openai/{args.model_name}",
        base_url=args.base_url,
        api_key="dummy",
        responses_api=False,
        config=GenerateConfig(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_connections=args.max_connections,
            timeout=args.timeout,
            max_retries=args.max_retries,
        ),
    )
    # User simulator at temperature 0 for stable customer behavior.
    user_model = get_model(
        f"openai/{args.user_model_name}",
        base_url=args.user_base_url,
        api_key="dummy",
        responses_api=False,
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=args.max_tokens,
            max_connections=args.max_connections,
            timeout=args.timeout,
            max_retries=args.max_retries,
        ),
    )

    kwargs = {"model_roles": {"user": user_model}, "message_limit": args.message_limit}
    if args.limit:
        kwargs["limit"] = args.limit

    logs = inspect_eval(args.task, model=agent, log_dir=args.log_dir, **kwargs)
    ok = all(l.status == "success" for l in logs)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
