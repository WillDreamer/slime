"""Run an inspect_evals task via the Python API with extra_body support.

Why this exists: whx's long-lived sglang servers (:30011/:30012) were launched
with --reasoning-parser qwen3 baked in, which routes all output of these
never-closing-</think> checkpoints into reasoning_content, blinding graders.
sglang accepts a per-request escape hatch {"separate_reasoning": false}, but
the inspect CLI has no --extra-body flag — only GenerateConfig does. So this
runner mirrors benchmarks/inspect/run_inspect.sh's config through the API.

Usage:
  run_inspect_api.py --task inspect_evals/gpqa_diamond --model-name <served> \
      --base-url http://127.0.0.1:30012/v1 --log-dir <dir> [--temperature 0] \
      [--task-arg cot=true] [--limit N] [--message-limit N]
"""

import argparse

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, get_model


def parse_task_arg(s):
    k, v = s.split("=", 1)
    if v.lower() in ("true", "false"):
        return k, v.lower() == "true"
    try:
        return k, int(v)
    except ValueError:
        return k, v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-connections", type=int, default=24)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--message-limit", type=int, default=None)
    p.add_argument("--task-arg", action="append", default=[])
    # Optional separate user-simulator model (tau2 role="user"). When given, tau2
    # runs with this model as the customer instead of self-play on the agent.
    p.add_argument("--user-model-name", default=None)
    p.add_argument("--user-base-url", default=None)
    args = p.parse_args()

    # tau2bench is skipped UNLESS a separate user simulator is provided. Without a
    # real user-sim the RL'd checkpoints self-play into echo loops to the message
    # cap (14h+/domain). With --user-model-name (e.g. GLM-4.7-Flash) the customer
    # is a distinct capable model, so the eval is meaningful — let it run.
    if "tau2" in args.task and not args.user_model_name:
        print(f"[skip] {args.task}: tau2bench disabled (no user simulator); skipping.", flush=True)
        raise SystemExit(0)

    config = GenerateConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
        timeout=args.timeout,
        max_retries=args.max_retries,
        # Disable server-side reasoning splitting so graders see full content.
        extra_body={"separate_reasoning": False},
    )
    model = get_model(
        f"sglang/{args.model_name}",
        base_url=args.base_url,
        api_key="dummy",
        config=config,
    )
    task_args = dict(parse_task_arg(s) for s in args.task_arg)
    kwargs = {}
    if args.limit:
        kwargs["limit"] = args.limit
    if args.message_limit:
        kwargs["message_limit"] = args.message_limit
    # Map the tau2 user-simulator ("user" role) to a distinct served model. Let
    # its glm45 reasoning parser run normally so the customer message is clean.
    if args.user_model_name:
        user_model = get_model(
            f"sglang/{args.user_model_name}",
            base_url=args.user_base_url or args.base_url,
            api_key="dummy",
            config=GenerateConfig(
                temperature=0.0,
                max_tokens=args.max_tokens,
                max_connections=args.max_connections,
                timeout=args.timeout,
                max_retries=args.max_retries,
            ),
        )
        kwargs["model_roles"] = {"user": user_model}
    logs = inspect_eval(
        args.task, model=model, task_args=task_args, log_dir=args.log_dir, **kwargs
    )
    ok = all(l.status == "success" for l in logs)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
