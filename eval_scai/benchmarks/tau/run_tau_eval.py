"""tau-bench (v1) evaluation driver — *training-aligned*.

Why this exists
---------------
The eval folder already evaluates tau via `inspect_evals/tau2_*` (tau2-bench,
inspect-ai harness). That is a *different benchmark on a different harness* from
what slime trains/evals on. The training pipeline (examples/tau-bench) measures
the model by calling **the exact same rollout function it trains on**:

    --custom-generate-function-path generate_with_tau.generate
    --eval-prompt-data retail-dev .../retail_test_tasks.jsonl   # EVAL_ARGS

i.e. slime feeds the custom `generate_with_tau.generate(args, sample, sp)` a
test-split task index per sample and averages the env reward. This driver
reproduces that path verbatim so the eval-folder number is apples-to-apples with
the training rollout reward (same tau_bench env, same GLM user simulator, same
ToolCallingAgent / qwen tool-call format, same reward + format penalties).

This runs tau1 *alongside* the inspect tau2 eval — it does not replace it.

How it works
------------
1. Put `examples/tau-bench` on sys.path and import `generate_with_tau` (gwt).
   Importing it pulls in slime (`GenerateState`, `post`, `Sample`) and the
   sibling training modules (`trainable_agents`, `openai_tool_adapter`), so this
   must run in the same env as training (the `slime` container python with
   PYTHONPATH including the slime root + Megatron-LM).
2. Build the minimal `args` Namespace that `GenerateState` / `init_http_client` /
   `gwt.generate` / `asolve` read, pointing the sglang `/generate` endpoint at
   the eval server.
3. Override `gwt.tau_config` (copy of TAU_CONFIGS) with the requested env +
   `task_split=test` so the eval uses held-out tasks. Everything else (agent
   strategy, user simulator model/provider/strategy) stays identical to training.
4. Run every test task concurrently through `gwt.generate`, then aggregate.

The user simulator is reached by tau_bench via the OpenAI-compatible provider,
so the caller must export OPENAI_API_BASE / OPENAI_API_KEY to point at the GLM
user-sim server (run_tau.sh does this).

Usage:
  run_tau_eval.py --hf-checkpoint <CKPT> --router-ip 127.0.0.1 --router-port 8400 \
      --env retail --task-split test --output run1/tau1_retail.json \
      [--tasks-file retail_test_tasks.jsonl] [--limit N] \
      [--temperature 0.7] [--max-new-tokens 2048] [--concurrency 16] \
      [--user-model zai-org/GLM-4.7-Flash] [--user-provider openai] \
      [--tau-example-dir <repo>/examples/tau-bench]
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
from argparse import Namespace

logging.basicConfig(level=os.environ.get("TAU1_LOG_LEVEL", "INFO"))
logger = logging.getLogger("tau1_eval")


def _default_tau_example_dir() -> str:
    # eval_scai/benchmarks/tau/ -> repo root is three levels up; training
    # modules live in <repo>/examples/tau-bench.
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(repo, "examples", "tau-bench")


def build_args(a) -> Namespace:
    """Minimal Namespace satisfying GenerateState, init_http_client, gwt.generate
    and TrainableAgentMixin.asolve. Field names mirror slime's rollout args."""
    return Namespace(
        # --- where to send /generate requests (asolve reads these directly) ---
        sglang_router_ip=a.router_ip,
        sglang_router_port=a.router_port,
        # --- tokenizer source for GenerateState ---
        hf_checkpoint=a.hf_checkpoint,
        # --- GenerateState semaphore + init_http_client concurrency math ---
        sglang_server_concurrency=a.concurrency,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        # --- GenerateState.sampling_params (unused by gwt.generate, which takes
        #     the sampling_params arg we pass, but required at construction) ---
        rollout_temperature=a.temperature,
        rollout_top_p=a.top_p,
        rollout_top_k=a.top_k,
        rollout_max_response_len=a.max_new_tokens,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_dp_size=1,
        sglang_enable_deterministic_inference=False,
        # --- misc flags read along the path ---
        use_distributed_post=False,
        partial_rollout=False,
    )


async def run_one(gwt, args, sampling_params, index, sem, Sample):
    async with sem:
        sample = Sample(index=index, prompt=str(index))
        sample.status = Sample.Status.PENDING
        try:
            out = await gwt.generate(args, sample, dict(sampling_params))
        except Exception as e:  # one flaky task must not kill the whole eval
            logger.warning(f"task {index} failed: {e.__class__.__name__}: {e}")
            return {"index": index, "error": f"{e.__class__.__name__}: {e}",
                    "reward": 0.0, "answer_correct": 0, "status": "aborted"}
        info = out.metadata or {}
        return {
            "index": index,
            "reward": float(out.reward) if out.reward is not None else 0.0,
            "answer_correct": int(info.get("answer_correct", int((out.reward or 0) > 0))),
            "raw_task_reward": float(info.get("raw_task_reward", out.reward or 0.0)),
            "status": str(getattr(out.status, "value", out.status)),
            "num_turns": info.get("num_turns"),
            "num_tool_calls": info.get("num_tool_calls"),
            "format_ok": info.get("format_ok"),
            "env_done": info.get("env_done"),
            "is_aborted": info.get("is_aborted"),
        }


def resolve_indices(gwt, a, cfg):
    """Test-task indices: prefer the same jsonl training eval reads (input-key
    'index'); otherwise enumerate all tasks in the split via tau_bench."""
    if a.tasks_file and os.path.exists(a.tasks_file):
        idxs = []
        with open(a.tasks_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idxs.append(int(json.loads(line)["index"]))
        logger.info(f"loaded {len(idxs)} task indices from {a.tasks_file}")
    else:
        # Count tasks straight from the task module. We deliberately avoid
        # gwt.get_env() here: constructing a tau_bench Env immediately resets it,
        # which fires a user-simulator LLM call — wasteful and a single point of
        # failure just to learn the task count. tasks_<split> exposes one
        # uppercase list (retail/test -> TASKS_TEST, airline/test -> TASKS).
        import importlib
        n = None
        try:
            mod = importlib.import_module(f"tau_bench.envs.{cfg['env']}.tasks_{cfg['task_split']}")
            for pref in ("TASKS_TEST", f"TASKS_{cfg['task_split'].upper()}", "TASKS"):
                if hasattr(mod, pref) and isinstance(getattr(mod, pref), list):
                    n = len(getattr(mod, pref)); break
            if n is None:  # fall back to first uppercase list symbol
                for s in dir(mod):
                    if s.isupper() and isinstance(getattr(mod, s), list):
                        n = len(getattr(mod, s)); break
        except Exception as e:
            logger.warning(f"task-count via module failed ({e}); falling back to get_env")
        if n is None:
            env = gwt.get_env(
                env_name=cfg["env"], user_strategy=cfg["user_strategy"],
                user_model=cfg["user_model"], user_provider=cfg["user_model_provider"],
                task_split=cfg["task_split"], task_index=0,
            )
            n = len(env.tasks)
        idxs = list(range(n))
        logger.info(f"enumerated {n} tasks for {cfg['env']}/{cfg['task_split']}")
    if a.limit:
        idxs = idxs[: a.limit]
    return idxs


async def amain(a):
    sys.path.insert(0, a.tau_example_dir)
    import generate_with_tau as gwt  # noqa: E402  (path-dependent import)
    from slime.utils.http_utils import init_http_client  # noqa: E402
    from slime.utils.types import Sample  # noqa: E402

    # slime's trainable_agents awaits env.reset()/env.step(), but the JD-ETH
    # tau_bench (both branches) ships a *sync* Env. Wrap it so those methods are
    # awaitable; offload the blocking user-simulator (litellm) call to a thread so
    # concurrent tasks don't serialize on the event loop.
    class _AsyncEnvProxy:
        def __init__(self, env):
            self._env = env

        def __getattr__(self, name):
            return getattr(self._env, name)

        async def reset(self, task_index=None):
            return await asyncio.to_thread(self._env.reset, task_index=task_index)

        async def step(self, action):
            return await asyncio.to_thread(self._env.step, action)

    _real_get_env = gwt.get_env
    gwt.get_env = lambda *aa, **kw: _AsyncEnvProxy(_real_get_env(*aa, **kw))

    args = build_args(a)
    init_http_client(args)  # sets the module-global httpx client `post` uses

    # Mirror training's RunConfig exactly, changing only env + split (+ optional
    # user-sim overrides). Copying TAU_CONFIGS guarantees identical agent/user
    # behavior to the training rollout.
    cfg = dict(gwt.TAU_CONFIGS)
    cfg["env"] = a.env
    cfg["task_split"] = a.task_split
    if a.user_model:
        cfg["user_model"] = a.user_model
    if a.user_provider:
        cfg["user_model_provider"] = a.user_provider
    gwt.tau_config = gwt.RunConfig(**cfg)
    logger.info(f"tau1 eval: env={cfg['env']} split={cfg['task_split']} "
                f"user_model={cfg['user_model']} provider={cfg['user_model_provider']}")

    # Eval sampling params (match EVAL_ARGS: temp 0.7, max_response_len 2048).
    sampling_params = {
        "temperature": a.temperature,
        "top_p": a.top_p,
        "top_k": a.top_k,
        "max_new_tokens": a.max_new_tokens,
    }

    indices = resolve_indices(gwt, a, cfg)
    if not indices:
        raise SystemExit("no tasks to evaluate")

    sem = asyncio.Semaphore(a.concurrency)
    results = await asyncio.gather(
        *[run_one(gwt, args, sampling_params, i, sem, Sample) for i in indices]
    )

    rewards = [r["reward"] for r in results]
    correct = [r["answer_correct"] for r in results]
    n = len(results)
    summary = {
        "env": a.env,
        "task_split": a.task_split,
        "n": n,
        # `accuracy` = task pass rate (raw env success), the comparable headline
        # number. `avg_reward` = the shaped reward used in training (includes the
        # format/truncation penalties), for direct comparison to rollout reward.
        "accuracy": (sum(correct) / n) if n else 0.0,
        "avg_reward": (statistics.mean(rewards)) if n else 0.0,
        "num_aborted": sum(1 for r in results if r.get("status") == "aborted"),
        "num_truncated": sum(1 for r in results if r.get("status") == "truncated"),
        "num_errors": sum(1 for r in results if "error" in r),
        "user_model": cfg["user_model"],
        "temperature": a.temperature,
    }
    out = {"summary": summary, "tasks": results}

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(
        f"tau1 {a.env}/{a.task_split}: accuracy={summary['accuracy']:.4f} "
        f"avg_reward={summary['avg_reward']:.4f} n={n} "
        f"aborted={summary['num_aborted']} -> {a.output}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-checkpoint", required=True, help="HF checkpoint dir (tokenizer source)")
    p.add_argument("--router-ip", default="127.0.0.1", help="sglang server host (/generate)")
    p.add_argument("--router-port", type=int, required=True, help="sglang server port (/generate)")
    p.add_argument("--env", default="retail", help="tau-bench domain: retail | airline")
    p.add_argument("--task-split", default="test", help="train | test | dev")
    p.add_argument("--tasks-file", default=None,
                   help="jsonl with one {'index': N} per line (tau1_mock output); "
                        "falls back to enumerating all split tasks if absent")
    p.add_argument("--limit", type=int, default=None, help="cap #tasks (smoke test)")
    p.add_argument("--temperature", type=float, default=0.7)   # matches eval-temperature
    p.add_argument("--max-new-tokens", type=int, default=2048)  # matches eval-max-response-len
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--user-model", default=None, help="override user-sim model (default: TAU_CONFIGS)")
    p.add_argument("--user-provider", default=None, help="override user-sim provider")
    p.add_argument("--tau-example-dir", default=_default_tau_example_dir(),
                   help="path to <repo>/examples/tau-bench (training modules)")
    p.add_argument("--output", required=True, help="output JSON path (e.g. runN/tau1_retail.json)")
    a = p.parse_args()
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
