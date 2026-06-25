"""
Async wrapper around Workspace-Bench's synchronous WorkspaceEnv.

WHY THIS EXISTS
    slime's rollout (examples/workspace-bench/trainable_agents.py) drives the
    environment with `await env.reset(...)` / `await env.step(...)`, because a
    single GPU rollout batch fans out *many* trajectories concurrently via
    asyncio.gather. WorkspaceEnv (vendored under ./ws_bench) is fully synchronous,
    and its TERMINAL step makes a BLOCKING Bedrock invoke_model call — the rubric
    judge (see judge.py / ws_bench/envs/base.py::calculate_reward). File-tool steps
    also do (cheap) blocking disk I/O.

    If we called the sync env directly from the event loop, every judge call
    (seconds, cross-region) would block the loop and serialize the whole batch.
    Wrapping each blocking call in run_in_executor(...) hands it to a worker
    thread; boto3 releases the GIL on socket I/O, so N judge calls run truly
    concurrently and the rollout batch overlaps with SGLang generation. This is
    the exact pattern tau-bench used for its blocking Bedrock user-sim.

This keeps the vendored ws_bench package free of any async glue.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ws_bench.envs import get_env as _get_sync_env
from ws_bench.envs.base import WorkspaceEnv
from ws_bench.types import Action, EnvResetResponse, EnvResponse

# Dedicated thread pool for the blocking env calls (file I/O + the terminal
# Bedrock judge invoke). We do NOT use asyncio.to_thread / the loop's default
# executor: that pool caps at min(32, cpu+4) workers and is shared with the rest
# of slime, so a rollout batch of hundreds of concurrent trajectories would
# serialize on ~32 threads (each judge call is seconds). A dedicated, larger pool
# lets the judge calls fan out to match the rollout concurrency. Sized via env;
# threads are cheap here because they spend ~all their time blocked on a socket
# (GIL released) or on disk.
_WS_ENV_WORKERS = int(os.environ.get("WS_ENV_THREAD_WORKERS", "256"))
_EXECUTOR = ThreadPoolExecutor(max_workers=_WS_ENV_WORKERS, thread_name_prefix="ws-env")


class AsyncWorkspaceEnv:
    """Thin async facade over a synchronous WorkspaceEnv.

    Exposes the exact attributes/methods the rollout touches:
        .tools_info, .wiki   (read straight off the sync env)
        await reset(task_index=...) -> EnvResetResponse
        await step(action)          -> EnvResponse
        cleanup()                   -> tear down the rollout's overlay (sync; cheap)
    Every blocking call is offloaded to a thread so the asyncio event loop that
    drives the rollout batch stays responsive.
    """

    def __init__(self, env: WorkspaceEnv):
        self._env = env

    @property
    def tools_info(self):
        return self._env.tools_info

    @property
    def wiki(self):
        return self._env.wiki

    @property
    def task(self):
        return self._env.task

    async def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        # reset materializes the per-rollout workspace overlay (disk I/O).
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_EXECUTOR, self._env.reset, task_index)

    async def step(self, action: Action) -> EnvResponse:
        # Tool actions are local disk ops; the terminal `finish` step also runs
        # the blocking Bedrock rubric judge for reward. All of it in the worker thread.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_EXECUTOR, self._env.step, action)

    def cleanup(self) -> None:
        self._env.cleanup()


def get_async_env(
    env_name: str,
    task_split: str,
    task_index: Optional[int] = None,
    data_dir: Optional[str] = None,
    rollout_uid: Optional[str] = None,
) -> AsyncWorkspaceEnv:
    """Build the sync WorkspaceEnv, then wrap it for async use.

    Signature mirrors ws_bench.envs.get_env so call sites read identically.
    """
    sync_env = _get_sync_env(
        env_name=env_name,
        task_split=task_split,
        task_index=task_index,
        data_dir=data_dir,
        rollout_uid=rollout_uid,
    )
    return AsyncWorkspaceEnv(sync_env)
