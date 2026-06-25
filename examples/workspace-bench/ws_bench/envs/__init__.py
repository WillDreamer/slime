from typing import Optional

from ws_bench.envs.base import WorkspaceEnv


def get_env(
    env_name: str,
    task_split: str,
    task_index: Optional[int] = None,
    data_dir: Optional[str] = None,
    rollout_uid: Optional[str] = None,
) -> WorkspaceEnv:
    """Factory mirroring tau_bench.envs.get_env, trimmed of user-sim params.

    Workspace-Bench has a single environment type ("workspace"); env_name is
    accepted for parity with the tau call sites but only "workspace" is valid.
    """
    if env_name == "workspace":
        return WorkspaceEnv(
            task_split=task_split,
            task_index=task_index,
            data_dir=data_dir,
            rollout_uid=rollout_uid,
        )
    raise ValueError(f"Unknown environment: {env_name}")
