"""Core types for the Workspace-Bench env — mirrors tau_bench/types.py.

The symbol names (Action, RESPOND_ACTION_NAME, EnvResponse, EnvResetResponse,
EnvInfo, RewardResult, RunConfig) are kept IDENTICAL to tau_bench so the cloned
trainable_agents.py / generate_with_ws.py / async_env.py import unchanged.

What differs from tau_bench:
  * `Task` is workspace-shaped (task_id + instruction + the rubric/output
    metadata loaded from the benchmark's metadata.json) instead of the
    retail-shaped (user_id, actions, outputs).
  * `RewardResult.info` is a free-form dict (per-rubric judge verdicts) instead
    of the retail DB-hash / output-substring infos.
  * `RunConfig` drops every user-simulator field (Workspace-Bench has no user
    sim). It keeps ONLY what agent_factory reads: agent_strategy, model,
    model_provider, temperature, env, task_split.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# There is no user simulator in Workspace-Bench, so a "respond" turn is just the
# agent emitting plain text (no tool call). We keep the name so the cloned agent
# loop (call_to_action_sglang, asolve) is byte-identical to tau's. A respond turn
# never terminates the episode here — only the `finish` terminate-tool does.
RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"


class Action(BaseModel):
    name: str
    kwargs: Dict[str, Any]


class Task(BaseModel):
    """One Workspace-Bench task, loaded from a per-task metadata.json.

    Fields mirror the benchmark's metadata.json (see docs/dataset.md):
      task          -> instruction (the prompt shown to the agent)
      rubrics       -> list of free-text criteria the judge scores pass/fail
      output_files  -> expected produced files (collected for the judge)
      file_system   -> persona/workspace key -> which workdir to mount
      data_manifest -> [{filename, stored_relpath}] copied into the workspace
    """

    task_id: str
    instruction: str
    rubrics: List[str] = []
    output_files: List[str] = []
    file_system: Optional[str] = None
    persona: Optional[str] = None
    data_manifest: List[Dict[str, Any]] = []
    absolute_id: Optional[int] = None
    rubric_types: Optional[Any] = None


class RewardResult(BaseModel):
    """Reward + judge diagnostics. `reward` is the fraction of rubrics passed."""

    reward: float
    info: Dict[str, Any] = {}


class SolveResult(BaseModel):
    reward: float
    messages: List[Dict[str, Any]]
    info: Dict[str, Any]
    total_cost: Optional[float] = None


class EnvInfo(BaseModel):
    task: Task
    source: Optional[str] = None
    reward_info: Optional[RewardResult] = None


class EnvResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: EnvInfo


class EnvResetResponse(BaseModel):
    observation: str
    info: EnvInfo


class RunConfig(BaseModel):
    """Trimmed RunConfig. Only the fields agent_factory / generate read survive.

    model / model_provider are UNUSED by the SGLang rollout path (inference goes
    through the in-cluster server) but agent_factory passes them to the agent
    constructor, so we keep placeholders — exactly as the tau migration did.
    """

    model_provider: str = "auto_router"
    model: str = "qwen3.5-4b"
    agent_strategy: str = "tool-calling"
    temperature: float = 0.0
    env: str = "workspace"
    task_split: str = "train"
    seed: int = 10
