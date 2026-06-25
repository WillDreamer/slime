"""WorkspaceEnv — the Workspace-Bench environment for slime RL.

Clones the structural skeleton of tau_bench/envs/base.py::Env (reset/step over a
tools_map, a terminate-tool that flips done=True, reward computed once at
termination), but swaps the two benchmark-specific pieces:

  * reset()           : instead of resetting a user simulator, it MATERIALIZES
                        the task's writable workspace (a per-rollout copy-on-write
                        overlay over the shared persona workspace) and copies the
                        task's data_manifest files into it.
  * calculate_reward(): instead of replaying ground-truth actions and hashing the
                        DB, it collects the produced output files and scores the
                        task's rubrics with a Bedrock Claude judge (judge.py),
                        returning the fraction of rubrics passed in [0, 1].

The rest of the contract trainable_agents.py relies on is preserved exactly:
`tools_info`, `wiki`, `reset(task_index)->EnvResetResponse`,
`step(action)->EnvResponse`, `terminate_tools`.
"""

import json
import logging
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional

from ws_bench.envs.tools import ALL_TOOLS, TERMINATE_TOOLS, WorkspaceContext
from ws_bench.envs.wiki import WIKI
from ws_bench.types import (
    Action,
    EnvInfo,
    EnvResetResponse,
    EnvResponse,
    RESPOND_ACTION_NAME,
    RewardResult,
    Task,
)

logger = logging.getLogger(__name__)

# Layout under WS_DATA_DIR (staged from s3://whx-agent/data/workspace-bench/):
#   tasks/<task_id>/metadata.json          per-task definition + small manifest files
#   workspaces/<workdir>/...               the large shared persona filesystems (RO)
#   ws_<split>_train_tasks.jsonl           {"index": i} rows (slime --prompt-data)
#   ws_<split>_dev_tasks.jsonl
#   task_index_map.json                    {"<split>": ["<task_id>", ...]}  index -> folder
_DATA_DIR_DEFAULT = os.environ.get("WS_DATA_DIR", "/tmp/instance_storage/whx/data_root/workspace-bench")
# Where per-rollout writable overlays live (NVMe). Torn down after reward.
_ROLLOUT_ROOT = os.environ.get(
    "WS_ROLLOUT_ROOT", os.path.join(os.environ.get("ROOT_DIR", "/tmp/instance_storage/whx"), "ws_rollouts")
)
_MAX_STEPS_DEFAULT = int(os.environ.get("WS_MAX_STEPS", "30"))

# Runtime fallback: map a task's file_system (localized label or English persona)
# to its *_raw workspace dir, in case prepare_ws_data.py did not normalize it.
# Mirrors the maps in prepare_ws_data.py.
_FS_LABEL_TO_WORKDIR = {
    "产品人员": "chanpin_raw",
    "开发人员": "kaifa_raw",
    "研究人员": "research_raw",
    "运营人员": "yunying_raw",
    "行政/后勤人员": "houqin_raw",
    "Product Manager": "chanpin_raw",
    "Backend Developer": "kaifa_raw",
    "Researcher": "research_raw",
    "Operations Manager": "yunying_raw",
    "Logistics Manager": "houqin_raw",
    "ProductManager_Workdir": "chanpin_raw",
    "BackendDeveloper_Workdir": "kaifa_raw",
    "Research_Workdir": "research_raw",
    "OperationsManager_Workdir": "yunying_raw",
    "LogisticsManager_Workdir": "houqin_raw",
}
_RAW_DIRS = {"chanpin_raw", "kaifa_raw", "research_raw", "yunying_raw", "houqin_raw"}


def _load_index_map(data_dir: str, split: str) -> List[str]:
    """Return the ordered list of task-folder names for `split`."""
    path = os.path.join(data_dir, "task_index_map.json")
    with open(path) as f:
        m = json.load(f)
    if isinstance(m, dict):
        ids = m.get(split) or m.get("default") or []
    else:
        ids = m  # a bare list applies to all splits
    if not ids:
        raise ValueError(f"task_index_map.json has no entries for split {split!r} at {path}")
    return list(ids)


def _load_metadata(data_dir: str, task_id: str) -> Dict[str, Any]:
    path = os.path.join(data_dir, "tasks", str(task_id), "metadata.json")
    with open(path) as f:
        return json.load(f)


def _task_from_metadata(task_id: str, meta: Dict[str, Any]) -> Task:
    """Normalize metadata.json into a Task (tolerant of field aliases)."""
    instruction = meta.get("task") or meta.get("instruction") or ""
    output_files = meta.get("output_files") or meta.get("output_file") or []
    if isinstance(output_files, str):
        output_files = [output_files]
    rubrics = meta.get("rubrics") or []
    if isinstance(rubrics, str):
        rubrics = [rubrics]
    file_system = meta.get("file_system") or meta.get("persona") or None
    return Task(
        task_id=str(task_id),
        instruction=instruction,
        rubrics=list(rubrics),
        output_files=list(output_files),
        file_system=file_system,
        persona=meta.get("persona"),
        data_manifest=meta.get("data_manifest") or [],
        absolute_id=meta.get("absolute_id"),
        rubric_types=meta.get("rubric_types"),
    )


class WorkspaceEnv:
    def __init__(
        self,
        task_split: str = "train",
        task_index: Optional[int] = None,
        data_dir: Optional[str] = None,
        rollout_uid: Optional[str] = None,
    ) -> None:
        self.data_dir = data_dir or _DATA_DIR_DEFAULT
        self.task_split = task_split
        self.rollout_uid = rollout_uid  # disambiguates parallel rollouts of one task
        # Per-env-instance unique token. Under train_async's rollout overlap the
        # SAME (task_index, group_index) -> same rollout_uid can be processed by
        # two concurrent generate() calls; if their overlay dirs collided, one
        # env's rmtree/cleanup would wipe the other's work_dir mid-materialize
        # (FileNotFound on a manifest file) or makedirs would race (FileExists).
        # This token makes every env instance's overlay path GLOBALLY UNIQUE, so
        # rmtree/cleanup only ever touch this instance's own private leaf dir.
        self._instance_token = uuid.uuid4().hex[:12]

        self.tools_map = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}
        self.tools_info = [t.get_info() for t in ALL_TOOLS]
        self.terminate_tools = list(TERMINATE_TOOLS)
        self.wiki = WIKI

        self._index_map = _load_index_map(self.data_dir, task_split)
        self.task_index = task_index if task_index is not None else 0
        self.task: Optional[Task] = None
        self.ctx: Optional[WorkspaceContext] = None
        self.actions: List[Action] = []

    # ---- workspace materialization ---------------------------------------
    def _persona_base_dir(self, task: Task) -> str:
        """Resolve the shared READ-ONLY persona workspace for this task.

        metadata.json's `file_system` names which workdir to mount. Tasks that
        ship all their files via data_manifest may have no base dir (returns "").
        """
        key = task.file_system
        if not key:
            return ""
        # Normalize to the *_raw dir name (prepare_ws_data.py usually does this,
        # but be resilient to un-normalized labels / personas).
        if key not in _RAW_DIRS:
            key = _FS_LABEL_TO_WORKDIR.get(key, _FS_LABEL_TO_WORKDIR.get(task.persona or "", key))
        cand = os.path.join(self.data_dir, "workspaces", str(key))
        if os.path.isdir(cand):
            return cand
        logger.warning(f"persona workspace not found for file_system={task.file_system!r} (looked at {cand})")
        return cand

    def _materialize(self, task: Task) -> WorkspaceContext:
        base_dir = self._persona_base_dir(task)
        # Overlay path leaf is GLOBALLY UNIQUE per env instance (rollout_uid is for
        # readability/grouping; _instance_token guarantees no collision even when
        # two concurrent generate() calls share the same rollout_uid under async
        # rollout overlap). The rollout root itself (self._rollout_dir) is owned
        # solely by this instance, so the rmtree/makedirs below can't race another.
        uid = self.rollout_uid or "r0"
        self._rollout_dir = os.path.join(
            _ROLLOUT_ROOT, str(task.task_id), f"{uid}-{self._instance_token}"
        )
        work_dir = os.path.join(self._rollout_dir, "work")
        output_dir = os.path.join(self._rollout_dir, "output")
        # Fresh overlay each reset. This instance owns _rollout_dir exclusively, so
        # blowing it away wholesale is safe and also clears a prior reset's state.
        shutil.rmtree(self._rollout_dir, ignore_errors=True)
        for d in (work_dir, output_dir):
            os.makedirs(d, exist_ok=True)

        # Copy the task's data_manifest files into the writable overlay. Each
        # entry is {filename, stored_relpath}: stored_relpath is relative to the
        # task folder; filename is where it should live in the workspace.
        task_dir = os.path.join(self.data_dir, "tasks", str(task.task_id))
        for entry in task.data_manifest or []:
            stored = entry.get("stored_relpath") or entry.get("path")
            dest_rel = entry.get("filename") or (os.path.basename(stored) if stored else None)
            if not stored or not dest_rel:
                continue
            src = os.path.join(task_dir, stored)
            if not os.path.isfile(src):
                logger.warning(f"data_manifest source missing: {src}")
                continue
            dest = os.path.join(work_dir, dest_rel.lstrip("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)

        return WorkspaceContext(
            base_dir=base_dir,
            work_dir=work_dir,
            output_dir=output_dir,
            output_files=list(task.output_files or []),
        )

    # ---- gym-like API ----------------------------------------------------
    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        if task_index is None:
            task_index = self.task_index
        self.task_index = task_index
        task_id = self._index_map[task_index % len(self._index_map)]
        meta = _load_metadata(self.data_dir, task_id)
        self.task = _task_from_metadata(task_id, meta)
        self.ctx = self._materialize(self.task)
        self.actions = []
        # The initial observation is the task instruction (the first user turn),
        # mirroring tau where reset's observation is the user's first message.
        return EnvResetResponse(
            observation=self.task.instruction,
            info=EnvInfo(task=self.task, source="task"),
        )

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)
        info = EnvInfo(task=self.task)
        reward = 0.0
        done = False

        if action.name == RESPOND_ACTION_NAME:
            # No user simulator: a plain reply just nudges the agent to keep going.
            observation = "Acknowledged. Continue working and call `finish` when the task is complete."
            info.source = "agent"
        elif action.name in self.tools_map:
            try:
                observation = self.tools_map[action.name].invoke(data=self.ctx, **action.kwargs)
            except Exception as e:
                observation = f"Error: {e}"
            info.source = action.name
            if action.name in self.terminate_tools:
                done = True
        else:
            observation = f"Unknown action {action.name}"
            info.source = action.name

        if done:
            reward_res = self.calculate_reward()
            reward = reward_res.reward
            info.reward_info = reward_res

        return EnvResponse(observation=observation, reward=reward, done=done, info=info)

    # ---- reward: rubric judge -------------------------------------------
    def _collect_outputs(self) -> None:
        """Copy the task's expected output files from the overlay into output_dir.

        Reads work_dir first, base_dir second (so an agent-produced file wins over
        a same-named base file). Missing outputs are simply absent — the judge then
        sees they were not produced and scores the relevant rubrics as failed.
        """
        assert self.ctx is not None and self.task is not None
        for rel in self.task.output_files or []:
            src = self.ctx.resolve_read(rel)
            if src is None or not os.path.isfile(src):
                continue
            dest = os.path.join(self.ctx.output_dir, rel.lstrip("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                shutil.copy2(src, dest)
            except OSError as e:
                logger.warning(f"could not collect output {rel!r}: {e}")

    def calculate_reward(self) -> RewardResult:
        assert self.ctx is not None and self.task is not None
        self._collect_outputs()
        # Import lazily so the env stays importable without boto3 (e.g. data prep).
        from judge import get_judge

        if not self.task.rubrics:
            logger.warning(f"task {self.task.task_id} has no rubrics; reward=0")
            return RewardResult(reward=0.0, info={"num_rubrics": 0, "judge_error": "no_rubrics"})

        try:
            result = get_judge().score_rubrics(
                task=self.task,
                rubrics=self.task.rubrics,
                work_dir=self.ctx.work_dir,
                base_dir=self.ctx.base_dir,
                output_dir=self.ctx.output_dir,
                output_files=self.task.output_files,
            )
        except Exception as e:  # one bad judge call must not kill the gather
            logger.warning(f"judge failed for task {self.task.task_id}: {e.__class__.__name__}: {e}")
            return RewardResult(reward=0.0, info={"judge_error": f"{e.__class__.__name__}: {e}"})

        return RewardResult(reward=float(result.get("pass_fraction", 0.0)), info=result)

    def cleanup(self) -> None:
        """Tear down THIS instance's private overlay to bound NVMe usage.

        Removes only self._rollout_dir (.../<task_id>/<uid>-<token>), which this
        env instance owns exclusively — never a shared parent — so a concurrent
        rollout of the same task is unaffected.
        """
        rollout_dir = getattr(self, "_rollout_dir", None)
        if rollout_dir:
            shutil.rmtree(rollout_dir, ignore_errors=True)


# Convenience alias so call sites can read `Env` like tau's base module.
Env = WorkspaceEnv
