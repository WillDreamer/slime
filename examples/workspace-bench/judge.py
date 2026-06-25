"""Bedrock Claude rubric judge — the reward signal for Workspace-Bench RL.

Reuses the boto3 invoke_model pattern from greenland_utils/api_usage.py (the same
credential chain the tau-bench Bedrock user-sim used: EcsContainer ->
greenland-dev-role in the Greenland container; AWS_PROFILE=greenland-dev on a dev
box). We deliberately do NOT use Workspace-Bench's upstream judge
(evaluation/src/agent_as_a_judge.py), which drives the judge through the
ClaudeCode.js / Anthropic-endpoint harness — that would require node + an external
Anthropic-compatible endpoint. A direct Bedrock invoke needs neither.

Reward = fraction of rubrics the judge marks `passed` (dense, in [0, 1]). This
matches Workspace-Bench's own unweighted pass/fail summary. The format penalty
(-0.1 on format-bad success) and truncation reward (-0.2) are layered on later by
the agent loop, exactly as in tau.

Configuration (env):
  WS_JUDGE_MODEL_ID    Bedrock model id (default us.anthropic.claude-sonnet-4-6)
  WS_BEDROCK_REGION    region (default us-east-1)
  WS_JUDGE_PROFILE     boto3 profile name (optional; unset -> default chain)
  WS_JUDGE_MAX_TOKENS  judge response budget (default 4096)
  WS_JUDGE_FILE_CHARS  per-output-file char budget inlined into the prompt (default 12000)
"""

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

WS_JUDGE_MODEL_ID = os.environ.get("WS_JUDGE_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
WS_BEDROCK_REGION = os.environ.get("WS_BEDROCK_REGION", "us-east-1")
WS_JUDGE_PROFILE = os.environ.get("WS_JUDGE_PROFILE") or None
WS_JUDGE_MAX_TOKENS = int(os.environ.get("WS_JUDGE_MAX_TOKENS", "4096"))
WS_JUDGE_FILE_CHARS = int(os.environ.get("WS_JUDGE_FILE_CHARS", "12000"))
WS_JUDGE_MAX_FILES = int(os.environ.get("WS_JUDGE_MAX_FILES", "12"))

_JUDGE_SYSTEM = (
    "You are a strict evaluator for a file-editing agent task. You are given the "
    "task instruction, the list of files the agent produced (with their contents), "
    "and a list of rubric criteria. For EACH rubric, decide whether the agent's "
    "work satisfies it.\n"
    "Judge ONLY from the evidence provided. If a rubric cannot be verified from the "
    "provided files, mark it as NOT passed (passed=false). Do not give the benefit "
    "of the doubt.\n"
    "Respond with ONLY a single JSON object, no prose, in EXACTLY this shape:\n"
    '{"rubrics": [{"index": 0, "passed": true, "confidence": 0.9, "evidence": "..."}, ...]}\n'
    "Include one entry per rubric, using the rubric's 0-based index. `passed` is a "
    "boolean, `confidence` a number in [0,1], `evidence` a short justification."
)


def _truncate(text: str, cap: int) -> str:
    if text is None:
        return ""
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    tail = cap - head
    return f"{text[:head]}\n...[truncated {len(text) - cap} chars]...\n{text[-tail:]}"


class RubricJudge:
    def __init__(
        self,
        model_id: str = WS_JUDGE_MODEL_ID,
        region: str = WS_BEDROCK_REGION,
        profile_name: Optional[str] = WS_JUDGE_PROFILE,
    ):
        cfg = Config(
            retries={"total_max_attempts": 4, "mode": "standard"},
            connect_timeout=10,
            read_timeout=120,
        )
        session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
        self.client = session.client("bedrock-runtime", region_name=region, config=cfg)
        self.model_id = model_id

    # ---- Bedrock invoke (mirrors api_usage.py temperature-drop fallback) --
    def _invoke(self, system: str, user: str) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": WS_JUDGE_MAX_TOKENS,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        try:
            resp = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except Exception as e:
            # opus-4-7/4-8 reject `temperature`; retry without it (api_usage.py).
            if "temperature" in str(e).lower():
                body.pop("temperature", None)
                resp = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
            else:
                raise
        payload = json.loads(resp["body"].read())
        return "".join(b.get("text", "") for b in payload.get("content", []))

    # ---- prompt construction --------------------------------------------
    def _build_user_prompt(
        self,
        task,
        rubrics: List[str],
        work_dir: str,
        base_dir: str,
        output_dir: str,
        output_files: List[str],
    ) -> str:
        # Inline the produced output-file contents (Bedrock cannot read the FS).
        files_blob: List[Dict[str, Any]] = []
        n = 0
        for rel in output_files or []:
            if n >= WS_JUDGE_MAX_FILES:
                break
            # Prefer the collected copy in output_dir; fall back to the overlay.
            candidates = [os.path.join(output_dir, rel.lstrip("/")), os.path.join(work_dir, rel.lstrip("/"))]
            content = None
            for p in candidates:
                if os.path.isfile(p):
                    try:
                        with open(p, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except OSError:
                        content = None
                    break
            if content is None:
                files_blob.append({"path": rel, "status": "NOT_PRODUCED"})
            else:
                files_blob.append({"path": rel, "content": _truncate(content, WS_JUDGE_FILE_CHARS)})
            n += 1

        # If the task declares no explicit output_files, give the judge a listing
        # of what the agent wrote into the overlay so it has something to grade.
        extra_written: List[str] = []
        if not output_files and work_dir and os.path.isdir(work_dir):
            for dirpath, _dirs, fnames in os.walk(work_dir):
                for fn in fnames:
                    rel = os.path.relpath(os.path.join(dirpath, fn), work_dir)
                    extra_written.append(rel)
                    if len(extra_written) >= 50:
                        break
                if len(extra_written) >= 50:
                    break

        payload = {
            "task": task.instruction,
            "rubrics": [{"index": i, "rubric": r} for i, r in enumerate(rubrics)],
            "produced_files": files_blob,
        }
        if extra_written:
            payload["files_written_by_agent"] = extra_written
        return json.dumps(payload, ensure_ascii=False)

    # ---- response parsing ------------------------------------------------
    @staticmethod
    def _parse(raw: str, n_rubrics: int) -> List[Dict[str, Any]]:
        """Extract per-rubric verdicts; default missing/garbled to passed=False."""
        verdicts: Dict[int, Dict[str, Any]] = {}
        obj = None
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
        if isinstance(obj, dict) and isinstance(obj.get("rubrics"), list):
            for item in obj["rubrics"]:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                passed = bool(item.get("passed"))
                conf = item.get("confidence")
                try:
                    conf = float(conf) if conf is not None else None
                except (TypeError, ValueError):
                    conf = None
                verdicts[idx] = {
                    "index": idx,
                    "passed": passed,
                    "confidence": conf,
                    "evidence": str(item.get("evidence", ""))[:500],
                }
        # Fill any rubric the judge omitted with a conservative fail.
        rows = []
        for i in range(n_rubrics):
            rows.append(
                verdicts.get(i, {"index": i, "passed": False, "confidence": 0.0, "evidence": "no verdict returned"})
            )
        return rows

    def score_rubrics(
        self,
        task,
        rubrics: List[str],
        work_dir: str,
        base_dir: str = "",
        output_dir: str = "",
        output_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        n = len(rubrics)
        user = self._build_user_prompt(task, rubrics, work_dir, base_dir, output_dir, output_files or [])
        raw = self._invoke(_JUDGE_SYSTEM, user)
        rows = self._parse(raw, n)
        num_passed = sum(1 for r in rows if r["passed"])
        pass_fraction = (num_passed / n) if n > 0 else 0.0
        return {
            "pass_fraction": pass_fraction,
            "num_rubrics": n,
            "num_passed": num_passed,
            "per_rubric": rows,
            "judge_model": self.model_id,
        }


# Module-level singleton (boto3 invoke_model is thread-safe; the async_env thread
# pool fans many concurrent judge calls through one client).
_JUDGE: Optional[RubricJudge] = None
_JUDGE_LOCK = threading.Lock()


def get_judge() -> RubricJudge:
    global _JUDGE
    if _JUDGE is None:
        with _JUDGE_LOCK:
            if _JUDGE is None:
                _JUDGE = RubricJudge()
    return _JUDGE


if __name__ == "__main__":
    # Smoke test: WS_JUDGE_PROFILE=greenland-dev python judge.py
    import types as _t

    fake_task = _t.SimpleNamespace(instruction="Write a file hello.txt containing the word 'hello'.")
    rubrics = ["A file named hello.txt exists.", "hello.txt contains the word 'hello'."]
    tmp = "/tmp/ws_judge_smoke"
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "hello.txt"), "w") as f:
        f.write("hello world")
    out = get_judge().score_rubrics(
        fake_task, rubrics, work_dir=tmp, output_dir=tmp, output_files=["hello.txt"]
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
