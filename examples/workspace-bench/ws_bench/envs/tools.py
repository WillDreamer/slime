"""File-editing tools for the Workspace-Bench agent.

These play the role tau-bench's retail/airline API tools play, but instead of
mutating an in-memory `data` dict they read/write a real workspace on disk. The
`data` argument passed to every `invoke` is a `WorkspaceContext` (see below)
carrying the rollout's directories.

Workspace isolation — SOFTWARE COPY-ON-WRITE (no privileges, no 20 GB copy):
  * base_dir  : the persona's workspace, READ-ONLY and SHARED across all rollouts
                on the node (staged once to NVMe).
  * work_dir  : this rollout's private, initially-near-empty overlay (only the
                task's data_manifest files + anything the agent writes live here).
  * Reads (read_file/list_dir/grep) resolve against work_dir first, falling back
    to base_dir — so the agent sees the union (its edits shadow the base).
  * Writes (write_file/edit_file) always land in work_dir; edit_file copies the
    base file up on first edit. n_samples_per_prompt parallel rollouts of one
    task therefore cost ~(touched files) each, not 16 x base_dir.

Every tool PATH-JAILS its argument: the resolved absolute path must stay under
base_dir or work_dir. Escapes return an error STRING (tau's Env.step catches
exceptions, but error strings are cleaner and keep the trajectory going).
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ws_bench.envs.tool import Tool

# Per-read character cap so a single read_file can't blow up the multi-turn
# context (every turn re-renders the whole history via apply_chat_template).
_READ_CHAR_CAP = int(os.environ.get("WS_READ_CHAR_CAP", "8192"))
# Max entries returned by list_dir / matches returned by grep (bound 20 GB trees).
_LIST_CAP = int(os.environ.get("WS_LIST_CAP", "200"))
_GREP_CAP = int(os.environ.get("WS_GREP_CAP", "50"))
# Files larger than this (bytes) are skipped by grep / summarized by read.
_MAX_GREP_FILE_BYTES = int(os.environ.get("WS_MAX_GREP_FILE_BYTES", str(2 * 1024 * 1024)))


@dataclass
class WorkspaceContext:
    """The per-rollout filesystem context handed to every tool's `invoke`.

    Mirrors the role tau-bench's mutable `data` dict plays, but for files.
    """

    base_dir: str  # RO shared persona workspace (may be "" if a task ships no base)
    work_dir: str  # RW private overlay for this rollout
    output_dir: str  # where output_files are collected at finish (for the judge)
    output_files: List[str] = field(default_factory=list)
    read_char_cap: int = _READ_CHAR_CAP
    # set by Finish.invoke so WorkspaceEnv.step knows the agent's closing summary
    finish_summary: str = ""

    # ---- path resolution + jailing ---------------------------------------
    def _jail(self, root: str, rel: str) -> Optional[str]:
        """Resolve `rel` under `root`; return abs path or None if it escapes."""
        if not root:
            return None
        root_real = os.path.realpath(root)
        cand = os.path.realpath(os.path.join(root, rel))
        if cand == root_real or cand.startswith(root_real + os.sep):
            return cand
        return None

    def resolve_read(self, rel: str) -> Optional[str]:
        """Existing path for reading: work_dir shadows base_dir."""
        rel = (rel or ".").lstrip("/")
        w = self._jail(self.work_dir, rel)
        if w is not None and os.path.exists(w):
            return w
        b = self._jail(self.base_dir, rel)
        if b is not None and os.path.exists(b):
            return b
        # Return the work_dir candidate (may not exist) so callers can report
        # "not found" rather than "escapes" when the path is legal but absent.
        return w

    def resolve_write(self, rel: str) -> Optional[str]:
        """Target path for writing — always under work_dir."""
        rel = (rel or "").lstrip("/")
        if not rel:
            return None
        return self._jail(self.work_dir, rel)

    def list_roots(self, rel: str) -> List[str]:
        """Both jailed roots for a relative dir (work first), for union listing."""
        rel = (rel or ".").lstrip("/")
        out = []
        for root in (self.work_dir, self.base_dir):
            p = self._jail(root, rel)
            if p is not None and os.path.isdir(p):
                out.append(p)
        return out


def _err(msg: str) -> str:
    return f"Error: {msg}"


class ListDir(Tool):
    @staticmethod
    def invoke(data: WorkspaceContext, path: str = ".") -> str:
        roots = data.list_roots(path)
        if not roots:
            return _err(f"directory not found or escapes workspace: {path!r}")
        seen: Dict[str, str] = {}  # name -> "dir"/"file (N bytes)"
        for root in roots:  # work_dir first; base_dir entries don't overwrite
            try:
                for entry in os.scandir(root):
                    if entry.name in seen:
                        continue
                    if entry.is_dir():
                        seen[entry.name] = "dir"
                    else:
                        try:
                            seen[entry.name] = f"file ({entry.stat().st_size} bytes)"
                        except OSError:
                            seen[entry.name] = "file"
            except OSError as e:
                return _err(f"cannot list {path!r}: {e}")
        names = sorted(seen.items())
        truncated = len(names) > _LIST_CAP
        lines = [f"{n}\t{kind}" for n, kind in names[:_LIST_CAP]]
        if truncated:
            lines.append(f"... ({len(names) - _LIST_CAP} more entries omitted)")
        return "\n".join(lines) if lines else "(empty directory)"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List the files and subdirectories at a path within the workspace (relative to the workspace root). Shows your edits merged over the original files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path relative to the workspace root. Defaults to '.' (the root).",
                        },
                    },
                    "required": [],
                },
            },
        }


class ReadFile(Tool):
    @staticmethod
    def invoke(
        data: WorkspaceContext,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        abs_path = data.resolve_read(path)
        if abs_path is None:
            return _err(f"path escapes workspace: {path!r}")
        if not os.path.isfile(abs_path):
            return _err(f"file not found: {path!r}")
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as e:
            return _err(f"cannot read {path!r}: {e}")
        if start_line is not None or end_line is not None:
            lines = text.splitlines()
            s = max(0, (start_line or 1) - 1)
            e = end_line if end_line is not None else len(lines)
            text = "\n".join(lines[s:e])
        if len(text) > data.read_char_cap:
            text = text[: data.read_char_cap] + f"\n... (truncated; file has {len(text)} chars, showing first {data.read_char_cap})"
        return text

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the text content of a file in the workspace (relative path). Optionally restrict to a line range. Long files are truncated.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to the workspace root."},
                        "start_line": {"type": "integer", "description": "1-based first line to read (optional)."},
                        "end_line": {"type": "integer", "description": "1-based last line to read, inclusive (optional)."},
                    },
                    "required": ["path"],
                },
            },
        }


class Grep(Tool):
    @staticmethod
    def invoke(data: WorkspaceContext, pattern: str, path: str = ".", max_matches: int = _GREP_CAP) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return _err(f"invalid regex {pattern!r}: {e}")
        roots = data.list_roots(path)
        if not roots:
            # path might be a file rather than a dir
            abs_path = data.resolve_read(path)
            if abs_path and os.path.isfile(abs_path):
                roots = []  # handled by single-file branch below
            else:
                return _err(f"directory not found or escapes workspace: {path!r}")
        max_matches = min(int(max_matches or _GREP_CAP), _GREP_CAP)
        matches: List[str] = []
        seen_files = set()

        def scan_file(abs_f: str, rel_f: str):
            if abs_f in seen_files:
                return
            seen_files.add(abs_f)
            try:
                if os.path.getsize(abs_f) > _MAX_GREP_FILE_BYTES:
                    return
                with open(abs_f, "r", encoding="utf-8", errors="replace") as fh:
                    for ln, line in enumerate(fh, 1):
                        if rx.search(line):
                            matches.append(f"{rel_f}:{ln}: {line.rstrip()[:200]}")
                            if len(matches) >= max_matches:
                                return
            except OSError:
                return

        if roots:
            for root in roots:
                base_for_rel = root
                for dirpath, _dirs, files in os.walk(root):
                    for fn in files:
                        abs_f = os.path.join(dirpath, fn)
                        rel_f = os.path.relpath(abs_f, base_for_rel)
                        scan_file(abs_f, rel_f)
                        if len(matches) >= max_matches:
                            break
                    if len(matches) >= max_matches:
                        break
        else:
            abs_path = data.resolve_read(path)
            scan_file(abs_path, path)

        if not matches:
            return "(no matches)"
        out = "\n".join(matches)
        if len(matches) >= max_matches:
            out += f"\n... (stopped at {max_matches} matches)"
        return out

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search workspace files for a regular-expression pattern and return matching 'path:line: text' entries. Bounded; large files are skipped.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "A Python regular expression."},
                        "path": {"type": "string", "description": "Directory or file to search, relative to the workspace root. Defaults to '.'."},
                        "max_matches": {"type": "integer", "description": "Maximum matches to return (optional)."},
                    },
                    "required": ["pattern"],
                },
            },
        }


class WriteFile(Tool):
    @staticmethod
    def invoke(data: WorkspaceContext, path: str, content: str) -> str:
        abs_path = data.resolve_write(path)
        if abs_path is None:
            return _err(f"path escapes workspace or is empty: {path!r}")
        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content if content is not None else "")
        except OSError as e:
            return _err(f"cannot write {path!r}: {e}")
        return f"Wrote {len(content or '')} characters to {path}"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a file in the workspace with the given content (relative path). This is how you produce or change files — describing a change is not enough.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to the workspace root."},
                        "content": {"type": "string", "description": "Full new content of the file."},
                    },
                    "required": ["path", "content"],
                },
            },
        }


class EditFile(Tool):
    @staticmethod
    def invoke(data: WorkspaceContext, path: str, old_str: str, new_str: str) -> str:
        # Read current content (work_dir shadows base_dir), then copy-up the edit.
        src = data.resolve_read(path)
        if src is None or not os.path.isfile(src):
            return _err(f"file not found: {path!r}")
        try:
            with open(src, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            return _err(f"cannot read {path!r}: {e}")
        if old_str == "":
            return _err("old_str must be non-empty")
        count = content.count(old_str)
        if count == 0:
            return _err(f"old_str not found in {path!r}")
        if count > 1:
            return _err(f"old_str is ambiguous in {path!r} ({count} occurrences); include more surrounding context")
        new_content = content.replace(old_str, new_str, 1)
        dst = data.resolve_write(path)
        if dst is None:
            return _err(f"path escapes workspace: {path!r}")
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8") as f:
                f.write(new_content)
        except OSError as e:
            return _err(f"cannot write {path!r}: {e}")
        return f"Edited {path} (replaced 1 occurrence)"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace exactly one occurrence of old_str with new_str in a workspace file. old_str must appear exactly once; include surrounding context to disambiguate.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to the workspace root."},
                        "old_str": {"type": "string", "description": "Exact text to replace (must occur exactly once)."},
                        "new_str": {"type": "string", "description": "Replacement text."},
                    },
                    "required": ["path", "old_str", "new_str"],
                },
            },
        }


class Finish(Tool):
    @staticmethod
    def invoke(data: WorkspaceContext, summary: str = "") -> str:
        # Terminate tool: WorkspaceEnv registers this in terminate_tools, so the
        # env sets done=True and computes the rubric-judge reward. Record the
        # summary on the context for diagnostics.
        try:
            data.finish_summary = summary or ""
        except Exception:
            pass
        return ""

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Call this when the task is complete and all required output files have been written. Ends the episode and triggers evaluation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "A brief summary of what you did (optional)."},
                    },
                    "required": [],
                },
            },
        }


class Think(Tool):
    @staticmethod
    def invoke(data: WorkspaceContext, thought: str) -> str:
        # No-op: does not change the workspace; just records a thought in the log.
        return ""

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "think",
                "description": (
                    "Use the tool to think about something. It will not read or change any file, "
                    "but just append the thought to the log. Use it when complex reasoning or some cache memory is needed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string", "description": "A thought to think about."},
                    },
                    "required": ["thought"],
                },
            },
        }


# Order matters only cosmetically (tools_info ordering in the prompt).
ALL_TOOLS = [ListDir, ReadFile, Grep, WriteFile, EditFile, Think, Finish]
TERMINATE_TOOLS = ["finish"]
