#!/usr/bin/env python3
"""Fix B (see eval_scai/TAU2_FORMAT_ISSUE.md): make tau2-bench tolerant of tool
calls whose `arguments` arrive as a bare scalar instead of a {param: value}
object — e.g. arguments='1008292232' instead of {"product_id": "1008292232"}.

Some RL/SFT'd checkpoints emit the argument value in the wrong place. The lenient
slime training harness accepted this; tau2's strict pydantic ToolCall(arguments:
dict) hard-fails, which trips TOO_MANY_ERRORS and zeros the episode. This patch
coerces a bare scalar into {single_required_param: value} when the target tool
has exactly one parameter, and otherwise falls back to {} with a warning instead
of crashing.

Idempotent: re-running is a no-op once the marker is present.
Disable at runtime with TAU2_LENIENT_TOOL_ARGS=0 (restores strict behaviour).

Usage: python3 _patch_tau2_lenient_args.py [TAU2_BENCH_DIR]
       (default TAU2_BENCH_DIR=/home/ec2-user/tau2-bench)
"""
import sys
import os

TAU2_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/ec2-user/tau2-bench"
TARGET = os.path.join(TAU2_DIR, "src", "tau2", "utils", "llm_utils.py")

MARKER = "[tau2-lenient-args]"

OLD = """    raw_tool_calls = response_choice.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in raw_tool_calls
    ]
    tool_calls = tool_calls or None
"""

NEW = """    raw_tool_calls = response_choice.message.tool_calls or []

    # [tau2-lenient-args] Coerce tool-call arguments that arrive as a bare scalar
    # instead of a {param: value} object (e.g. '1008292232' instead of
    # {"product_id": "1008292232"}). Some RL/SFT'd checkpoints put the value in
    # the wrong place; without this, pydantic ToolCall(arguments: dict) hard-fails
    # and the episode trips TOO_MANY_ERRORS. Set TAU2_LENIENT_TOOL_ARGS=0 to
    # restore strict behaviour. See eval_scai/TAU2_FORMAT_ISSUE.md.
    _lenient_args = os.environ.get("TAU2_LENIENT_TOOL_ARGS", "1") != "0"
    _param_index = {}
    if _lenient_args and tools_schema:
        for _ts in tools_schema:
            _fn = _ts.get("function", {}) if isinstance(_ts, dict) else {}
            _p = _fn.get("parameters") or {}
            _param_index[_fn.get("name")] = (
                list(_p.get("required") or []),
                list((_p.get("properties") or {}).keys()),
            )

    def _coerce_tool_args(_name, _raw):
        _parsed = json.loads(_raw)
        if isinstance(_parsed, dict) or not _lenient_args:
            return _parsed
        _req, _props = _param_index.get(_name, ([], []))
        _key = _req[0] if len(_req) == 1 else (_props[0] if len(_props) == 1 else None)
        if _key is not None:
            logger.warning(
                f"[tau2-lenient-args] {_name}: bare argument {_parsed!r} coerced to {{{_key!r}: ...}}"
            )
            return {_key: _parsed}
        logger.warning(
            f"[tau2-lenient-args] {_name}: non-dict argument {_parsed!r} with !=1 param; using {{}}"
        )
        return {}

    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=_coerce_tool_args(
                tool_call.function.name, tool_call.function.arguments
            ),
        )
        for tool_call in raw_tool_calls
    ]
    tool_calls = tool_calls or None
"""


def main():
    if not os.path.isfile(TARGET):
        print(f"[patch] SKIP: not found: {TARGET}")
        return 0
    src = open(TARGET, "r", encoding="utf-8").read()
    if MARKER in src:
        print(f"[patch] already applied: {TARGET}")
        return 0
    if OLD not in src:
        print(f"[patch] ERROR: anchor block not found in {TARGET} "
              f"(tau2-bench version changed?). No edit made.")
        return 2
    if src.count(OLD) != 1:
        print(f"[patch] ERROR: anchor block found {src.count(OLD)} times; expected 1.")
        return 2
    open(TARGET, "w", encoding="utf-8").write(src.replace(OLD, NEW))
    print(f"[patch] applied Fix B to {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
