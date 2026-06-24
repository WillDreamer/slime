#!/usr/bin/env python3
"""Take only the FIRST tool call per assistant turn in tau2 (opt-in).

Some checkpoints (Search-R1 SFT) emit many tool calls in one turn with no turn
boundary (no <|im_end|>), which tau2 executes all of -> TOO_MANY_ERRORS -> 0.
tau-bench *training* only ever executed the first call
(openai_tool_adapter.py: "Multiple tool calls identified, only taking first").
This patch replicates that, but ONLY when TAU2_FIRST_TOOL_CALL=1 is set in the
environment (default OFF), so lanes that don't set it are completely unchanged.

Idempotent. Usage: python3 _patch_tau2_first_toolcall.py [TAU2_BENCH_DIR]
"""
import sys, os
TAU2 = sys.argv[1] if len(sys.argv) > 1 else "/home/ec2-user/tau2-bench"
TARGET = os.path.join(TAU2, "src", "tau2", "utils", "llm_utils.py")
MARKER = "[tau2-first-tool-call]"

OLD = "    tool_calls = tool_calls or None\n\n    message = AssistantMessage(\n"
NEW = ("    tool_calls = tool_calls or None\n"
       "    # [tau2-first-tool-call] opt-in (TAU2_FIRST_TOOL_CALL=1): some ckpts spray\n"
       "    # many tool calls per turn with no turn boundary; tau-bench training only\n"
       "    # executed the first. Take first to avoid TOO_MANY_ERRORS. Default OFF.\n"
       "    if tool_calls and len(tool_calls) > 1 and os.environ.get(\"TAU2_FIRST_TOOL_CALL\", \"0\") == \"1\":\n"
       "        tool_calls = tool_calls[:1]\n"
       "\n"
       "    message = AssistantMessage(\n")


def main():
    if not os.path.isfile(TARGET):
        print(f"[patch-first] SKIP: not found: {TARGET}"); return 0
    src = open(TARGET, encoding="utf-8").read()
    if MARKER in src:
        print(f"[patch-first] already applied: {TARGET}"); return 0
    if src.count(OLD) != 1:
        print(f"[patch-first] ERROR: anchor found {src.count(OLD)} times (expected 1); no edit."); return 2
    open(TARGET, "w", encoding="utf-8").write(src.replace(OLD, NEW))
    print(f"[patch-first] applied to {TARGET}"); return 0


if __name__ == "__main__":
    sys.exit(main())
