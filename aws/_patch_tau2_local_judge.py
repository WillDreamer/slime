#!/usr/bin/env python3
# Idempotent patch: route tau2's EVAL-TIME LLM calls to a LOCAL served model.
# tau2 hardcodes cloud defaults for these (config.py):
#   DEFAULT_LLM_NL_ASSERTIONS   = gpt-4.1   (the reward judge for NL assertions)
#   DEFAULT_LLM_ENV_INTERFACE   = gpt-4.1
#   DEFAULT_LLM_EVAL_USER_SIMULATOR = claude-opus-4-5  (review calls)
# Offline (OPENAI_API_KEY=dummy) these hit api.openai.com and fail with
# "Incorrect API key: dummy" -> the sim becomes an infrastructure_error and the
# task can't be scored. Point them at a local endpoint instead. Model + api_base
# come from env so each eval lane can target its paired GLM replica:
#   TAU2_JUDGE_MODEL      (default openai/GLM-4.7-Flash)
#   TAU2_JUDGE_API_BASE   (default http://127.0.0.1:7006/v1)
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/home/ec2-user/tau2-bench"
cfg = os.path.join(root, "src/tau2/config.py")
MARK = "# === tau2 local-judge patch ==="

src = open(cfg).read()
if MARK in src:
    print("local-judge patch already applied:", cfg)
    sys.exit(0)

block = f'''

{MARK}
import os as _os
_JB = _os.environ.get("TAU2_JUDGE_API_BASE", "http://127.0.0.1:7006/v1")
_JM = _os.environ.get("TAU2_JUDGE_MODEL", "openai/GLM-4.7-Flash")
DEFAULT_LLM_NL_ASSERTIONS = _JM
DEFAULT_LLM_NL_ASSERTIONS_ARGS = {{"temperature": 0.0, "api_base": _JB}}
DEFAULT_LLM_ENV_INTERFACE = _JM
DEFAULT_LLM_ENV_INTERFACE_ARGS = {{"temperature": 0.0, "api_base": _JB}}
DEFAULT_LLM_EVAL_USER_SIMULATOR = _JM
# === end tau2 local-judge patch ===
'''
with open(cfg, "a") as f:
    f.write(block)
print("local-judge patch applied to", cfg)
