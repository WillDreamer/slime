#!/bin/bash
# Runs INSIDE the `slime` container. Ensures the tau eval dependencies exist,
# re-cloning/installing only if missing (so a freshly-recreated container is
# repaired, while an intact AMI transfer is a fast no-op).
#   tau1 -> tau_bench  (JD-ETH fork, feature/litellm-retry) + litellm
#   tau2 -> official tau2-bench (sierra-research)
# Note: the GLM user-sim must be served with --reasoning-parser glm45 (set in
# autostart-resume.sh) so tau_bench/tau2 receive clean customer messages.
set -u
log(){ echo "[setup-tau $(date -u +%FT%TZ)] $*"; }

# --- tau1: tau_bench (JD-ETH) -------------------------------------------------
if python -c "import tau_bench" 2>/dev/null; then
  log "tau_bench present"
else
  log "tau_bench missing -> clone + install"
  cd /home/ec2-user && rm -rf tau-bench
  git clone --depth 1 -b feature/litellm-retry https://github.com/JD-ETH/tau-bench.git
  cd tau-bench && pip install -e . --no-deps && pip install litellm
fi
python -c "import litellm" 2>/dev/null || pip install litellm

# --- tau2: official tau2-bench (sierra-research) ------------------------------
if python -c "import tau2" 2>/dev/null; then
  log "tau2-bench importable"
elif [ -d /home/ec2-user/tau2-bench/src/tau2 ]; then
  # bind-mounted copy already on disk (code+data+patches) -> register editable,
  # no clone / no internet needed (this is the current-root resume path).
  log "tau2-bench present on disk -> pip install -e (no clone)"
  cd /home/ec2-user/tau2-bench && pip install -e . >/dev/null 2>&1 || pip install -e .
else
  log "tau2-bench missing -> clone + install"
  cd /home/ec2-user && rm -rf tau2-bench
  git clone --depth 1 https://github.com/sierra-research/tau2-bench.git
  cd tau2-bench && pip install -e .
fi

# Fixes (idempotent; applied on every setup so a fresh clone keeps them).
# See eval_scai/TAU2_FORMAT_ISSUE.md. Toggle off via TAU2_LENIENT_TOOL_ARGS=0 /
# TAU2_FIRST_TOOL_CALL=0 at runtime.
python "$(dirname "$0")/_patch_tau2_lenient_args.py"   /home/ec2-user/tau2-bench || log "WARN: lenient-args patch"
python "$(dirname "$0")/_patch_tau2_first_toolcall.py" /home/ec2-user/tau2-bench || log "WARN: first-tool-call patch"

log "deps ready (tau2-bench)"
