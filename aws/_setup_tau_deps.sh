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
  log "tau2-bench present"
else
  log "tau2-bench missing -> clone + install"
  cd /home/ec2-user && rm -rf tau2-bench
  git clone --depth 1 https://github.com/sierra-research/tau2-bench.git
  cd tau2-bench && pip install -e .
fi

# Fix B: make tau2 tolerant of tool calls whose `arguments` is a bare scalar
# instead of a {param: value} object (see eval_scai/TAU2_FORMAT_ISSUE.md).
# Idempotent + applied on every setup, so a fresh re-clone of tau2-bench keeps it.
# Disable at runtime with TAU2_LENIENT_TOOL_ARGS=0.
python "$(dirname "$0")/_patch_tau2_lenient_args.py" /home/ec2-user/tau2-bench \
  || log "WARN: tau2 lenient-args patch did not apply (see output above)"

log "deps ready (tau_bench + tau2-bench)"
