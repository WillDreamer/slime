#!/bin/bash
# Runs INSIDE the `slime` container. (Re)launches the resume-safe tau1+tau2
# parallel eval (run_parallel.sh), unless it's already running. Detached + logged.
# Idempotent: run_parallel.sh waits for all servers, skips completed tau1 temp0
# cells, and tau2-bench auto-resumes each --save-to. Safe to call any time.
set -u
if pgrep -f "[r]un_parallel.sh" >/dev/null 2>&1; then
  echo "[_run_tau] tau eval already running — not relaunching."
  exit 0
fi
cd /home/ec2-user/slime/eval_scai
nohup setsid bash /home/ec2-user/slime/eval_scai/run_parallel.sh \
  > /home/ec2-user/slime/eval_scai/logs/MASTER_parallel.log 2>&1 < /dev/null &
echo "[_run_tau] tau1+tau2 eval (re)launched; log: eval_scai/logs/MASTER_parallel.log"
