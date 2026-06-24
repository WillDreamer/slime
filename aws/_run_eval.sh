#!/bin/bash
# Runs INSIDE the `slime-search` container. (Re)starts the resume-safe eval
# driver, unless it is already running. Detached + logged; survives this shell.
set -u
if pgrep -f "[r]un_search_eval_all.sh" >/dev/null 2>&1 || pgrep -f "[e]val_search.py" >/dev/null 2>&1; then
  echo "[_run_eval] eval already running — not relaunching."
  exit 0
fi
cd /home/ec2-user/slime/eval_scai/benchmarks/search
nohup bash /home/ec2-user/slime/aws/run_search_eval_all.sh \
  > /data/Search_data/eval_all.log 2>&1 &
echo "[_run_eval] eval (re)launched; log: /data/Search_data/eval_all.log"
