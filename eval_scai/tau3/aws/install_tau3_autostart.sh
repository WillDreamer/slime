#!/bin/bash
# Install the tau3 eval auto-resume as a boot service. Run once per machine
# (or bake into the AMI). Copies the autostart script to /usr/local/bin (root fs,
# so it runs even before the data volume is mounted) and enables the systemd unit.
#   sudo /mnt/old-data3/home/ec2-user/slime/eval_scai/tau3/aws/install_tau3_autostart.sh
# Add 'start' to also kick it off now:  sudo install_tau3_autostart.sh start
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

install -m 0755 "$HERE/autostart-tau3.sh" /usr/local/bin/autostart-tau3.sh
install -m 0644 "$HERE/tau3eval.service"  /etc/systemd/system/tau3eval.service
systemctl daemon-reload
systemctl enable tau3eval
echo "installed + enabled tau3eval.service (script -> /usr/local/bin/autostart-tau3.sh)"
if [ "${1:-}" = "start" ]; then
  echo "starting tau3eval now..."
  systemctl start tau3eval
  systemctl --no-pager status tau3eval | head -8
fi
