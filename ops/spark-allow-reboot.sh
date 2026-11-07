#!/bin/bash
# ONE-TIME on the Spark: let lok reboot/power-off the Spark without a password (polkit rule; nothing else is granted),
# then reboot now.   ssh -t spark sudo bash /srv/vllm-research/work/HeteroFlash/ops/spark-allow-reboot.sh
# Undo: sudo rm /etc/polkit-1/rules.d/50-lok-reboot.rules
set -eu
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
cat > /etc/polkit-1/rules.d/50-lok-reboot.rules <<'R'
// lok may reboot the Spark over SSH without a password (HeteroFlash ops). Nothing else.
polkit.addRule(function(action, subject) {
  if (subject.user == "lok" &&
      (action.id == "org.freedesktop.login1.reboot" ||
       action.id == "org.freedesktop.login1.reboot-multiple-sessions" ||
       action.id == "org.freedesktop.login1.reboot-ignore-inhibit")) {
    return polkit.Result.YES;
  }
});
R
chmod 644 /etc/polkit-1/rules.d/50-lok-reboot.rules
systemctl reload polkit 2>/dev/null || systemctl restart polkit
sleep 2
echo "lok can now reboot: $(sudo -u lok busctl call org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager CanReboot)"
echo "rebooting now"
systemctl reboot
