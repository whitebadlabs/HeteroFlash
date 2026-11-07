#!/bin/bash
# ONE-TIME on the Spark (run BEFORE moving its Cat6 from the router to the workstation):
#   ssh -t spark sudo bash /srv/vllm-research/work/HeteroFlash/ops/spark-direct-link.sh
# 1. Wi-Fi keeps the LAN: add the Spark's old wired LAN address (SPARK_LAN_IP in git-ignored ops/local.env) as a second address on the Wi-Fi connection, so everything that targets
#    the Spark's old wired address (LiteLLM route to Qwen, NAS egress fences) keeps working over Wi-Fi.
# 2. The wired port becomes the direct link: SPARK_LINK_IP/24, no gateway (never-default), no DNS, MTU 9000.
# 3. Let lok reboot the Spark without a password (polkit; nothing else).
# The network change runs detached (survives this SSH session dropping) and logs to /var/log/heteroflash-direct-link.log.
# Undo: nmcli con mod "Wired connection 3" ipv4.method auto ipv4.addresses "" ipv4.never-default no ipv4.ignore-auto-dns no 802-3-ethernet.mtu auto
#       nmcli con mod "exactly once semantics" -ipv4.addresses $SPARK_LAN_IP/24 ; nmcli con up "Wired connection 3"
set -eu
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
. "$(dirname "$0")/local.env"
: "${SPARK_LAN_IP:?set SPARK_LAN_IP in ops/local.env}"
WIFI="exactly once semantics"; WIRED="Wired connection 3"
cat > /etc/polkit-1/rules.d/50-lok-reboot.rules <<'R'
// lok may reboot the Spark over SSH without a password (HeteroFlash ops). Nothing else.
polkit.addRule(function(action, subject) {
  if (subject.user == "lok" &&
      (action.id == "org.freedesktop.login1.reboot" ||
       action.id == "org.freedesktop.login1.reboot-multiple-sessions" ||
       action.id == "org.freedesktop.login1.reboot-ignore-inhibit")) { return polkit.Result.YES; }
});
R
chmod 644 /etc/polkit-1/rules.d/50-lok-reboot.rules
systemctl reload polkit 2>/dev/null || systemctl restart polkit || true
echo "reboot permission for lok installed"
setsid nohup bash -c "
  set -x
  nmcli con mod '$WIFI' +ipv4.addresses $SPARK_LAN_IP/24 ipv4.route-metric 100
  nmcli con up '$WIFI'
  sleep 5
  nmcli con mod '$WIRED' ipv4.method manual ipv4.addresses SPARK_LINK_IP/24 ipv4.gateway '' ipv4.never-default yes \
        ipv4.ignore-auto-dns yes ipv4.dns '' ipv6.method link-local 802-3-ethernet.mtu 9000 connection.autoconnect yes
  nmcli con up '$WIRED' || true
  sleep 3
  ip -br addr; ip route
" > /var/log/heteroflash-direct-link.log 2>&1 < /dev/null &
echo "network change applying in the background (log: /var/log/heteroflash-direct-link.log)."
echo "Your SSH session may drop; reconnect after ~30 s (Wi-Fi keeps $SPARK_LAN_IP and its DHCP address)."
