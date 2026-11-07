#!/bin/bash
# ONE-TIME on the workstation: second 10G port = direct link to the Spark: WS_LINK_IP/24, no gateway, no DNS, MTU 9000.
#   ssh -t maxq sudo bash /srv/tools/heteroflash/ops/maxq-direct-link.sh
# Undo: sudo rm /etc/netplan/60-heteroflash-direct.yaml && sudo netplan apply
set -eu
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
cat > /etc/netplan/60-heteroflash-direct.yaml <<'Y'
network:
  version: 2
  ethernets:
    enp209s0f1np1:
      dhcp4: false
      dhcp6: false
      addresses: [WS_LINK_IP/24]
      mtu: 9000
      optional: true
Y
chmod 600 /etc/netplan/60-heteroflash-direct.yaml
netplan generate && netplan apply
sleep 3
ip -br addr show enp209s0f1np1; ip route | grep -v "^default" | head; ip route | grep "^default"
echo "done: the default route must still be via the main port (enp209s0f0np0)"
