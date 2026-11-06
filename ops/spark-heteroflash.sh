#!/bin/bash
# HeteroFlash link prep on the Spark (run once):  ssh -t spark sudo bash /srv/vllm-research/work/HeteroFlash/ops/spark-heteroflash.sh
#   1. ConnectX-7 / QSFP diagnosis (read-only)   2. install RDMA tools   3. disable deep CPU idle states (temporary)
# Undo 3: sudo bash /srv/vllm-research/work/HeteroFlash/ops/spark-heteroflash-off.sh   (reboot also undoes it)
set -u
OPS=/srv/vllm-research/work/HeteroFlash/ops
OUT=/srv/vllm-research/heteroflash-link-$(date -u +%Y%m%dT%H%M%SZ).log
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
exec > >(tee -a "$OUT") 2>&1
echo "== 1. ConnectX diagnosis"
lspci -nn | grep -iE "15b3|mellanox|connectx" || echo "no Mellanox/ConnectX PCI device listed"
ip -br link
ls /sys/class/infiniband 2>&1
lsmod | grep -E "^mlx5" || echo "mlx5 modules not loaded"
journalctl -k -b --no-pager 2>/dev/null | grep -iE "mlx5|connectx" | tail -15
echo "== 2. RDMA tools"
DEBIAN_FRONTEND=noninteractive apt-get install -y -q rdma-core ibverbs-utils perftest ethtool >/dev/null 2>&1 && echo installed || echo "apt install failed"
ibv_devinfo 2>&1 | head -20
echo "== 3. CPU idle states"
[ -f "$OPS/cstate-backup-spark.txt" ] || for f in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do echo "$f $(cat "$f")"; done > "$OPS/cstate-backup-spark.txt"
n=0; for d in /sys/devices/system/cpu/cpu*/cpuidle/state*; do [ "$(cat "$d/latency" 2>/dev/null || echo 0)" -gt 0 ] && { echo 1 > "$d/disable"; n=$((n+1)); }; done
echo "disabled $n deep idle-state entries"
chown "${SUDO_USER:-lok}" "$OUT" 2>/dev/null; echo "log: $OUT  - tell Claude"
