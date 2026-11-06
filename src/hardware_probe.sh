#!/bin/sh
# Read-only inventory for each machine; review output before publicly sharing it.
set -u
printf '%s\n' '=== CPU architecture ==='
uname -m
command -v lscpu >/dev/null 2>&1 && lscpu || true
printf '%s\n' '=== memory ==='
command -v free >/dev/null 2>&1 && free -h || true
command -v numactl >/dev/null 2>&1 && numactl --hardware || true
printf '%s\n' '=== GPU inventory (no GPU workload) ==='
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv || true
printf '%s\n' '=== NIC link device names, no IP addresses ==='
if command -v ethtool >/dev/null 2>&1; then
  for p in /sys/class/net/*; do
    name=${p##*/}
    printf 'NIC %s: ' "$name"
    ethtool "$name" 2>/dev/null | grep -E 'Speed:|Link detected:' | tr '\n' ' ' || true
    printf '\n'
  done
fi
printf '%s\n' '=== RDMA devices ==='
command -v ibv_devinfo >/dev/null 2>&1 && ibv_devinfo | head -80 || true
printf '%s\n' '=== version info ==='
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -4 || true
command -v python3 >/dev/null 2>&1 && python3 --version || true
