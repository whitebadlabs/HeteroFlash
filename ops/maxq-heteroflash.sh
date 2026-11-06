#!/bin/bash
# HeteroFlash setup on MaxQ (run once):  ssh -t maxq sudo bash /srv/tools/heteroflash/ops/maxq-heteroflash.sh
#   A. link tuning (temporary until reboot or the off-script): disable deep CPU idle states, install RDMA tools
#   B. M1 router-trace capture: start the glm53-trace container (eager, no MTP) as dockerd, verify the hook
# Undo everything: sudo bash /srv/tools/heteroflash/ops/maxq-heteroflash-off.sh
set -u
OPS=/srv/tools/heteroflash/ops
LOG=/srv/cache/heteroflash/setup-$(date -u +%Y%m%dT%H%M%SZ).log
TR=/srv/cache/heteroflash/traces
t() { date -u +%H:%M:%SZ; }
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
exec > >(tee -a "$LOG") 2>&1
echo "$(t) log: $LOG"

echo "$(t) A1. CPU idle states (backup -> $OPS/cstate-backup.txt)"
if [ ! -f "$OPS/cstate-backup.txt" ]; then
  for f in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do echo "$f $(cat "$f")"; done > "$OPS/cstate-backup.txt"
fi
n=0
for d in /sys/devices/system/cpu/cpu*/cpuidle/state*; do
  [ "$(cat "$d/latency" 2>/dev/null || echo 0)" -gt 0 ] && { echo 1 > "$d/disable" && n=$((n+1)); }
done
echo "    disabled $n deep idle-state entries (state names: $(cat /sys/devices/system/cpu/cpu0/cpuidle/state*/name | tr '\n' ' '))"

echo "$(t) A2. RDMA / NIC tools"
DEBIAN_FRONTEND=noninteractive apt-get install -y -q rdma-core ibverbs-utils perftest ethtool >/dev/null 2>&1 \
  && echo "    installed rdma-core ibverbs-utils perftest ethtool" || echo "    apt install failed (continuing)"
ibv_devinfo 2>&1 | head -5

echo "$(t) B1. trace dirs"
mkdir -p "$TR/raw"; chmod 1777 "$TR" "$TR/raw"
DD="sudo -u dockerd env XDG_RUNTIME_DIR=/run/user/1002 DOCKER_HOST=unix:///run/user/1002/docker.sock"
if $DD docker ps --format '{{.Names}}' | grep -qx glm53; then
  echo "    the serving container glm53 is running; stop it first (research mode) - nothing changed"; exit 1
fi

start_trace() {  # $1 = compilation config JSON
  python3 -I -c 'import json,sys; s=open(sys.argv[1]).read(); open(sys.argv[2],"w").write(s.replace(chr(34)+"__COMPILATION_CONFIG__"+chr(34), json.dumps(sys.argv[3])))' \
    "$OPS/maxq-trace/compose.yaml" "$OPS/maxq-trace/compose.run.yaml" "$1"
  chmod 644 "$OPS/maxq-trace/compose.run.yaml"
  (cd "$OPS/maxq-trace" && $DD docker compose -f compose.run.yaml up -d) || return 1
  i=0
  while [ $i -lt 120 ]; do   # up to 40 min
    sleep 20; i=$((i+1))
    st=$($DD docker inspect -f '{{.State.Status}}' glm53-trace 2>/dev/null)
    [ "$st" = running ] || { echo "    container status: $st"; $DD docker logs --tail 40 glm53-trace; return 2; }
    curl -fs -o /dev/null http://127.0.0.1:8001/health && return 0
  done
  return 3
}

echo "$(t) B2. start glm53-trace (boot takes ~10-15 min)"
start_trace '{"mode":0,"cudagraph_mode":"NONE"}'; rc=$?
if [ $rc = 2 ] && $DD docker logs glm53-trace 2>&1 | grep -qiE "mode|compilation"; then
  echo "    retrying with legacy compilation level key"
  (cd "$OPS/maxq-trace" && $DD docker compose -f compose.run.yaml down) >/dev/null 2>&1
  start_trace '{"level":0,"cudagraph_mode":"NONE"}'; rc=$?
fi
[ $rc = 0 ] || { echo "$(t) FAILED to start (rc=$rc). Logs kept in $LOG. Stopping."; (cd "$OPS/maxq-trace" && $DD docker compose -f compose.run.yaml down) >/dev/null 2>&1; exit 1; }
echo "$(t)     healthy"

echo "$(t) B3. probe request + hook check"
curl -fs http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":16}' >/dev/null
sleep 40
if ls "$TR"/raw/diag_*.json >/dev/null 2>&1 && ls "$TR"/raw/r0_*.npz >/dev/null 2>&1; then
  cat "$TR"/raw/diag_*.json; echo
  echo "$(t) PASS: router logits are being captured. Leave it running and tell Claude."
else
  echo "$(t) FAIL: no trace files. Module diagnostics:"; ls -la "$TR/raw"; $DD docker logs --tail 60 glm53-trace
  (cd "$OPS/maxq-trace" && $DD docker compose -f compose.run.yaml down) >/dev/null 2>&1
  exit 1
fi
