#!/bin/bash
# Restart the glm53-trace container so it loads the current trace hook and compose (e.g. hook v2 for M2 routing surgery).
#   ssh -t maxq sudo bash /srv/tools/heteroflash/ops/maxq-trace-restart.sh
set -u
OPS=/srv/tools/heteroflash/ops
TR=/srv/cache/heteroflash/traces
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
DD="sudo -u dockerd env XDG_RUNTIME_DIR=/run/user/1002 DOCKER_HOST=unix:///run/user/1002/docker.sock"
CFG=${1:-'{"mode":0,"cudagraph_mode":"NONE"}'}
(cd "$OPS/maxq-trace" && $DD docker compose -f compose.run.yaml down) >/dev/null 2>&1
python3 -I -c 'import json,sys; s=open(sys.argv[1]).read(); open(sys.argv[2],"w").write(s.replace(chr(34)+"__COMPILATION_CONFIG__"+chr(34), json.dumps(sys.argv[3])))' \
  "$OPS/maxq-trace/compose.yaml" "$OPS/maxq-trace/compose.run.yaml" "$CFG"
chmod 644 "$OPS/maxq-trace/compose.run.yaml"
(cd "$OPS/maxq-trace" && $DD docker compose -f compose.run.yaml up -d) || exit 1
for i in $(seq 120); do
  sleep 20
  [ "$($DD docker inspect -f '{{.State.Status}}' glm53-trace 2>/dev/null)" = running ] || { $DD docker logs --tail 40 glm53-trace; echo FAILED; exit 1; }
  curl -fs -o /dev/null http://127.0.0.1:8001/health && { echo "$(date -u +%H:%M:%SZ) healthy - tell Claude"; exit 0; }
done
echo "timed out waiting for health"; exit 1
