#!/bin/bash
# Undo maxq-heteroflash.sh: stop the trace container, restore CPU idle states. Trace data is kept.
set -u
OPS=/srv/tools/heteroflash/ops
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
DD="sudo -u dockerd env XDG_RUNTIME_DIR=/run/user/1002 DOCKER_HOST=unix:///run/user/1002/docker.sock"
(cd "$OPS/maxq-trace" && $DD docker compose -f compose.run.yaml down) && echo "glm53-trace stopped"
if [ -f "$OPS/cstate-backup.txt" ]; then
  while read -r f v; do echo "$v" > "$f"; done < "$OPS/cstate-backup.txt" && echo "CPU idle states restored"
fi
