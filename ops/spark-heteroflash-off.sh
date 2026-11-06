#!/bin/bash
# Restore Spark CPU idle states saved by spark-heteroflash.sh
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
F=/srv/vllm-research/work/HeteroFlash/ops/cstate-backup-spark.txt
[ -f "$F" ] && while read -r f v; do echo "$v" > "$f"; done < "$F" && echo restored
