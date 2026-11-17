#!/bin/bash
# Symmetric 90-phase swap over the direct 10 GbE link: Spark client (SPARK_LINK_IP) <-> MaxQ server (WS_LINK_IP).
# usage: link_suite.sh <label> [msgs] [computes]
lab=$1; MSGS=${2:-"8192 32768 65536"}; CUS=${3:-"0 50 150 300"}; port=47100
for m in $MSGS; do for cu in $CUS; do for mh in "0 0" "1 0" "1 20"; do set -- $mh; port=$((port+1))
  ssh maxq "cd /srv/cache/heteroflash/dualmem && (timeout 120 ./phase_swap s WS_LINK_IP $port $m $cu $1 $2 >/dev/null 2>&1 &)"
  sleep 0.5
  echo -n "$lab "; ssh spark "/srv/vllm-research/runs/hfdual/phase_swap c WS_LINK_IP $port $m $cu $1 90 150 $2"
done; done; done
