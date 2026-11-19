#!/bin/bash
# 2026-11-19: wait for MaxQ traces (ecorr-more drive.sh TRACE_DONE), copy to data/<name>/, run analyze.py as systemd user units.
DATA=NAS_PATH; HERE=$DATA/repo; R=/srv/cache/heteroflash/ecorr-more
declare -A OUTN=([dsv2]=dsv2lite_v1 [qwen3moe]=qwen3moe_v1 [gptoss]=gptoss_v1)
declare -A LOG=([dsv2]=drive1b [qwen3moe]=drive0b [gptoss]=drive1)
pending="dsv2 qwen3moe"
while [ -n "$pending" ]; do
  left=""
  for n in $pending; do
    st=$(ssh maxq "grep -E \"$n (TRACE_DONE|smoke failed|download failed)|skip.*$n\" $R/logs/${LOG[$n]}.log 2>/dev/null | tail -1")
    if echo "$st" | grep -q TRACE_DONE; then
      mkdir -p $DATA/data/$n; o=${OUTN[$n]}
      ssh maxq "cat $R/routes/$o.npz" > $DATA/data/$n/$o.npz; ssh maxq "cat $R/routes/$o.json" > $DATA/data/$n/$o.json
      echo "$(date -u +%H:%M:%SZ) $n pulled $(du -h $DATA/data/$n/$o.npz | cut -f1)"
      systemd-run --user --unit="ecorr-an-$n" --collect -p WorkingDirectory="$DATA" -E PYTHONPATH="$DATA/pylib" \
        -E OPENBLAS_NUM_THREADS=2 -E OMP_NUM_THREADS=2 -p Nice=15 \
        /bin/bash -c "exec /usr/bin/python3 -u $HERE/scripts/analyze.py --reduced data/$n/$o --out out/$n.json --train-cap 20000 --eval-cap 20000 >> out/${n}_run.log 2>&1"
    elif [ -n "$st" ]; then echo "$(date -u +%H:%M:%SZ) $n: $st"
    else left="$left $n"; fi
  done
  pending=$(echo $left); [ -n "$pending" ] && sleep 120
done
echo "$(date -u +%H:%M:%SZ) ALL_PULLED"
