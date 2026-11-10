#!/bin/bash
# Restricted decode-benchmark entry point for a delegated agent (forced SSH command on the Spark lok account).
# The caller's command arrives in SSH_ORIGINAL_COMMAND and is parsed against an allowlist; nothing else runs.
#   run CONFIG MTP_N [KV_LAYOUT] [V2RUNNER]   start one decode benchmark in the background (one at a time)
#       CONFIG   fp8 | nvfp4attn        (GPU-only 2-bit-expert checkpoints with MTP layer)
#       MTP_N    1..5                   (MTP speculative tokens)
#       KV       BLHNC (default) | LBNHC
#       V2RUNNER 1 (default) | 0
#   status | log | results | list      read-only
# Never stops or restarts other containers: refuses to start if any container is running.
set -u
H=/srv/vllm-research/work/HeteroFlash; W=/srv/vllm-research/work/cache/codex_bench
OUT=$H/results/2026-11-08-route3; LOCK=$W/lock; LOG=$W/current.log
read -r -a A <<< "${SSH_ORIGINAL_COMMAND:-${*:-}}"
cmd=${A[0]:-help}
case "$cmd" in
  status)
    if [ -e $W/running ]; then echo "running: $(cat $W/running)"; else echo idle; fi
    docker ps --format '{{.Names}} {{.Status}}'; exit 0 ;;
  log) tail -n 60 $LOG 2>/dev/null || echo "no log"; exit 0 ;;
  results) for f in $(ls -t $W/*.result 2>/dev/null | head -20); do echo "== $(basename $f .result)"; cat "$f"; done; exit 0 ;;
  list) echo "configs: fp8 nvfp4attn | MTP_N 1-5 | KV BLHNC LBNHC | V2RUNNER 1 0"; exit 0 ;;
  run) ;;
  *) sed -n 2,11p "$0"; exit 2 ;;
esac
cfg=${A[1]:-}; n=${A[2]:-}; kv=${A[3]:-BLHNC}; v2=${A[4]:-1}
case "$cfg" in fp8) M=/srv/vllm-research/models/mixed/full_iq2xxs_fp8;; nvfp4attn) M=/srv/vllm-research/models/mixed/full_iq2xxs_nvfp4attn;; *) echo "bad CONFIG"; exit 2;; esac
[[ "$n" =~ ^[1-5]$ ]] || { echo "bad MTP_N"; exit 2; }
[[ "$kv" =~ ^(BLHNC|LBNHC)$ ]] || { echo "bad KV_LAYOUT"; exit 2; }
[[ "$v2" =~ ^[01]$ ]] || { echo "bad V2RUNNER"; exit 2; }
[ -z "$(docker ps -q)" ] || { echo "refusing: containers are running (this script never stops services):"; docker ps --format '{{.Names}}'; exit 3; }
exec 9>$LOCK; flock -n 9 || { echo "refusing: a benchmark is already running"; exit 3; }
tag="codex_${cfg}_mtp${n}_${kv}_v2r${v2}_$(date -u +%m%d%H%M)"
echo "$tag" > $W/running
setsid bash -c "
  cd $H
  HF_WINDOW_OK=1 RESTART_QFN=0 SKIP_LONG=1 MTP_N=$n KV_LAYOUT=$kv V2RUNNER=$v2 timeout 3600 bash ops/route3/serve_eval.sh $M $tag mtp > $LOG 2>&1
  { echo \"config=$cfg mtp_n=$n kv=$kv v2runner=$v2\"; grep -h decode_tok_s $OUT/${tag}_mtp${n}_speed.txt 2>/dev/null || echo 'no result (see log)'; } > $W/$tag.result
  docker rm -f hf-r3 >/dev/null 2>&1; rm -f $W/running
" 9>&- </dev/null >/dev/null 2>&1 &
echo "started $tag (about 10-15 min: model load ~8 min). Poll with: status | log | results"
