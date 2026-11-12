#!/bin/bash
# Restricted Spark entry point for a delegated agent (runs as `agents` or `lok`). Arguments come from
# SSH_ORIGINAL_COMMAND or $* and are parsed against an allowlist; nothing else runs.
#   run CONFIG MTP_N [KV] [V2]   one decode benchmark in the background, then shut down
#       CONFIG fp8 | nvfp4attn (GPU-only 2-bit experts) ; MTP_N 1..5 ; KV BLHNC|LBNHC ; V2 1|0
#   serve CONFIG [MTP_N] [HOURS] start an interactive OpenAI-compatible server on the Spark at 127.0.0.1:8095
#       CONFIG fp8 | nvfp4attn | quality  (quality = FP8 fixed path + all experts as Q4_K on the workstation CPU;
#              needs the workstation expert server up) ; MTP_N default 3 ; HOURS 1..12, default 4 (auto-stop)
#       Chat: curl http://127.0.0.1:8095/v1/chat/completions (model "GLM-5.3-Flash"; tools + reasoning parsers on)
#   stop                          stop the server started by `serve`
#   status | log | results | list read-only
# Never stops or restarts other containers: refuses to start if any container is running.
set -u
export DOCKER_HOST=unix:///var/run/docker.sock   # system daemon holds the serving image (agents defaults to its rootless daemon)
H=/srv/vllm-research/work/HeteroFlash; W=/srv/vllm-research/work/cache/codex_bench
OUT=$W/out; LOCK=$W/lock; LOG=$W/current.log
FP8=/srv/vllm-research/models/mixed/full_iq2xxs_fp8; NV=/srv/vllm-research/models/mixed/full_iq2xxs_nvfp4attn
read -r -a A <<< "${SSH_ORIGINAL_COMMAND:-${*:-}}"
cmd=${A[0]:-help}
busy() { [ -z "$(docker ps -q)" ] || { echo "refusing: containers are running (this script never stops services):"; docker ps --format '{{.Names}}'; exit 3; }; }
case "$cmd" in
  status)
    if [ -e $W/serving ]; then echo "serving: $(cat $W/serving)"; elif [ -e $W/running ]; then echo "running: $(cat $W/running)"; else echo idle; fi
    docker ps --format '{{.Names}} {{.Status}}'; exit 0 ;;
  log) tail -n 60 $LOG 2>/dev/null || echo "no log"; exit 0 ;;
  results) for f in $(ls -t $W/*.result 2>/dev/null | head -20); do echo "== $(basename $f .result)"; cat "$f"; done; exit 0 ;;
  list) echo "run: fp8 nvfp4attn | MTP_N 1-5 | KV BLHNC LBNHC | V2 1 0;  serve: fp8 nvfp4attn quality [MTP_N] [HOURS 1-12]"; exit 0 ;;
  stop)
    [ -e $W/serving ] || { echo "nothing is being served by this script"; exit 0; }
    rm -f $W/serving; docker rm -f hf-r3 >/dev/null 2>&1; echo "stopped"; exit 0 ;;
  run|serve) ;;
  *) sed -n 2,14p "$0"; exit 2 ;;
esac
cfg=${A[1]:-}
if [ "$cmd" = run ]; then
  n=${A[2]:-}; kv=${A[3]:-BLHNC}; v2=${A[4]:-1}
  case "$cfg" in fp8) M=$FP8;; nvfp4attn) M=$NV;; *) echo "bad CONFIG"; exit 2;; esac
  [[ "$kv" =~ ^(BLHNC|LBNHC)$ ]] || { echo "bad KV_LAYOUT"; exit 2; }
  [[ "$v2" =~ ^[01]$ ]] || { echo "bad V2RUNNER"; exit 2; }
else
  n=${A[2]:-3}; hours=${A[3]:-4}; kv=BLHNC; v2=1; EXTRA=""
  [[ "$hours" =~ ^([1-9]|1[0-2])$ ]] || { echo "bad HOURS (1-12)"; exit 2; }
  case "$cfg" in
    fp8) M=$FP8;; nvfp4attn) M=$NV;;
    quality) M=$FP8
      . $H/ops/local.env
      timeout 3 bash -c "exec 3<>/dev/tcp/$WS_IP/47201" 2>/dev/null || { echo "refusing: the workstation Q4_K expert server is not reachable (ask the owner to start it)"; exit 3; }
      EXTRA="MB=1 EMAP=1 REMOTE_PLAN=$H/results/2026-11-08-route3/remote_plan_all.json REMOTE_ADDR=$WS_IP:47201";;
    *) echo "bad CONFIG"; exit 2;;
  esac
fi
[[ "$n" =~ ^[1-5]$ ]] || { echo "bad MTP_N"; exit 2; }
busy
exec 9>$LOCK; flock -n 9 || { echo "refusing: a benchmark or server is already running"; exit 3; }
if [ "$cmd" = run ]; then
  tag="codex_${cfg}_mtp${n}_${kv}_v2r${v2}_$(date -u +%m%d%H%M)"
  echo "$tag" > $W/running
  setsid bash -c "
    cd $H
    HF_OUT=$OUT HF_WINDOW_OK=1 RESTART_QFN=0 SKIP_LONG=1 MTP_N=$n KV_LAYOUT=$kv V2RUNNER=$v2 timeout 3600 bash ops/route3/serve_eval.sh $M $tag mtp > $LOG 2>&1
    { echo \"config=$cfg mtp_n=$n kv=$kv v2runner=$v2\"; grep -h decode_tok_s $OUT/${tag}_mtp${n}_speed.txt 2>/dev/null || echo 'no result (see log)'; } > $W/$tag.result
    docker rm -f hf-r3 >/dev/null 2>&1; rm -f $W/running
  " </dev/null >/dev/null 2>&1 &
  echo "started $tag (about 10-15 min: model load ~8 min). Poll with: status | log | results"
else
  tag="codex_serve_${cfg}_mtp${n}_$(date -u +%m%d%H%M)"
  end=$(( $(date +%s) + hours * 3600 ))
  echo "$tag starting (auto-stop after ${hours}h)" > $W/running
  setsid bash -c "
    cd $H
    if env $EXTRA HF_OUT=$OUT HF_WINDOW_OK=1 RESTART_QFN=0 MTP_N=$n timeout 3600 bash ops/route3/serve_eval.sh $M $tag serve > $LOG 2>&1 && grep -q SERVING $LOG; then
      echo \"$tag on 127.0.0.1:8095 until \$(date -d @$end '+%H:%M %Z')\" > $W/serving; rm -f $W/running
      while [ -e $W/serving ] && [ \$(date +%s) -lt $end ] && docker ps -q --filter name=^hf-r3\$ | grep -q .; do sleep 30; done
    fi
    rm -f $W/serving $W/running; docker rm -f hf-r3 >/dev/null 2>&1
  " </dev/null >/dev/null 2>&1 &
  echo "starting server $tag (about 10 min). Poll status until it says serving; then use 127.0.0.1:8095"
fi
