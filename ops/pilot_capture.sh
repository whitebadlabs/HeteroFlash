#!/bin/bash
# QAT pilot step 1: bounded all-domain capture of MoE-layer inputs on the control (EXL3, Max-Q rig).
# Budget (pre-registered, docs/QAT_PILOT_PROTOCOL.md): <= 2 GPU-hours on both GPUs => hard cap 55 min wall.
# Candidate layers (model layer -> MoE idx): 3->0, 17->14, 25->22, 32->29, 42->39, 43->40. Runs on the NAS.
set -u
T=${PC_LOG:-/tmp}; C=/srv/cache/heteroflash
LAYERS="0 14 22 29 39 40"
CAP_MIN=${PC_MINUTES:-55}
t() { date -u +%H:%M:%SZ; }
ssh maxq "touch $C/GPU_RUN.lock"
trap 'ssh maxq "/srv/tools/heteroflash/ops/lok/hf-run stop >/dev/null 2>&1; rm -f $C/GPU_RUN.lock"; echo "$(t) lock released"' EXIT
echo "$(t) wait for any running eval (rpc-server) and the control measurement to finish"
while ssh maxq 'pgrep -x ggml-rpc-server >/dev/null'; do sleep 60; done
while ! grep -q CONTROL_DONE $T/control_measure.log 2>/dev/null; do sleep 60; done
ssh maxq "mkdir -p $C/traces/raw/pilot_capture"
echo "$(t) start the capture server (eager + hook, max-num-seqs 32)"
G0=$(date +%s)
ssh maxq "/srv/tools/heteroflash/ops/lok/hf-run capture 2>&1 | tail -2"
ssh maxq "curl -fs -o /dev/null -w 'health %{http_code}\n' http://127.0.0.1:8001/health" || { echo "capture server not healthy"; exit 3; }
echo "$(t) capture (all 6 domains, layers $LAYERS), hard cap $CAP_MIN min"
ssh maxq "cd $C && timeout $((CAP_MIN * 60)) python3 -I $C/analysis/capture_client.py --prompts $C/prompts/m1_calib_v1.jsonl --control $C/traces/control.json --layers $LAYERS --tokens-per-domain ${PC_TOKENS:-150000} --max-tokens 2048 --concurrency 32 --log $C/traces/pilot_capture_log.jsonl 2>&1 | tail -8"
ssh maxq "python3 -I - <<'PY'
import json, os
p='$C/traces/control.json'; c=json.load(open(p)); c['capture_inputs'] = None
json.dump(c, open(p + '.tmp', 'w')); os.replace(p + '.tmp', p)
PY"
sleep 45
ssh maxq "/srv/tools/heteroflash/ops/lok/hf-run stop >/dev/null 2>&1; true"
W=$(( $(date +%s) - G0 ))
echo "$(t) capture wall ${W}s = $(python3 -c "print(round($W*2/3600,2))") GPU-hours (2 GPUs incl. load)"
ssh maxq "ls $C/traces/raw/inputs_*_[0-9]*_*.npz 2>/dev/null | sed -E 's/.*inputs_([a-z_]+)_[0-9]+_.*/\1/' | sort | uniq -c; du -ch $C/traces/raw/inputs_*_*_*.npz 2>/dev/null | tail -1"
echo "$(t) PILOT_CAPTURE_DONE"
