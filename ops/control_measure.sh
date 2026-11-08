#!/bin/bash
# Score the served EXL3 control (GLM-5.3-Flash EXL3 TR3 3.5 bpw, vLLM TP2 on the Max-Q rig) on the balanced eval set,
# token-identical to the llama.cpp reference (base_q8_0_bal.kld). Runs on the NAS. Approved: ~0.5 Max-Q GPU-h.
set -u
T=${CM_LOG:-/tmp}; H=/srv/vllm-research/work/HeteroFlash; C=/srv/cache/heteroflash
REF=/srv/vllm-research/models/gguf/kld/base_q8_0_bal.kld
. "$(dirname "$0")/local.env"
t() { date -u +%H:%M:%SZ; }
ssh spark "test -s $REF" || { echo "no reference $REF"; exit 2; }
echo "$(t) extract evaluation token ids from the reference"
ssh spark "cd $H && python3 -I src/control_kld.py tokens $REF data/q_eval_balanced_v1.tokens.json"
ssh spark "cat $H/data/q_eval_balanced_v1.tokens.json" | ssh maxq "cat > $C/analysis/q_eval_balanced_v1.tokens.json"
echo "$(t) copy reference logits to the workstation over the direct link"
SPARK_LINK=SPARK_LINK_IP
ssh -n spark "cd $(dirname $REF); setsid nohup python3 -m http.server 8766 --bind $SPARK_LINK > /tmp/hf_ref_http.log 2>&1 < /dev/null & echo \$! > /tmp/hf_ref_http.pid; sleep 2"
ssh maxq "mkdir -p $C/kld && curl -fsS -o $C/kld/base_q8_0_bal.kld.part http://$SPARK_LINK:8766/$(basename $REF) && mv $C/kld/base_q8_0_bal.kld.part $C/kld/base_q8_0_bal.kld; ls -la $C/kld/"
ssh spark 'kill $(cat /tmp/hf_ref_http.pid) 2>/dev/null; rm -f /tmp/hf_ref_http.pid'
echo "$(t) configure the hook: logits capture only (no router capture)"
ssh maxq "python3 -I - <<'PY'
import json
p='$C/traces/control.json'; c=json.load(open(p))
c.update({'capture': False, 'capture_inputs': None, 'capture_logits': {'min_rows': 2000, 'from': 1024, 'to': 2047, 'debug': True}})
json.dump(c, open(p + '.tmp', 'w')); import os; os.replace(p + '.tmp', p); print(c)
PY"
ssh maxq "mkdir -p $C/traces/raw/ctl_logits_old && find $C/traces/raw -maxdepth 1 -name 'logits_*.npy' -exec mv {} $C/traces/raw/ctl_logits_old/ \; ; true"
G0=$(date +%s)
echo "$(t) start the control (trace variant: eager, hook) on the Max-Q GPUs"
ssh maxq "/srv/tools/heteroflash/ops/lok/hf-run trace 2>&1 | tail -3"
ssh maxq "curl -fs -o /dev/null -w 'health %{http_code}\n' http://127.0.0.1:8001/health" || { echo "control not healthy"; exit 3; }
echo "$(t) score 42 chunks"
ssh maxq "python3 -I $C/analysis/control_kld.py client $C/analysis/q_eval_balanced_v1.tokens.json http://127.0.0.1:8001/v1/completions | tail -3"
sleep 5
ssh maxq "python3 -I - <<'PY'
import json, os
p='$C/traces/control.json'; c=json.load(open(p)); c['capture_logits'] = None
json.dump(c, open(p + '.tmp', 'w')); os.replace(p + '.tmp', p)
PY"
ssh maxq "/srv/tools/heteroflash/ops/lok/hf-run stop >/dev/null 2>&1; true"
echo "$(t) control GPU time: $(( ($(date +%s) - G0) )) s wall on 2 GPUs"
ssh maxq "ls $C/traces/raw/logits_*.npy 2>/dev/null | wc -l; tail -3 $C/traces/raw/logits_debug_*.txt 2>/dev/null"
echo "$(t) compare against the reference"
ssh maxq "export DOCKER_HOST=unix:///run/user/1000/docker.sock; docker run --rm -v $C:/c --entrypoint python3 localhost/vllm-glm53f-exl3-sm120-turbo:r6 /c/analysis/control_kld.py compare /c/kld/base_q8_0_bal.kld /c/traces/raw /c/analysis/q_eval_balanced_v1.manifest.json /c/analysis/control_exl3_vs_q8_bal.json 2>&1 | tail -2"
echo "$(t) CONTROL_DONE"
# Sequencing (2026-11-08): the bounded pilot capture runs right after the control measurement, before any eval.
echo "$(t) run the pilot capture"
PC_LOG=$T bash "$(dirname "$0")/pilot_capture.sh" > $T/pilot_capture.log 2>&1
grep -vE "^\s*$" $T/pilot_capture.log | tail -10
