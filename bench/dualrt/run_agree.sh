#!/bin/bash
# Teacher-forced agreement run: agree-mode server (no MTP, eager, 16-token batches, no prefix cache) + agree_client.py.
#   run_agree.sh TAG   (env: HF_SPLIT, HF_SPLIT_* as for serve.sh)
set -u
TAG=$1; H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt
[ -s $OUT/agree_texts.json ] || python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("/srv/vllm-research/work/HeteroFlash/results/2026-11-17-dualrt/cfgB_base_r1_client.jsonl")]
pick = {"essay_completions": "Write a long essay about rivers.", "prose": None, "hermes:tc_simple_python_357": None}
texts = []
for r in rows:
    if r["rep"] == 0 and r["workload"] in pick:
        t = (pick[r["workload"]] or "") + r["text"].replace("|", "\n")
        texts.append({"id": r["workload"], "text": t[:2400]})
json.dump(texts, open("/srv/vllm-research/work/HeteroFlash/results/2026-11-17-dualrt/agree_texts.json", "w"), indent=1)
print(len(texts), "texts")
PY
export HF_SPLIT_STATS=${HF_SPLIT_STATS:-/out/${TAG}_stats.json}
rm -f $OUT/drt_load.done
bash $H/bench/dualrt/serve.sh $TAG agree || { docker rm -f hf-drt >/dev/null 2>&1; exit 1; }
if [ "${HF_SPLIT:-0}" = 1 ] && [ "${HF_SPLIT_NRES:-0}" -gt 0 ]; then
  touch $OUT/drt_load.req
  for i in $(seq 120); do
    curl -s -m 120 http://127.0.0.1:8095/v1/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","prompt":"Hello","max_tokens":4}' >/dev/null
    [ -s $OUT/drt_load.done ] && break; sleep 2
  done
  echo "$(date -u +%H:%M:%SZ) $(cat $OUT/drt_load.done 2>/dev/null || echo 'pool load did not report')"
fi
python3 -I $H/bench/dualrt/agree_client.py $OUT/agree_texts.json $OUT/${TAG}_agree.json > $OUT/${TAG}_agree.log 2>&1
touch $OUT/${TAG}_stats.json.req; sleep 4
docker logs hf-drt > $OUT/${TAG}_serve.log 2>&1; docker rm -f hf-drt >/dev/null 2>&1
echo "$(date -u +%H:%M:%SZ) DONE $TAG"
