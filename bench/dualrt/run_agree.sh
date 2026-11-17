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
bash $H/bench/dualrt/serve.sh $TAG agree || { docker rm -f hf-drt >/dev/null 2>&1; exit 1; }
python3 -I $H/bench/dualrt/agree_client.py $OUT/agree_texts.json $OUT/${TAG}_agree.json > $OUT/${TAG}_agree.log 2>&1
[ -n "${HF_SPLIT_STATS:-}" ] && { touch $OUT/$(basename $HF_SPLIT_STATS).req; sleep 4; }
docker logs hf-drt > $OUT/${TAG}_serve.log 2>&1; docker rm -f hf-drt >/dev/null 2>&1
echo "$(date -u +%H:%M:%SZ) DONE $TAG"
