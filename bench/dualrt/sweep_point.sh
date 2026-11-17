#!/bin/bash
# One measurement point on a held server: set switches + balance rule, run the client, snapshot exchange counters.
#   sweep_point.sh TAG NAME "LMH KDA" "A B" [ONLY] [REPS]
set -u
TAG=$1; NAME=$2; G=$3; AB=$4; ONLY=${5:-essay,chat}; REPS=${6:-1}
H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt
echo "$G" > $OUT/drt_gates.req; echo "$AB" > $OUT/drt_ab.req
for i in $(seq 30); do curl -s -m 60 http://127.0.0.1:8095/v1/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","prompt":"Hi","max_tokens":2}' >/dev/null; [ -e $OUT/drt_gates.req ] || [ -e $OUT/drt_ab.req ] || break; done
touch $OUT/${TAG}_stats.json.req; sleep 3; cp $OUT/${TAG}_stats.json $OUT/${TAG}_${NAME}_stats0.json
ONLY=$ONLY REPS=$REPS python3 -I $H/bench/dualrt/client.py > $OUT/${TAG}_${NAME}_client.jsonl 2>> $OUT/${TAG}_client.err
touch $OUT/${TAG}_stats.json.req; sleep 3; cp $OUT/${TAG}_stats.json $OUT/${TAG}_${NAME}_stats1.json
python3 - $OUT/${TAG}_${NAME}_stats0.json $OUT/${TAG}_${NAME}_stats1.json <<'PY'
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
out = {}
for k in ("mailbox_decode_T<=4_req_pairs_serverus_relayus", "dense_req_serverus_relayus"):
    d = [y - x for x, y in zip(a[k], b[k])]; out[k.split("_")[0] + "_" + k.split("_")[1]] = {"req": d[0], "per_req": [round(v / max(d[0], 1), 1) for v in d[1:]]}
pa, pb = a["per_layer_calls_U_cap_R"], b["per_layer_calls_U_cap_R"]
c = sum(pb[k][0] - pa[k][0] for k in pa) or 1
out["assign"] = {"windows": c, "U": round(sum(pb[k][1] - pa[k][1] for k in pa) / c, 2), "gpu": round(sum(pb[k][2] - pa[k][2] for k in pa) / c, 2),
                 "resident_in_union": round(sum(pb[k][3] - pa[k][3] for k in pa) / c, 2)}
print(json.dumps(out))
PY
python3 $H/bench/dualrt/summarize.py $OUT/${TAG}_${NAME}_client.jsonl
