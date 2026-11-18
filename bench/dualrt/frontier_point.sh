#!/bin/bash
# 2026-11-17 frontier: one speed point on a held split server (serve_hold.sh): set run-time switches, run the client,
# snapshot exchange/assign counters before and after.
#   frontier_point.sh TAG NAME "LMH KDA FRS" "A B" "DROP" "ALT" [ONLY] [REPS]
#   DROP = "K TAU RENORM" or "json {...} TAU RENORM" (- = unchanged); ALT = "l1,l2,.." or none (- = unchanged)
set -u
TAG=$1; NAME=$2; G=$3; AB=$4; DROP=$5; ALT=$6; ONLY=${7:-essay,chat,hermes}; REPS=${8:-1}
H=/srv/vllm-research/work/HeteroFlash; OUT=$H/results/2026-11-17-dualrt; FO=${FO:-$H/results/2026-11-17-frontier}; mkdir -p $FO
echo "$G" > $OUT/drt_gates.req; echo "$AB" > $OUT/drt_ab.req
[ "$DROP" != - ] && echo "$DROP" > $OUT/drt_drop.req
[ "$ALT" != - ] && echo "$ALT" > $OUT/drt_alt.req
[ -n "${DEFER:-}" ] && echo "$DEFER" > $OUT/drt_defer.req   # 2026-11-18: "none|all|l1,.. LATE"
for i in $(seq 40); do curl -s -m 60 http://127.0.0.1:8095/v1/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","prompt":"Hi","max_tokens":2}' >/dev/null
  [ -e $OUT/drt_gates.req ] || [ -e $OUT/drt_ab.req ] || [ -e $OUT/drt_drop.req ] || [ -e $OUT/drt_alt.req ] || [ -e $OUT/drt_defer.req ] || break; done
touch $OUT/${TAG}_stats.json.req; sleep 3; cp $OUT/${TAG}_stats.json $FO/${TAG}_${NAME}_stats0.json
ONLY=$ONLY REPS=$REPS python3 -I $H/bench/dualrt/client.py > $FO/${TAG}_${NAME}_client.jsonl 2>> $FO/${TAG}_client.err
touch $OUT/${TAG}_stats.json.req; sleep 3; cp $OUT/${TAG}_stats.json $FO/${TAG}_${NAME}_stats1.json
python3 - $FO/${TAG}_${NAME}_stats0.json $FO/${TAG}_${NAME}_stats1.json "$G" "$AB" "$DROP" "$ALT" <<'PY'
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
out = {"gates": sys.argv[3], "ab": sys.argv[4], "drop": sys.argv[5], "alt": sys.argv[6], "defer": __import__("os").environ.get("DEFER", "")}
for k in ("mailbox_decode_T<=4_req_pairs_serverus_relayus", "dense_req_serverus_relayus"):
    d = [y - x for x, y in zip(a[k], b[k])]; out[k.split("_")[0] + "_" + k.split("_")[1]] = {"req": d[0], "per_req": [round(v / max(d[0], 1), 1) for v in d[1:]]}
out["alt_requests"] = b.get("alt_requests", 0) - a.get("alt_requests", 0)
pa, pb = a["per_layer_calls_U_cap_R"], b["per_layer_calls_U_cap_R"]
c = sum(pb[k][0] - pa[k][0] for k in pa) or 1
out["assign"] = {"windows": c, "U": round(sum(pb[k][1] - pa[k][1] for k in pa) / c, 2), "gpu": round(sum(pb[k][2] - pa[k][2] for k in pa) / c, 2),
                 "resident_in_union": round(sum(pb[k][3] - pa[k][3] for k in pa) / c, 2)}
print(json.dumps(out))
PY
python3 $H/bench/dualrt/summarize.py $FO/${TAG}_${NAME}_client.jsonl
