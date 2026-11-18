#!/usr/bin/env python3
"""GPU wait accounting for one speed point (2026-11-18 deferral study): diff of the hook's stats dumps before and after
the point (frontier_point.sh stats0/stats1), plus the client run for cycle time.
  waits.py STATS0.json STATS1.json CLIENT.jsonl
Per MoE layer, decode windows (T <= 4): blocked ms per window (wait kernel start -> reply seen, GPU idle), GPU work
between pack and wait (overlap), zero-wait share. Dense (lm_head) waits too. Ceiling: the cycle if every MoE wait were
hidden (no change in acceptance), and the serial-relay floor (42 x decode round trip) that deferral cannot go below."""
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
rows = [json.loads(l) for l in open(sys.argv[3]) if l.strip()]
wa, wb = a["wait"], b["wait"]
lay = {}
for L in wb["decode_main"]:
    d = [y - x for x, y in zip(wa["decode_main"][L], wb["decode_main"][L])]
    dl = [y - x for x, y in zip(wa["decode_late"][L], wb["decode_late"][L])]
    n = max(d[0], 1)
    lay[L] = {"n": d[0], "blocked_ms": round(d[1] / n / 1e6, 4), "overlap_ms": round(d[2] / n / 1e6, 4), "zero_share": round(d[3] / n, 3),
              "late_n": dl[0], "late_blocked_ms": round(dl[1] / max(dl[0], 1) / 1e6, 4)}
nwin = max(v["n"] for v in lay.values())
blocked = sum(v["blocked_ms"] * v["n"] for v in lay.values()) / max(nwin, 1)
late_blocked = sum(v["late_blocked_ms"] * v["late_n"] for v in lay.values()) / max(nwin, 1)
dd = [y - x for x, y in zip(wa["dense"], wb["dense"])]
dense = [{"slot": i, "n": dd[2 * i], "blocked_ms": round(dd[2 * i + 1] / max(dd[2 * i], 1) / 1e6, 4)} for i in range(len(dd) // 2)]
mk = "mailbox_decode_T<=4_req_pairs_serverus_relayus"
md = [y - x for x, y in zip(a[mk], b[mk])]
rt_ms = md[3] / max(md[0], 1) / 1e3; srv_ms = md[2] / max(md[0], 1) / 1e3
tok = sum(r["tokens"] for r in rows); s = sum(r["s"] for r in rows); cyc = sum(r["drafts"] for r in rows)
cycle_ms = 1e3 * s / max(cyc, 1)
dense_per_win = sum(x["n"] * x["blocked_ms"] for x in dense) / max(nwin, 1)
out = {"windows_T<=4": nwin, "cycles_client": cyc, "tok_s": round(tok / s, 2), "committed_per_cycle": round(tok / max(cyc, 1), 3),
       "ms_per_cycle_incl_prefill": round(cycle_ms, 1),
       "moe_blocked_ms_per_window": round(blocked, 2), "late_slot_blocked_ms_per_window": round(late_blocked, 2),
       "dense_blocked_ms_per_window": round(dense_per_win, 2),
       "decode_req_per_window": round(md[0] / max(nwin, 1), 1), "decode_round_trip_ms": round(rt_ms, 3), "decode_server_ms": round(srv_ms, 3),
       "serial_relay_floor_ms": round(md[0] / max(nwin, 1) * rt_ms, 1),
       "ceiling_ms_per_cycle_all_moe_waits_hidden": round(cycle_ms * cyc / max(nwin, 1) - blocked, 1) if nwin else None,
       "per_layer": lay, "dense": dense}
w_per_cyc = nwin / max(cyc, 1)
out["windows_per_cycle"] = round(w_per_cyc, 3)
out["ceiling_tok_s_same_acceptance"] = round(tok / (s - blocked * nwin / 1e3), 2)
print(json.dumps(out))
