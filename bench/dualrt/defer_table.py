#!/usr/bin/env python3
"""2026-11-18 deferral: one row per quality pass (vs Q8: dNLL, KLD, top-1; per-domain KLD; paired KLD vs the same-boot
reference). Usage: defer_table.py FO TAG REF name1 name2 ..."""
import json, os, sys
fo, tag, ref, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
DOM = ["writing_chat", "knowledge_qa", "math", "code", "multilingual", "instruction_following", "long_context"]
print("| point | chunks | dNLL (SE) | KLD | top-1 | worst domain KLD | code KLD / top-1 | paired KLD vs %s |" % ref)
print("|---|---|---|---|---|---|---|---|")
for n in names:
    f = os.path.join(fo, "q_%s_%s_vs_q8_bal.json" % (tag, n))
    if not os.path.exists(f):
        print("| %s | missing | | | | | | |" % n); continue
    d = json.load(open(f)); o = d["_overall"]
    worst = max((k for k in DOM if k in d), key=lambda k: d[k]["kld_mean"])
    p = os.path.join(fo, "q_%s_%s_vs_%s.json" % (tag, n, ref))
    pk = json.load(open(p)) if os.path.exists(p) else None
    print("| %s | %s/%s | %+.4f (%.4f) | %.3f | %.1f%% | %s %.3f | %.3f / %.1f%% | %s |" % (
        n, o.get("chunks_scored", "?"), o.get("chunks_total", "?"), o["dnll_mean"], o.get("dnll_se", float("nan")),
        o["kld_mean"], 100 * o["same_top"], worst, d[worst]["kld_mean"], d["code"]["kld_mean"], 100 * d["code"]["same_top"],
        "%.4f +-%.4f" % (pk["kld_mean"], pk["kld_se"]) if pk else "-"))
