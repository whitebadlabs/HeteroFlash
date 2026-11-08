import json, re
B = "results/2026-11-08-balanced-eval"
order = ["q6_K", "q4k", "G0_q4kim", "GL_iq3late", "GM_iq3mid", "GS_iq3small", "mixA_q4k3-15_iq3xxs16-44", "iq3_xxs"]
doms = ["writing_chat", "knowledge_qa", "math", "code", "multilingual", "instruction_following", "long_context"]
out = {}
for q in order:
    d = json.load(open(f"{B}/{q}.json"))
    s = open(f"{B}/{q}_summary.txt").read()
    m = re.search(r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*(-?[\d.]+)\s*±\s*([\d.]+)", s)
    k = re.search(r"Mean    KLD:\s*([\d.]+)", s)
    t = re.search(r"Same top p:\s*([\d.]+)", s)
    row = {"dnll": float(m.group(1)), "dnll_se": float(m.group(2)), "kld": float(k.group(1)), "top": float(t.group(1)),
           "by_domain": {x: d[x] for x in doms if x in d}}
    out[q] = row
    dd = " ".join("%s:%+.3f/%.3f" % (x[:6], d[x]["dnll"], d[x]["kld"]) for x in doms if x in d)
    print("%-26s dNLL %+.4f±%.4f KLD %.4f top %.1f | %s" % (q, row["dnll"], row["dnll_se"], row["kld"], row["top"], dd))
json.dump(out, open(f"{B}/summary.json", "w"), indent=1)
