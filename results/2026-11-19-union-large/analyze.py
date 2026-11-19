import json, random, statistics as stt, sys
names = sys.argv[1:]
summary = {}
for n in names:
    R = json.load(open(f"{n}_results.json")); r = R["results"]
    ks = sorted({v["K"] for k, v in r.items() if k != "hookref"}, reverse=True)
    maxp = {}
    for ci, dom, p, *_ in r[f"k{ks[0]}"]["per_chunk"]:
        maxp[dom] = max(maxp.get(dom, 0), p)
    hk = max(c["meankl"] for c in R["checks"]["hookref"])
    print(f"\n=== {n}  top_k={R['top_k']}  chunks={len(R['chunks'])}  tokens={r['hookref']['ntok']}  checks: rerun={R['checks']['unhooked_rerun']['meankl']} hookref_max={hk}")
    print("K | plainKL | unionKL | ratio | ratio14 | medChunkRatio | boot95 | top1 plain/union | reads/win plain/union/full | wins | tail rec | kept/tok union")
    summary[n] = {}
    for K in ks:
        p, u = r[f"k{K}"], r[f"k{K}u"]
        pc = {c[0]: c for c in p["per_chunk"]}; uc = {c[0]: c for c in u["per_chunk"]}
        ids = sorted(pc)
        wins = sum(uc[i][3] < pc[i][3] for i in ids)
        sub = [i for i in ids if pc[i][2] == 0 or pc[i][2] == maxp[pc[i][1]]]
        r14 = sum(uc[i][3] for i in sub) / sum(pc[i][3] for i in sub)
        med = stt.median(uc[i][3] / pc[i][3] for i in ids)
        rng = random.Random(0); bs = []
        for _ in range(2000):
            s = [rng.choice(ids) for _ in ids]
            bs.append(sum(uc[i][3] for i in s) / sum(pc[i][3] for i in s))
        bs.sort(); lo, hi = bs[50], bs[1949]
        ratio = u["kl"] / p["kl"]
        print(f"{K} | {p['kl']:.4f} | {u['kl']:.4f} | {ratio:.3f} | {r14:.3f} (n={len(sub)}) | {med:.3f} | [{lo:.2f},{hi:.2f}] | {p['top1']:.4f}/{u['top1']:.4f} | {p['reads_per_window_kept']:.2f}/{u['reads_per_window_kept']:.2f}/{p['reads_per_window_full']:.2f} | {wins}/{len(ids)} | {u['tail_recovered_frac']:.3f} | {u['kept_picks_per_tok']:.2f}")
        doms = {d: (p["dom"][d], u["dom"][d], u["dom"][d] / p["dom"][d]) for d in p["dom"]}
        print("   dom:", "  ".join(f"{d}:{a:.3f}->{b:.3f}({c:.2f})" for d, (a, b, c) in doms.items()))
        summary[n][K] = dict(plain=p["kl"], union=u["kl"], ratio=ratio, ratio14=r14, med=med, ci=[lo, hi], top1=[p["top1"], u["top1"]],
                             reads=[p["reads_per_window_kept"], u["reads_per_window_kept"], p["reads_per_window_full"]], wins=[wins, len(ids)],
                             tail=u["tail_recovered_frac"], dom=doms)
json.dump(summary, open("summary.json", "w"), indent=1)
