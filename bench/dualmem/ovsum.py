import sys
rows=[l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
rows=[(t,float(e),float(g)) for t,e,g in rows]
tags=sorted(set(r[0] for r in rows))
iv={t:(min(e for x,e,g in rows if x==t),max(e for x,e,g in rows if x==t)) for t in tags}
for t in tags:
    others=[o for o in tags if o!=t]
    alone=[g for x,e,g in rows if x==t and all(not(iv[o][0]-0.6<=e<=iv[o][1]+0.6) for o in others)]
    tog=[g for x,e,g in rows if x==t and all(iv[o][0]+0.6<=e<=iv[o][1]-0.6 for o in others)]
    f=lambda a: f"n={len(a)} median={sorted(a)[len(a)//2]:.1f} min={min(a):.1f} max={max(a):.1f}" if a else "n=0"
    print(f"{t}: alone {f(alone)} | together {f(tog)}")
