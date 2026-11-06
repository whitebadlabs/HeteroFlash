#!/usr/bin/env python3
"""Paired bootstrap for ACTUAL measured CSV rows, one baseline/candidate row per pair_id.

Does not infer independence from repeated trials of one prompt; pair_id should identify
an independent prompt/block according to the preregistered sampling design.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict

METRICS = ("accuracy", "decode_tps", "cold_ttft_ms", "cached_ttft_ms", "kl_teacher")

def quantile(values, p):
    vals=sorted(values)
    if not vals:
        return float('nan')
    ix=(len(vals)-1)*p
    lo=int(math.floor(ix)); hi=int(math.ceil(ix))
    return vals[lo]*(hi-ix)+vals[hi]*(ix-lo) if lo != hi else vals[lo]

def summarize(rows, baseline, candidate, replicates=2000, seed=17, quality_margin=0.02):
    grouped=defaultdict(dict)
    for row in rows:
        key=(row['pair_id'],row['system'])
        if key[1] not in (baseline,candidate): continue
        if key[1] in grouped[key[0]]:
            raise ValueError(f"duplicate observation for same pair/system: {key}")
        grouped[key[0]][key[1]]=row
    rng=random.Random(seed)
    result={'reference':baseline,'candidate':candidate,'paired_input_count':0,'metrics':{}}
    for metric in METRICS:
        paired=[]
        for _,systems in grouped.items():
            if baseline not in systems or candidate not in systems: continue
            a=systems[baseline].get(metric,''); b=systems[candidate].get(metric,'')
            if a != '' and b != '':
                paired.append((float(a),float(b)))
        if not paired: continue
        n=len(paired)
        result['paired_input_count']=max(result['paired_input_count'],n)
        differences=[b-a for a,b in paired]
        observed=sum(differences)/n
        boot=[]
        for _ in range(replicates):
            boot.append(sum(differences[rng.randrange(n)] for _ in range(n))/n)
        ci=[quantile(boot,0.025),quantile(boot,0.975)]
        lower_onesided=quantile(boot,0.05)
        item={'n_pairs':n,'mean_candidate_minus_reference':observed,
              'paired_bootstrap_95pct_CI':ci, 'lower_95pct_one_sided':lower_onesided}
        if metric == 'accuracy':
            item['exploratory_noninferiority_margin']=quality_margin
            item['passes_exploratory_lower_bound']=lower_onesided > -quality_margin
        if metric in ('cold_ttft_ms','cached_ttft_ms'):
            item['candidate_p95_ms']=quantile([b for _,b in paired],0.95)
            p95boot=[]
            for _ in range(replicates):
                sample=[paired[rng.randrange(n)][1] for _ in range(n)]
                p95boot.append(quantile(sample,0.95))
            item['candidate_p95_upper_95pct_one_sided_ms']=quantile(p95boot,0.95)
        result['metrics'][metric]=item
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('csv',help='CSV of paired ACTUAL results')
    p.add_argument('--baseline',default='T')
    p.add_argument('--candidate',default='S1')
    p.add_argument('--replicates',type=int,default=2000)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--margin',type=float,default=0.02)
    a=p.parse_args()
    with open(a.csv,newline='') as f: rows=list(csv.DictReader(f))
    print(json.dumps(summarize(rows,a.baseline,a.candidate,a.replicates,a.seed,a.margin),indent=2))

if __name__=='__main__': main()
