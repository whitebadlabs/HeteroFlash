#!/usr/bin/env python3
"""Paired dNLL between two scored runs from their per-token arrays (HF_PERTOKEN npz, chunk order, same tokens):
B - A per token, mean and SE over tokens, plus each run's own dNLL vs the Q8 proxy.   paired_pertoken.py A.npz B.npz"""
import json, sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
n = min(a["dnll"].size, b["dnll"].size); d = b["dnll"][:n].astype(np.float64) - a["dnll"][:n]
print(json.dumps({"A": sys.argv[1].split("/")[-1], "B": sys.argv[2].split("/")[-1], "tokens": int(n),
                  "paired_dnll_B_minus_A": round(float(d.mean()), 5), "paired_se": round(float(d.std(ddof=1) / np.sqrt(n)), 5),
                  "A_dnll": round(float(a["dnll"][:n].mean()), 5), "B_dnll": round(float(b["dnll"][:n].mean()), 5),
                  "A_kld": round(float(a["kld"][:n].mean()), 4), "B_kld": round(float(b["kld"][:n].mean()), 4),
                  "A_same": round(float(a["same"][:n].mean()), 4), "B_same": round(float(b["same"][:n].mean()), 4)}))
