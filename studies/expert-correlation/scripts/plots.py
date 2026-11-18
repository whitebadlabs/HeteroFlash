#!/usr/bin/env python3
"""Three plots for the results page; each has one panel per model.

Usage: plots.py OUTDIR NAME=RESULTS.json [NAME=RESULTS.json ...]   (results from analyze.py)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = sys.argv[1]
MODELS = [(kv.split("=", 1)[0], json.load(open(kv.split("=", 1)[1]))) for kv in sys.argv[2:]]
os.makedirs(OUT, exist_ok=True)
short = {"code": "code", "instruction_following": "instr", "knowledge_qa": "know-qa", "long_context": "long-ctx",
         "math": "math", "multilingual": "multiling", "writing_chat": "writing", "chat": "chat"}
plt.rcParams.update({"font.size": 9, "figure.dpi": 140})
n = len(MODELS)

# 1. layer N -> N+1 predictability per layer
fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 3.4), squeeze=False)
for ax, (name, R) in zip(axs[0], MODELS):
    D, K, E = R["domains"], R["K"], R["E"]
    nl = R["next_layer_recall"]
    same = np.array([nl[f"{d}->{d}"]["transition_per_layer"] for d in D]).mean(0)
    cross = np.array([nl[f"{a}->{b}"]["transition_per_layer"] for a in D for b in D if a != b]).mean(0)
    pop = np.array([nl[f"{d}->{d}"]["popularity_per_layer"] for d in D]).mean(0)
    ml = R.get("moe_layers") or list(range(R["layers"]))
    x = np.array(ml[1:])
    ax.plot(x, same, label="predicted from layer N, same task type", color="#1f5fa8")
    ax.plot(x, cross, label="predicted from layer N, other task type", color="#1f5fa8", ls="--")
    ax.plot(x, pop, label=f"baseline: {K} most-used experts of layer N+1", color="#999999")
    ax.axhline(K / E, color="#cccccc", lw=0.8, ls=":")
    ax.set_xlabel("predicted MoE layer N+1 (model layer index)")
    ax.set_ylabel(f"share of the {K} picks predicted (recall@{K})")
    ax.set_ylim(0, 1)
    ax.set_title(f"{name}: {E} experts, top-{K}")
axs[0][0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.suptitle("How well a token's experts in layer N predict its experts in layer N+1 (holdout prompts)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_next_layer_recall.png")); plt.close(fig)

# 2. task dependence: co-selection partner lift, partners chosen on A (calib), measured on B (holdout)
fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 4.0), squeeze=False)
for ax, (name, R) in zip(axs[0], MODELS):
    D = R["domains"]
    lab = [short.get(d, d) for d in D]
    cp = R["coselect_partner"]
    Mx = np.array([[cp[f"{a}->{b}"]["lift"] for b in D] for a in D])
    im = ax.imshow(Mx, cmap="viridis")
    for i in range(len(D)):
        for j in range(len(D)):
            ax.text(j, i, f"{Mx[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if Mx[i, j] < Mx.mean() else "black")
    ax.set_xticks(range(len(D)), lab, rotation=45, ha="right"); ax.set_yticks(range(len(D)), lab)
    ax.set_xlabel("measured on (holdout prompts)"); ax.set_ylabel("partners chosen on (calib prompts)")
    ax.set_title(name)
    fig.colorbar(im, ax=ax, shrink=0.8, label="lift = P(partner | expert) / P(partner)")
fig.suptitle("Co-selection strength of each expert's top-7 partners, and how it carries across task types", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_coselection_task_matrix.png")); plt.close(fig)


# 3. simulation: cross-GPU sends and load imbalance vs simulated GPUs
def avg(runs, D, G, fit, key, metric, per_domain_fit=False):
    vals = []
    for b in D:
        f = b if per_domain_fit else fit
        v = runs.get(f"G{G}|{f}|{key}|{b}")
        if v and v.get(metric) is not None:
            vals.append(v[metric])
    return float(np.mean(vals)) if vals else np.nan


series = [("random equal-size", "mix", "random_0", False, "#bbbbbb"),
          ("vLLM DefaultEplbPolicy (mixed load)", "mix", "default", False, "#444444"),
          ("correlated, fit on mixed tasks", "mix", "correlated_s1.0", False, "#1f5fa8"),
          ("correlated, fit per task type", None, "correlated_s1.0", True, "#d0702c")]
fig, axs = plt.subplots(n, 2, figsize=(8.0, 3.0 * n), squeeze=False)
for row, (name, R) in zip(axs, MODELS):
    runs, D = R["sim"]["runs"], R["domains"]
    Gs = sorted({int(k.split("|")[0][1:]) for k in runs})
    w = 0.2
    for i, (lbl, fit, key, per, col) in enumerate(series):
        s = [avg(runs, D, G, fit, key, "sends", per) for G in Gs]
        im_ = [avg(runs, D, G, fit, key, "imbalance_batch", per) for G in Gs]
        row[0].bar(np.arange(len(Gs)) + (i - 1.5) * w, s, w, label=lbl, color=col)
        row[1].bar(np.arange(len(Gs)) + (i - 1.5) * w, im_, w, color=col)
    for ax in row:
        ax.set_xticks(range(len(Gs)), [f"{G} GPUs" for G in Gs])
    row[0].set_ylabel(f"{name}\ncross-GPU sends per token per layer")
    row[1].set_ylabel("load imbalance (max / mean GPU)\nper 2048-token batch")
    row[1].set_ylim(1.0, None)
h, l_ = axs[0][0].get_legend_handles_labels()
fig.legend(h, l_, frameon=False, fontsize=7.5, loc="lower center", ncol=2)
fig.suptitle("Simulated replay of holdout prompts (mean over task types)", fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(OUT, "fig3_sim_sends_imbalance.png")); plt.close(fig)
print("ok")
