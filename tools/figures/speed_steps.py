#!/usr/bin/env python3
"""Generate docs/speed_steps.svg: cumulative steps from the first deployment to 42.7 tok/s.

Standard library only. Each row adds one change on top of all rows above it; values are the
single-user decode speeds in README.md ("From 6.5 to 42.7 tok/s"). Rows were measured on
different days and slightly different workloads, so the increments are approximate.
Run from anywhere:  python3 tools/figures/speed_steps.py
"""
import os
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "speed_steps.svg")

STEPS = [
    ("First deployment (off-the-shelf llama.cpp, 1 GbE)", 6.5),
    ("+ 10G link + MTP speculative drafting", 17.0),
    ("+ our vLLM-based runtime, all experts on CPU", 16.7),
    ("+ hot experts on the Spark GPU, rest on the CPU", 22.6),
    ("+ low-latency link tuning", 24.7),
    ("+ skip the weakest expert pick (top-7)", 26.4),
    ("+ balance rule (GPU/CPU share per step)", 27.3),
    ("+ 4-bit attention", 29.6),
    ("+ 4-bit shared experts and dense layers", 31.3),
    ("+ top-5, window reuse, mean stand-in", 36.2),
    ("+ top-4", 38.1),
    ("+ GPU kernel work", 40.3),
    ("+ late weak picks (top-3 on time + 2 late)", 42.7),
]
FLAT = 0.5          # changes smaller than this are drawn neutral and labelled "about the same"

W, H = 980, 40 + len(STEPS) * 42 + 100
L, R, XMAX = 370, 70, 44


def px(v):
    return L + v / XMAX * (W - L - R)


o = []
a = o.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  f'font-family="Helvetica,Arial,sans-serif"><rect width="{W}" height="{H}" fill="#fff"/>')
a('<text x="20" y="28" font-size="18" font-weight="bold" fill="#111">From 6.5 to 42.7 tok/s: '
  'each step adds one change to all the steps above it</text>')
a('<text x="20" y="48" font-size="12" fill="#555">GLM-5.3-Flash, one user, decode tok/s. Steps were measured on '
  'different days and slightly different workloads, so increments are approximate.</text>')
y, prev = 72, 0.0
for i, (lab, v) in enumerate(STEPS):
    assert 0.56 * 12.5 * len(lab) < L - 20, lab          # label fits left of the bars
    base = 0 if i == 0 else prev
    x0, x1 = sorted([px(base), px(v)])
    flat = i > 0 and abs(v - prev) < FLAT
    col = "#9aa3ad" if i == 0 or flat else "#1f6feb"
    if i == len(STEPS) - 1:
        col = "#1e7b3a"
    a(f'<text x="{L - 10}" y="{y + 17}" font-size="12.5" fill="#222" text-anchor="end">{escape(lab)}</text>')
    if i > 0:
        a(f'<line x1="{px(prev):.1f}" y1="{y - 14}" x2="{px(prev):.1f}" y2="{y + 4}" stroke="#bbb" stroke-dasharray="3 3"/>')
    a(f'<rect x="{x0:.1f}" y="{y + 4}" width="{max(x1 - x0, 2):.1f}" height="20" fill="{col}" rx="2"/>')
    xt = max(x1, px(prev)) + 6
    a(f'<text x="{xt:.1f}" y="{y + 19}" font-size="12.5" fill="#111" font-weight="bold">{v:.1f}</text>')
    if i > 0:
        inc = "about the same" if flat else f"{v - prev:+.1f}"
        a(f'<text x="{xt + 38:.1f}" y="{y + 19}" font-size="12" fill="{"#777" if flat else "#1f6feb"}">{inc}</text>')
        assert xt + 38 + 0.56 * 12 * len(inc) < W - 4
    prev = v
    y += 42
for v in range(0, 45, 10):
    a(f'<line x1="{px(v):.1f}" y1="{y}" x2="{px(v):.1f}" y2="{y + 5}" stroke="#333"/>'
      f'<text x="{px(v):.1f}" y="{y + 19}" font-size="11" fill="#444" text-anchor="middle">{v}</text>')
a(f'<line x1="{px(0):.1f}" y1="{y}" x2="{px(XMAX):.1f}" y2="{y}" stroke="#333"/>')
a(f'<text x="{(px(0) + px(XMAX)) / 2:.1f}" y="{y + 38}" font-size="12.5" fill="#222" text-anchor="middle">'
  f'decode tok/s (one user). Final: 42.7 tok/s at KL 0.0996 vs a Q8 copy of the original BF16 weights</text>')
a("</svg>")
svg = "\n".join(o) + "\n"
ET.fromstring(svg)
with open(OUT, "w") as f:
    f.write(svg)
print("wrote", os.path.normpath(OUT))
