#!/usr/bin/env python3
"""Generate docs/frontier.svg: HeteroFlash speed vs quality on GLM-5.3-Flash.

Standard library only. Numbers are the measured points quoted in README.md.
Every marker carries an SVG <title> (hover text when the SVG is opened directly).
Text sizes are estimated, and the script asserts that no label overlaps another
label, a marker or the frontier line, and that everything stays inside the canvas.
Run from anywhere:  python3 tools/figures/frontier.py
"""
import os
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "frontier.svg")

W, H = 920, 600
XL, XR, X0, X1 = 80, 880, 22.0, 42.0          # x pixels and tok/s range
YB, YT, K1 = 475, 115, 0.12                   # main panel: KL 0 at YB, KL K1 at YT
Y_REF = 80                                    # 2-bit reference line, above an axis break
BLUE, BLUE_HALO, RED, GREEN, GREY = "#0b3d91", "#cfe0ff", "#c0392b", "#1e7b3a", "#b8b8b8"


def px(v):
    return XL + (v - X0) / (X1 - X0) * (XR - XL)


def py(k):
    return YB - k / K1 * (YB - YT)


# (tok/s, KL, hover text) for each kept change, in order
FRONTIER = [
    (24.7, 0.027, "Split by memory + low-latency link: hot experts computed on the Spark GPU, the rest on the "
                  "workstation CPU next to its own RAM. 24.7 tok/s, KL 0.027."),
    (26.4, 0.038, "Top-7: skip each token's weakest expert pick. 26.4 tok/s, KL 0.038."),
    (27.3, 0.038, "Balance rule: decides per step how many experts the GPU takes vs the CPU. 27.3 tok/s, KL 0.038."),
    (29.6, 0.052, "4-bit (NVFP4) attention weights. 29.6 tok/s, KL 0.052."),
    (34.3, 0.057, "HeteroFlash (top-6): 6 of 8 experts per token, plus window reuse and the mean stand-in. "
                  "34.3 tok/s, KL 0.057. Measured before the GPU kernel work."),
    (36.2, 0.065, "HeteroFlash (top-5): 5 of 8 experts per token, plus window reuse and the mean stand-in. "
                  "36.2 tok/s, KL 0.065. Measured before the GPU kernel work."),
    (38.1, 0.085, "Top-4 before the GPU kernel work: 38.1 tok/s, KL 0.085."),
    (40.3, 0.0853, "HeteroFlash (top-4), current best: 4 of 8 experts per token plus the GPU kernel work. "
                   "40.3 tok/s, KL 0.085."),
]
REJECTED = [
    (23.0, 0.079, "Rejected: weak picks one layer late. Good quality (KL 0.079) but only about 23 tok/s, because "
                  "the workstation has no spare bandwidth."),
    (30.0, 0.053, "Rejected: 2-bit fallback tier for skipped picks. Better quality (KL 0.053) but slower "
                  "(about 30 tok/s): the Spark GPU has no spare time."),
    (37.7, 0.078, "Rejected: truncated-expert fallback. KL 0.078 at 37.7 tok/s, a quality gain below the bar we set."),
]
TWO_BIT = (27.5, 0.239, "Standard 2-bit build: every routed expert in 2 bits (IQ2_XXS), all on the Spark. "
                        "27-28 tok/s, KL 0.239.")

o = []
texts = []      # (x0, y0, x1, y1, label) estimated boxes
marks = []      # (cx, cy, r)
segs = []       # frontier line segments


def text(x, y, s, size=12, fill="#222", anchor="start", bold=False, record=True):
    w = (0.60 if bold else 0.56) * size * len(s)
    x0 = {"start": x, "middle": x - w / 2, "end": x - w}[anchor]
    if record:
        texts.append((x0, y - 0.8 * size, x0 + w, y + 0.25 * size, s))
    wt = ' font-weight="bold"' if bold else ""
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{wt}>{escape(s)}</text>')


def point(cx, cy, r, tip, body):
    marks.append((cx, cy, r))
    o.append(f'<g><title>{escape(tip)}</title>'
             f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{max(r, 12)}" fill="#fff" fill-opacity="0"/>{body}</g>')


o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="Helvetica,Arial,sans-serif">')
o.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
text(XL, 28, "GLM-5.3-Flash on 1 DGX Spark + a workstation CPU: speed vs quality", 19, "#111", bold=True)
text(XL, 48, "Decode tok/s for one user vs KL to the original model (Q8 copy of the BF16 weights). "
             "Lower-right is better.", 12.5, "#555")

# inside-the-bar shading and grid
o.append(f'<rect x="{XL}" y="{py(0.1):.1f}" width="{XR - XL}" height="{YB - py(0.1):.1f}" fill="#f2faf4"/>')
for v in range(22, 43, 2):
    o.append(f'<line x1="{px(v):.1f}" y1="{YB}" x2="{px(v):.1f}" y2="{YT}" stroke="#ececec"/>')
    text(px(v), YB + 18, str(v), 12, "#444", "middle", record=False)
for i in range(0, 7):
    k = i * 0.02
    o.append(f'<line x1="{XL}" y1="{py(k):.1f}" x2="{XR}" y2="{py(k):.1f}" stroke="#ececec"/>')
    text(XL - 8, py(k) + 4, f"{k:g}", 12, "#444", "end", record=False)
text(XL - 8, Y_REF + 4, "0.239", 12, RED, "end", record=False)

# axes with a break between the main panel and the 2-bit reference line
o.append(f'<line x1="{XL}" y1="{YB}" x2="{XR}" y2="{YB}" stroke="#333"/>')
o.append(f'<line x1="{XL}" y1="{YB}" x2="{XL}" y2="{YT - 8}" stroke="#333"/>')
o.append(f'<line x1="{XL}" y1="{Y_REF - 8}" x2="{XL}" y2="{Y_REF + 14}" stroke="#333"/>')
for yy in (YT - 14, Y_REF + 20):
    o.append(f'<line x1="{XL - 7}" y1="{yy + 4}" x2="{XL + 7}" y2="{yy - 4}" stroke="#333" stroke-width="1.5"/>')

# reference lines
o.append(f'<line x1="{XL}" y1="{py(0.1):.1f}" x2="{XR}" y2="{py(0.1):.1f}" stroke="{GREEN}" stroke-width="1.5" stroke-dasharray="6 4"/>')
text(XL + 8, py(0.1) - 6, "quality bar: KL 0.1", 12, GREEN)
o.append(f'<line x1="{XL}" y1="{Y_REF}" x2="{XR}" y2="{Y_REF}" stroke="{RED}" stroke-opacity="0.75" stroke-width="2" stroke-dasharray="7 5"/>')
text(XL + 8 + 230, Y_REF - 8, "standard 2-bit build: KL 0.239 (27.5 tok/s), 2.8x the drift of HeteroFlash top-4",
     12, RED)
x2b = px(TWO_BIT[0])
point(x2b, Y_REF, 5, TWO_BIT[2], f'<circle cx="{x2b:.1f}" cy="{Y_REF}" r="5" fill="{RED}"/>')

# rejected ideas: small and light
for v, k, tip in REJECTED:
    point(px(v), py(k), 4, tip, f'<circle cx="{px(v):.1f}" cy="{py(k):.1f}" r="4" fill="none" stroke="{GREY}" stroke-width="1.5"/>')

# frontier line and points (the focus)
pts = [(px(v), py(k)) for v, k, _ in FRONTIER]
segs = list(zip(pts, pts[1:]))
o.append('<polyline fill="none" stroke="%s" stroke-width="4" stroke-linejoin="round" points="%s"/>'
         % (BLUE, " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)))
for i, ((x, y), (_, _, tip)) in enumerate(zip(pts, FRONTIER)):
    if i == len(pts) - 1:
        body = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="{BLUE_HALO}"/>'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{BLUE}" stroke="#fff" stroke-width="2"/>')
        point(x, y, 16, tip, body)
    else:
        point(x, y, 6.5, tip, f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{BLUE}" stroke="#fff" stroke-width="1.5"/>')

# labels
(p1, p2, p3, p4, p5, p6, p7, p8) = pts
text(p1[0] - 12, p1[1] + 22, "split by memory + link tuning (24.7, 0.027)", 12, BLUE)
text(p2[0] + 6, p2[1] + 21, "top-7, then + balance rule (KL 0.038)", 12, BLUE)
text(p4[0] - 12, p4[1] - 14, "+ 4-bit attention (29.6, 0.052)", 12, BLUE, "end")
text(p5[0], p5[1] + 25, "HeteroFlash (top-6)", 12, BLUE, "middle", bold=True)
text(p5[0], p5[1] + 39, "34.3 tok/s, KL 0.057", 11, BLUE, "middle")
text(p6[0] + 14, p6[1] + 10, "HeteroFlash (top-5)", 12, BLUE, bold=True)
text(p6[0] + 14, p6[1] + 24, "36.2 tok/s, KL 0.065", 11, BLUE)
text(p7[0] - 10, p7[1] - 10, "38.1 (before kernel work)", 11, BLUE, "end")
text(XR - 20, YT + 25, "HeteroFlash (top-4)", 15, BLUE, "end", bold=True)
text(XR - 20, YT + 43, "40.3 tok/s, KL 0.085", 13, BLUE, "end", bold=True)
o.append(f'<line x1="{p8[0]:.1f}" y1="{YT + 50}" x2="{p8[0]:.1f}" y2="{p8[1] - 18:.1f}" stroke="{BLUE}" stroke-width="1"/>')

# legend
LX, LY = 150, 188
o.append(f'<rect x="{LX}" y="{LY}" width="300" height="72" rx="6" fill="#fff" stroke="#ccc"/>')
o.append(f'<line x1="{LX + 10}" y1="{LY + 16}" x2="{LX + 34}" y2="{LY + 16}" stroke="{BLUE}" stroke-width="4"/>'
         f'<circle cx="{LX + 22}" cy="{LY + 16}" r="5" fill="{BLUE}"/>')
text(LX + 44, LY + 20, "HeteroFlash: each kept change", 12, BLUE, bold=True)
o.append(f'<line x1="{LX + 10}" y1="{LY + 36}" x2="{LX + 34}" y2="{LY + 36}" stroke="{RED}" stroke-width="2" stroke-dasharray="7 5"/>')
text(LX + 44, LY + 40, "standard 2-bit build (reference)", 12, RED)
o.append(f'<circle cx="{LX + 22}" cy="{LY + 56}" r="4" fill="none" stroke="{GREY}" stroke-width="1.5"/>')
text(LX + 44, LY + 60, "tried and rejected (hover for details)", 12, "#888")
texts.append((LX, LY, LX + 300, LY + 72, "legend box"))

# axis titles and footnotes
text((XL + XR) / 2, YB + 42, "decode speed (tok/s, one user, higher is better)", 14, "#222", "middle")
o.append(f'<text transform="translate(26 {(YB + YT) / 2:.0f}) rotate(-90)" font-size="14" fill="#222" '
         f'text-anchor="middle">KL divergence (lower is better)</text>')
text(XL, YB + 72, "Grey: weak picks one layer late, 2-bit fallback tier, truncated experts. "
                  "Whole-partial deferral (32.5 tok/s, KL 0.44) is off the chart.", 11, "#666")
text(XL, YB + 87, "All points except 40.3 were measured before the GPU kernel work. "
                  "The y axis is broken above KL 0.12.", 11, "#666")
text(XL, YB + 102, "Not shown: first deployment 6.5 tok/s; two-GPU server 151 tok/s at KL 0.042.", 11, "#666")
o.append("</svg>")


# ---- layout checks ----
def overlap(a, b, pad=1.0):
    return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad


def seg_hits_box(s, b):
    (xa, ya), (xb, yb) = s
    for t in [i / 50 for i in range(51)]:
        x, y = xa + (xb - xa) * t, ya + (yb - ya) * t
        if b[0] - 1 <= x <= b[2] + 1 and b[1] - 1 <= y <= b[3] + 1:
            return True
    return False


bad = []
for i, a in enumerate(texts):
    if a[0] < 2 or a[2] > W - 2 or a[1] < 2 or a[3] > H - 2:
        bad.append(("outside canvas", a[4]))
    for b in texts[i + 1:]:
        if a[4] == "legend box" or b[4] == "legend box":
            inside = lambda t, L: t[0] >= L[0] and t[2] <= L[2] and t[1] >= L[1] and t[3] <= L[3]
            if inside(a, b) or inside(b, a):
                continue
        if overlap(a, b):
            bad.append(("text/text", a[4], b[4]))
    for cx, cy, r in marks:
        if overlap(a, (cx - r, cy - r, cx + r, cy + r)):
            bad.append(("text/marker", a[4], (round(cx), round(cy))))
    for s in segs:
        if seg_hits_box(s, a):
            bad.append(("text/line", a[4]))
assert not bad, bad

svg = "\n".join(o) + "\n"
ET.fromstring(svg)
with open(OUT, "w") as f:
    f.write(svg)
print("wrote", os.path.normpath(OUT))
