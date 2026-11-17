#!/usr/bin/env python3
"""Numpy-free variant of runtime/gguf_expert_index.py (same 48-byte records) for hosts without numpy.
  gguf_idx_pure.py GGUF OUT.idx"""
import struct, sys
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
# ggml type -> (block elements, block bytes)
BT = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
      16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 30: (1, 2)}
f = open(sys.argv[1], "rb")
def rd(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def rstr(): n = rd("Q"); return f.read(n).decode("utf-8", "replace")
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        at = rd("I"); n = rd("Q")
        if at == 8: return [rstr() for _ in range(n)]
        if at == 9: return [rval(9) for _ in range(n)]
        f.seek(SZ[at] * n, 1); return None
    return struct.unpack("<" + {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t], f.read(SZ[t]))[0]
assert f.read(4) == b"GGUF"; ver = rd("I"); nt = rd("Q"); nkv = rd("Q")
align = 32
for _ in range(nkv):
    k = rstr(); t = rd("I"); v = rval(t)
    if k == "general.alignment": align = v
infos = []
for _ in range(nt):
    name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; ty = rd("I"); off = rd("Q"); infos.append((name, dims, ty, off))
base = (f.tell() + align - 1) // align * align
recs = []
for name, dims, ty, off in infos:
    for pi, p in enumerate(("gate", "up", "down")):
        if name.endswith(f"ffn_{p}_exps.weight"):
            L = int(name.split(".")[1]); cols, rows, ne = dims[0], dims[1], dims[2]
            be, bb = BT[ty]; rb = cols // be * bb
            recs.append(struct.pack("<6i3Q", L, pi, ty, rows, cols, ne, rb, base + off, rb * rows * ne))
open(sys.argv[2], "wb").write(b"".join(recs)); print(len(recs), "expert tensors indexed")
