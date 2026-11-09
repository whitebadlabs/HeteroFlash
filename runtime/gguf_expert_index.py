#!/usr/bin/env python3
"""Write the expert-tensor index expert_server_gguf needs: for each MoE layer and projection, the GGML type, rows, cols,
bytes per row and the absolute file offset of the [n_experts, rows, row_bytes] block inside the GGUF.
  gguf_expert_index.py GGUF OUT.idx GGUF_PY
Binary record (little-endian, 48 B): i32 layer, i32 proj (0 gate, 1 up, 2 down), i32 ggml_type, i32 rows, i32 cols,
i32 n_experts, u64 row_bytes, u64 offset, u64 n_bytes."""
import struct, sys
sys.path.insert(0, sys.argv[3])
import gguf
r = gguf.GGUFReader(sys.argv[1])
recs = []
for t in r.tensors:
    for pi, p in enumerate(("gate", "up", "down")):
        if t.name.endswith(f"ffn_{p}_exps.weight"):
            L = int(t.name.split(".")[1])
            cols, rows, ne = (int(v) for v in t.shape.tolist())
            rb = t.n_bytes // (ne * rows)
            recs.append(struct.pack("<6i3Q", L, pi, int(t.tensor_type), rows, cols, ne, rb, int(t.data_offset), int(t.n_bytes)))
open(sys.argv[2], "wb").write(b"".join(recs))
print(len(recs), "expert tensors indexed")
