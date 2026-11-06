#!/usr/bin/env python3
"""Router trace schema, writer, reader and statistics for GLM-5.3-Flash MoE layers.

Engine-agnostic: any runtime (HF transformers hooks, vLLM, ExLlamaV3, a custom
engine) calls `TraceWriter.record(...)` with the routing decision of one MoE
layer for a contiguous run of tokens. An optional PyTorch adapter
(`attach_hf_router_hooks`) hooks the upstream `Glm5NextTextTopkRouter`, whose
forward returns `(router_logits, topk_weights, topk_indices)` (transformers
commit fd3cf0140de03f7619ac95b5cfed05b711ff9e22, 2026-11-04).

File format: gzip JSON Lines. First line is a `meta` record; every following
line is a `route` record covering `n` consecutive tokens of one request, phase
and layer. Arrays are row-major `n x k`.

Nothing in this module fabricates routing data. The statistics are only as
real as the trace that is fed in; `meta.source` records where it came from.
"""
from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_VERSION = 1
PHASES = ("prefill", "decode")


@dataclass
class TraceMeta:
    model_id: str
    model_revision: str            # checkpoint commit / file hash; required for reproducibility
    engine: str                    # e.g. "transformers@fd3cf01", "vllm@<sha>", "exllamav3@<sha>"
    source: str                    # "real" or "synthetic" (synthetic traces must say so)
    num_experts: int
    top_k: int
    moe_layers: list[int]          # absolute decoder-layer indices that have routers
    routed_scaling_factor: float
    norm_topk_prob: bool
    topn_recorded: int = 0         # extra candidates recorded per token for reference-k experiments
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d.update(type="meta", schema_version=SCHEMA_VERSION)
        return d


def _check_route(meta: TraceMeta, rec: dict) -> None:
    n, k = rec["n"], meta.top_k
    if rec["phase"] not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if rec["layer"] not in meta.moe_layers:
        raise ValueError(f"layer {rec['layer']} is not a MoE layer in meta")
    if len(rec["expert_ids"]) != n or any(len(r) != k for r in rec["expert_ids"]):
        raise ValueError("expert_ids must be n x top_k")
    if len(rec["gate_weights"]) != n or any(len(r) != k for r in rec["gate_weights"]):
        raise ValueError("gate_weights must be n x top_k")
    for row in rec["expert_ids"]:
        if len(set(row)) != k or min(row) < 0 or max(row) >= meta.num_experts:
            raise ValueError("expert ids must be distinct and in range")
    if meta.topn_recorded:
        for key in ("topn_ids", "topn_scores"):
            if len(rec.get(key, [])) != n or any(len(r) != meta.topn_recorded for r in rec[key]):
                raise ValueError(f"{key} must be n x topn_recorded")


class TraceWriter:
    """Append-only gzip JSONL writer. Use as a context manager."""

    def __init__(self, path: str | Path, meta: TraceMeta):
        if meta.source not in ("real", "synthetic"):
            raise ValueError("meta.source must be 'real' or 'synthetic'")
        self.meta = meta
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = gzip.open(self.path, "wt", encoding="utf-8")
        self._fh.write(json.dumps(meta.to_json(), separators=(",", ":")) + "\n")
        self.records = 0

    def record(self, *, request_id: str, domain: str, phase: str, layer: int, token_start: int,
               expert_ids: list[list[int]], gate_weights: list[list[float]],
               topn_ids: list[list[int]] | None = None, topn_scores: list[list[float]] | None = None) -> None:
        rec = {"type": "route", "request_id": request_id, "domain": domain, "phase": phase,
               "layer": int(layer), "token_start": int(token_start), "n": len(expert_ids),
               "expert_ids": [[int(x) for x in r] for r in expert_ids],
               "gate_weights": [[round(float(x), 6) for x in r] for r in gate_weights]}
        if self.meta.topn_recorded:
            rec["topn_ids"] = [[int(x) for x in r] for r in (topn_ids or [])]
            rec["topn_scores"] = [[round(float(x), 6) for x in r] for r in (topn_scores or [])]
        _check_route(self.meta, rec)
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.records += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_trace(path: str | Path) -> tuple[TraceMeta, Iterator[dict]]:
    """Return (meta, iterator of route records). Validates every record."""
    fh = gzip.open(Path(path), "rt", encoding="utf-8")
    first = json.loads(fh.readline())
    if first.get("type") != "meta" or first.get("schema_version") != SCHEMA_VERSION:
        fh.close()
        raise ValueError("not a router trace (missing meta / wrong schema_version)")
    first = {k: v for k, v in first.items() if k not in ("type", "schema_version")}
    meta = TraceMeta(**first)

    def it():
        with fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("type") != "route":
                    raise ValueError("unexpected record type")
                _check_route(meta, rec)
                yield rec
    return meta, it()


# ---------------------------------------------------------------- statistics

def selection_counts(meta: TraceMeta, records: Iterable[dict], phase: str | None = None,
                     domain: str | None = None) -> dict[int, list[int]]:
    """Per MoE layer, how often each expert was selected."""
    counts = {l: [0] * meta.num_experts for l in meta.moe_layers}
    for r in records:
        if (phase and r["phase"] != phase) or (domain and r["domain"] != domain):
            continue
        c = counts[r["layer"]]
        for row in r["expert_ids"]:
            for e in row:
                c[e] += 1
    return counts


def coverage(counts: list[int], resident_fraction: float) -> float:
    """Fraction of selections served by the most-selected `resident_fraction` of experts."""
    if not 0 <= resident_fraction <= 1:
        raise ValueError("resident_fraction must be in [0, 1]")
    total = sum(counts)
    if total == 0:
        return 0.0
    m = round(resident_fraction * len(counts))
    return sum(sorted(counts, reverse=True)[:m]) / total


def hot_set(counts: list[int], n: int) -> list[int]:
    """Expert ids of the n most-selected experts (ties broken by id)."""
    return [e for e, _ in sorted(enumerate(counts), key=lambda t: (-t[1], t[0]))[:n]]


def all_hot_probability(meta: TraceMeta, records: Iterable[dict], hot: dict[int, set[int]]) -> dict[int, float]:
    """Per layer, fraction of tokens whose k experts are ALL in the hot set
    (that layer needs no remote round trip for that token)."""
    hit = defaultdict(int)
    tot = defaultdict(int)
    for r in records:
        h = hot[r["layer"]]
        for row in r["expert_ids"]:
            tot[r["layer"]] += 1
            hit[r["layer"]] += all(e in h for e in row)
    return {l: hit[l] / tot[l] for l in tot}


def window_union(meta: TraceMeta, records: Iterable[dict], window: int, phase: str = "decode") -> dict[int, float]:
    """Mean number of distinct experts per layer over consecutive `window`-token
    groups of the same request (proxy for the expert union a speculative verify
    step of `window` tokens must load). Windows never span requests."""
    if window < 1:
        raise ValueError("window must be >= 1")
    rows = defaultdict(list)                      # (request, layer) -> [(token_index, ids)]
    for r in records:
        if r["phase"] != phase:
            continue
        for i, row in enumerate(r["expert_ids"]):
            rows[(r["request_id"], r["layer"])].append((r["token_start"] + i, row))
    sums, cnt = defaultdict(float), defaultdict(int)
    for (req, layer), seq in rows.items():
        seq.sort()
        for s in range(0, len(seq) - window + 1, window):
            u = set()
            for _, ids in seq[s:s + window]:
                u.update(ids)
            sums[layer] += len(u)
            cnt[layer] += 1
    return {l: sums[l] / cnt[l] for l in cnt}


def empirical_probabilities(counts: dict[int, list[int]]) -> dict[int, list[float]]:
    """Per layer, P(expert | a selection slot) — sums to 1 per layer (the
    `Expert.probability` convention used by placement_sim)."""
    out = {}
    for l, c in counts.items():
        t = sum(c)
        if t == 0:
            raise ValueError(f"layer {l} has no selections in the trace")
        out[l] = [x / t for x in c]
    return out


# ------------------------------------------------------- optional HF adapter

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")


def attach_hf_router_hooks(model, writer: TraceWriter, context: dict, topn: int = 0):
    """Register forward hooks on every `*TopkRouter` module of an HF model.

    `context` is a mutable dict the caller updates before each forward:
    {"request_id", "domain", "phase", "token_start"}; batch size must be 1.
    Returns the list of hook handles (call .remove() on each to detach).
    Requires torch; imported lazily so the rest of this module stays stdlib-only.
    """
    import torch  # noqa: F401  (ensures torch is present)

    handles = []
    for name, mod in model.named_modules():
        if not type(mod).__name__.endswith("TopkRouter"):
            continue
        m = _LAYER_RE.search(name)
        if not m:
            raise ValueError(f"cannot infer layer index from module name {name!r}")
        layer = int(m.group(1))

        def hook(module, inputs, output, layer=layer):
            logits, weights, indices = output
            rec_kwargs = {}
            if writer.meta.topn_recorded:
                scores = logits.float().sigmoid()
                choice = scores + module.e_score_correction_bias.float()
                top = choice.topk(writer.meta.topn_recorded, dim=-1)
                rec_kwargs = {"topn_ids": top.indices.tolist(),
                              "topn_scores": scores.gather(1, top.indices).tolist()}
            writer.record(request_id=context["request_id"], domain=context["domain"],
                          phase=context["phase"], layer=layer, token_start=context["token_start"],
                          expert_ids=indices.tolist(), gate_weights=weights.float().tolist(), **rec_kwargs)
        handles.append(mod.register_forward_hook(hook))
    if not handles:
        raise ValueError("no *TopkRouter modules found")
    return handles


def summarize(path: str | Path, fractions=(0.25, 0.5, 0.62, 0.75), windows=(1, 4, 6)) -> dict:
    meta, recs = read_trace(path)
    recs = list(recs)
    counts = selection_counts(meta, recs)
    cov = {f: sum(coverage(c, f) for c in counts.values()) / len(counts) for f in fractions}
    unions = {w: window_union(meta, recs, w) for w in windows}
    return {"source": meta.source, "model_revision": meta.model_revision, "engine": meta.engine,
            "records": len(recs),
            "tokens_by_phase": {p: sum(r["n"] for r in recs if r["phase"] == p and r["layer"] == meta.moe_layers[0])
                                for p in PHASES},
            "mean_coverage_by_resident_fraction": {str(k): round(v, 4) for k, v in cov.items()},
            "mean_window_union": {str(w): (round(sum(u.values()) / len(u), 3) if u else None)
                                  for w, u in unions.items()}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Summarize a router trace")
    ap.add_argument("trace")
    print(json.dumps(summarize(ap.parse_args().trace), indent=2))
