#!/usr/bin/env python3
"""Synthetic placement/quantization research prototype.

This is NOT an actual GLM kernel, profiler, or SLA predictor. Real access traces,
calibrated kernel costs, quantization error, and measured network latency are
mandatory before decisions based on its output can be trusted.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

TIERS = ("spark_gpu", "spark_arm", "remote_cpu")
BITS = {"gpu_int4": 4.0, "gpu_int2p5": 2.5,
        "arm_int4": 4.0, "arm_int8": 8.0,
        "remote_int4": 4.0, "remote_int8": 8.0}
LEGAL = {"spark_gpu": ("gpu_int4", "gpu_int2p5"),
         "spark_arm": ("arm_int4", "arm_int8"),
         "remote_cpu": ("remote_int4", "remote_int8")}

@dataclass(frozen=True)
class Expert:
    layer: int
    expert_id: int
    probability: float  # probability of expert among selected picks, sum=1/layer
    sensitivity: float  # SYNTHETIC rank proxy, NOT KL or calibrated error
    parameters: int

@dataclass
class Placement:
    layer: int
    expert_id: int
    tier: str
    quant: str


def make_synthetic_trace(layers: int, experts_per_layer: int,
                         parameters_per_expert: int, zipf: float, seed: int):
    if layers <= 0 or experts_per_layer <= 0 or parameters_per_expert <= 0:
        raise ValueError("Positive dimensions required")
    rng = random.Random(seed)
    generated = []
    for l in range(layers):
        ranks = list(range(1, experts_per_layer + 1))
        rng.shuffle(ranks)
        norm = sum(r ** (-zipf) for r in range(1, experts_per_layer + 1))
        for e, rank in enumerate(ranks):
            p = rank ** (-zipf) / norm
            # Only a deterministic synthetic proxy. No model saliency implied.
            sensitivity = (0.4 + 1.2 * rng.random()) * (0.5 + p * experts_per_layer * 0.05)
            generated.append(Expert(l, e, p, sensitivity, parameters_per_expert))
    return generated


def bytes_for(expert: Expert, quant: str) -> float:
    return expert.parameters * BITS[quant] / 8.0


def allocations(experts: list[Expert], config: dict) -> list[Placement]:
    """Greedy hotness-driven capacity-feasible starting plan, not an optimal MILP.

    Every expert has one home tier. Higher-sensitivity experts may receive int8 on
    a CPU tier if the memory budget allows. A real optimizer should use KLD and
    measured per-expert kernel cost and solve the actual joint placement problem.
    """
    spark_budget = float(config["spark_total_gb"] - config["spark_reserved_gb"]) * 1e9
    remote_budget = float(config["remote_available_gb"]) * 1e9
    if spark_budget < 0 or remote_budget <= 0:
        raise ValueError("Invalid memory capacity/reservations")
    # Start all weights in remote DDR5 at int4.
    by_id = {(x.layer, x.expert_id): Placement(x.layer, x.expert_id, "remote_cpu", "remote_int4") for x in experts}
    remote_used = sum(bytes_for(x, "remote_int4") for x in experts)
    if remote_used > remote_budget:
        raise ValueError(f"Remote int4 seed cannot fit: {remote_used/1e9:.2f} GB > {remote_budget/1e9:.2f} GB")
    spark_used = 0.0
    # The "hot" score is usage per resident byte, with a modest sensitivity term.
    order = sorted(experts, key=lambda x: x.probability * (1 + 0.1*x.sensitivity) / bytes_for(x, "gpu_int4"), reverse=True)
    gpu_target = spark_budget * float(config.get("gpu_share_of_expert_budget", 0.85))
    for e in order:
        size = bytes_for(e, "gpu_int4")
        if spark_used + size <= gpu_target:
            by_id[(e.layer, e.expert_id)] = Placement(e.layer, e.expert_id, "spark_gpu", "gpu_int4")
            spark_used += size
            remote_used -= bytes_for(e, "remote_int4")
    # Local CPU tier competes with the GPU for the SAME Spark physical RAM.
    for e in order:
        key = (e.layer, e.expert_id)
        if by_id[key].tier != "remote_cpu":
            continue
        size = bytes_for(e, "arm_int4")
        if spark_used + size <= spark_budget:
            by_id[key] = Placement(e.layer, e.expert_id, "spark_arm", "arm_int4")
            spark_used += size
            remote_used -= bytes_for(e, "remote_int4")
    # Spend spare *remote* RAM upgrading high-sensitivity remote experts to INT8.
    for e in sorted(experts, key=lambda x: x.probability*x.sensitivity, reverse=True):
        key = (e.layer, e.expert_id)
        if by_id[key].tier != "remote_cpu":
            continue
        additional = bytes_for(e, "remote_int8") - bytes_for(e, "remote_int4")
        if remote_used + additional <= remote_budget:
            by_id[key].quant = "remote_int8"
            remote_used += additional
    return [by_id[(e.layer, e.expert_id)] for e in experts]


def validate(experts: list[Expert], placements: list[Placement], config: dict) -> dict:
    original_ids = {(e.layer, e.expert_id) for e in experts}
    assigned_ids = [(p.layer, p.expert_id) for p in placements]
    if len(set(assigned_ids)) != len(assigned_ids) or set(assigned_ids) != original_ids:
        raise ValueError("Every expert must be assigned exactly once")
    exmap = {(e.layer, e.expert_id): e for e in experts}
    usage = {tier: 0.0 for tier in TIERS}
    counts = {tier: 0 for tier in TIERS}
    for p in placements:
        if p.tier not in LEGAL or p.quant not in LEGAL[p.tier]:
            raise ValueError(f"Invalid tier/quant: {p}")
        usage[p.tier] += bytes_for(exmap[p.layer, p.expert_id], p.quant)
        counts[p.tier] += 1
    spark_cap = float(config["spark_total_gb"] - config["spark_reserved_gb"]) * 1e9
    remote_cap = float(config["remote_available_gb"]) * 1e9
    if usage["spark_gpu"] + usage["spark_arm"] > spark_cap + 1.0:
        raise ValueError("Shared Spark memory budget exceeded")
    if usage["remote_cpu"] > remote_cap + 1.0:
        raise ValueError("Remote DDR budget exceeded")
    return {"usage_GB_decimal": {tier: round(usage[tier]/1e9, 3) for tier in TIERS},
            "counts": counts,
            "spark_shared_expert_used_GB": round((usage["spark_gpu"]+usage["spark_arm"])/1e9,3)}


def tier_work_proxy(experts: list[Expert], placements: list[Placement], config: dict) -> dict:
    """Illustrative *proxy*, not wall time prediction: ignores measured kernels.

    Selected-expert traffic is estimated independently from synthetic probabilities.
    This proxy properly counts Spark GPU+Arm against the same memory interface.
    """
    per_layer = {}
    pairs = { (p.layer, p.expert_id): p for p in placements }
    for e in experts:
        t = pairs[e.layer, e.expert_id]
        d = per_layer.setdefault(e.layer, {"gpu":0.,"arm":0.,"remote":0.,"quality_proxy":0.,"remote_picks":0.})
        n_picks = config["selected_top_k"] * e.probability
        b = bytes_for(e, t.quant) * n_picks
        key = {"spark_gpu":"gpu","spark_arm":"arm","remote_cpu":"remote"}[t.tier]
        d[key] += b
        # Synthetic heuristic that cannot be used as model-quality evidence.
        penalty = {"gpu_int4":0.02,"gpu_int2p5":0.10,"arm_int4":0.02,"arm_int8":0.005,
                   "remote_int4":0.02,"remote_int8":0.005}[t.quant]
        d["quality_proxy"] += n_picks * e.sensitivity * penalty
        if t.tier == "remote_cpu":
            d["remote_picks"] += n_picks
    total_proxy = 0.
    for l, d in per_layer.items():
        # Relative units based on assumed throughput; only rank order can be explored.
        shared_ms = 1000*(d["gpu"] + d["arm"])/(config["assumed_spark_bw_GBps"]*1e9)
        arm_ms = 1000*d["arm"]/(config["assumed_arm_bw_GBps"]*1e9)
        remote_ms = 1000*d["remote"]/(config["assumed_remote_cpu_bw_GBps"]*1e9)
        if d["remote_picks"] > 0:
            remote_ms += config["assumed_remote_roundtrip_ms"]
        total_proxy += max(shared_ms, arm_ms, remote_ms)
    return {"tier_overlap_proxy_ms_per_target_pass":round(total_proxy,3),
            "synthetic_quality_cost_not_KL":round(sum(x["quality_proxy"] for x in per_layer.values()),4),
            "warning":"Synthetic, uncalibrated cost proxy. Do NOT use as measured throughput or publication evidence."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    experts = make_synthetic_trace(config["layers"], config["experts_per_layer"],
                                   config["parameters_per_expert"], config["zipf_exponent"], config["seed"])
    placements = allocations(experts, config)
    memory = validate(experts, placements, config)
    proxy = tier_work_proxy(experts, placements, config)
    data = {"status":"SYNTHETIC_ONLY_NOT_MEASURED", "config":config, "memory":memory,
            "proxy":proxy, "placements":[asdict(x) for x in placements]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({k:v for k,v in data.items() if k!="placements"},indent=2))

if __name__ == "__main__":
    main()
