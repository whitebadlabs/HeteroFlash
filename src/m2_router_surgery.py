#!/usr/bin/env python3
"""M2: training-free router surgery sweep against the live trace server (stdlib only).

The server runs src/trace_hook (v2), which reads a control file and applies
routing surgery (k_active, k_ref) inside the real model. This driver:
  diag      ask the hook to dump one forward's module call sequence (find the routed-output class)
  baseline  greedy top-8 generations for the held-out prompts; stores prompt/generated token ids
  grid      for each config, teacher-force prompt+baseline-generation and record prompt_logprobs
  report    per config: exact NLL delta on the actual tokens, top-1 agreement, top-K forward KL bound

Teacher forcing uses /v1/completions with token-id prompts, max_tokens=1 and
prompt_logprobs=K (server started with --max-logprobs >= K). vLLM always
returns the logprob of the actual token at each position, so NLL deltas are
exact. KL is restricted to the reference distribution's top-K tokens; a token
missing from the candidate's top-K gets the candidate's K-th logprob, which
under-estimates its divergence. The bound is reported as such.
"""
import argparse
import gzip
import json
import math
import os
import time
import urllib.request

BASE = "http://127.0.0.1:8001"
MODEL = "GLM-5.3-Flash"


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def set_control(ctl_path, **kw):
    cur = {"capture": False, "k_active": 8, "k_ref": "active", "routed_class": None, "diag": False, "fixed_quant": None}
    cur.update(kw)
    tmp = ctl_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f)
    os.replace(tmp, ctl_path)
    # the hook re-reads at most once per second, at the start of a forward pass: poke twice
    for _ in range(2):
        time.sleep(1.2)
        post("/v1/completions", {"model": MODEL, "prompt": "ok", "max_tokens": 1, "temperature": 0})
    return cur


def holdout(prompts_file, domains=None, limit=0):
    rows = [json.loads(l) for l in open(prompts_file)]
    rows = [r for r in rows if r["split"] == "holdout" and (not domains or r["domain"] in domains)]
    return rows[:limit] if limit else rows


def cmd_baseline(a):
    out = []
    for i, p in enumerate(holdout(a.prompts, a.domains, a.limit)):
        tok = post("/tokenize", {"model": MODEL, "messages": p["messages"], "add_generation_prompt": True})
        ids = tok["tokens"]
        r = post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": a.max_tokens, "temperature": 0,
                                     "logprobs": 0, "return_tokens_as_token_ids": True})
        gen = [int(t.split(":")[1]) for t in r["choices"][0]["logprobs"]["tokens"]]
        out.append({"prompt_id": p["prompt_id"], "domain": p["domain"], "prompt_ids": ids, "gen_ids": gen})
        print(f"baseline {i + 1}: {p['domain']} {len(ids)}+{len(gen)}", flush=True)
    with gzip.open(a.out, "wt") as f:
        json.dump(out, f)


def cmd_grid(a):
    base = json.load(gzip.open(a.baseline, "rt"))
    configs = json.loads(a.configs)
    done = set()
    if os.path.exists(a.out):
        for line in gzip.open(a.out, "rt"):
            r = json.loads(line)
            done.add((r["config"], r["prompt_id"]))
    with gzip.open(a.out, "at") as f:
        for cfg in configs:
            name = cfg.pop("name")
            set_control(a.control, **cfg)
            for b in base:
                if (name, b["prompt_id"]) in done:
                    continue
                if len(b["prompt_ids"]) + len(b["gen_ids"]) > a.max_len:
                    continue      # long-context prompts are measured separately (prompt_logprobs memory)
                r = post("/v1/completions", {"model": MODEL, "prompt": b["prompt_ids"] + b["gen_ids"], "max_tokens": 1,
                                             "temperature": 0, "prompt_logprobs": a.k})
                pl = r["choices"][0]["prompt_logprobs"][len(b["prompt_ids"]):]
                rec = {"config": name, "settings": cfg, "prompt_id": b["prompt_id"], "domain": b["domain"],
                       "actual": [], "top": []}
                for pos, tid in zip(pl, b["gen_ids"]):
                    d = {int(k): v["logprob"] for k, v in pos.items()}
                    rec["actual"].append(d.get(tid))
                    rec["top"].append(sorted(d.items(), key=lambda kv: -kv[1])[:a.k])
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print(f"config {name} done", flush=True)
    set_control(a.control)  # back to plain top-8, capture off


def cmd_report(a):
    by = {}
    for line in gzip.open(a.grid, "rt"):
        r = json.loads(line)
        by.setdefault(r["config"], {})[r["prompt_id"]] = r
    ref = by[a.reference]
    rows = {}
    for name, recs in by.items():
        nll_d, kl, top1, n = [], [], 0, 0
        per_dom = {}
        for pid, r in recs.items():
            q0 = ref.get(pid)
            if not q0:
                continue
            for i, (lp, lp0) in enumerate(zip(r["actual"], q0["actual"])):
                if lp is None or lp0 is None:
                    continue
                nll_d.append(lp0 - lp)
                P = dict((int(t), v) for t, v in q0["top"][i])
                Q = dict((int(t), v) for t, v in r["top"][i])
                qmin = min(Q.values())
                k = sum(math.exp(lp_) * (lp_ - Q.get(t, qmin)) for t, lp_ in P.items())
                kl.append(max(k, 0.0))
                top1 += max(P, key=P.get) == max(Q, key=Q.get)
                n += 1
                per_dom.setdefault(r["domain"], []).append(lp0 - lp)
        if not n:
            continue
        kl.sort()
        rows[name] = {"settings": next(iter(recs.values()))["settings"], "tokens": n,
                      "mean_nll_increase": round(sum(nll_d) / n, 5), "top1_agreement": round(top1 / n, 4),
                      "kl_topk_lower_bound_mean": round(sum(kl) / n, 5), "kl_topk_lower_bound_p95": round(kl[int(0.95 * n)], 4),
                      "nll_increase_by_domain": {d: round(sum(v) / len(v), 5) for d, v in per_dom.items()}}
    json.dump(rows, open(a.out, "w"), indent=1)
    print(json.dumps(rows, indent=1))


def cmd_diag(a):
    set_control(a.control, diag=True)
    post("/v1/completions", {"model": MODEL, "prompt": "Diagnostic forward.", "max_tokens": 2, "temperature": 0})
    time.sleep(2)
    set_control(a.control)
    print("look for diag_calls_<pid>.json in the trace raw dir")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diag", "baseline", "grid", "report"])
    ap.add_argument("--control", default="/srv/cache/heteroflash/traces/control.json")
    ap.add_argument("--prompts", default="/srv/cache/heteroflash/prompts/m1_calib_v1.jsonl")
    ap.add_argument("--domains", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--baseline")
    ap.add_argument("--grid")
    ap.add_argument("--configs", default=json.dumps([
        {"name": "k8_ref", "k_active": 8}, {"name": "k8_repeat", "k_active": 8},
        {"name": "k6_active", "k_active": 6}, {"name": "k4_active", "k_active": 4},
        {"name": "k3_active", "k_active": 3}, {"name": "k2_active", "k_active": 2},
        {"name": "fixed_fp8", "fixed_quant": {"fmt": "fp8"}}, {"name": "fixed_nvfp4", "fixed_quant": {"fmt": "nvfp4"}},
        {"name": "k4_fixed_nvfp4", "k_active": 4, "fixed_quant": {"fmt": "nvfp4"}}]))
    ap.add_argument("--reference", default="k8_ref")
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=4096, help="skip teacher-forcing sequences longer than this")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"diag": cmd_diag, "baseline": cmd_baseline, "grid": cmd_grid, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
