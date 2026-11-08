# Runtime-v1: per-layer heterogeneous expert exchange, measured (2026-11-08)

Bet 6 of `docs/BREAKTHROUGH_PLAN_2026-11-07.md`: the Spark sends, per MoE layer, the tokens and gate weights for the experts that live in workstation RAM; the workstation CPU computes them and returns the partial sum.

All numbers here are **[M]**, measured on the Spark (client, Arm CPU only, no GPU) and the workstation (int4 expert server, 32 Zen5 cores) over the 10 GbE direct link, with the workstation otherwise idle. Weights are real GLM-5.3-Flash experts (int4 g128 RTN, `experts_int4_rtn`), decoder layers 12/20/30/38/44 loaded (20.4 GB); each 42-layer step cycles through them.

## What was built (runtime/)

- **`expert_server.c` (extended).**
  - **Busy-poll transport per connection** (request flag bit0): TCP_NODELAY, TCP_QUICKACK re-armed after each read, SO_BUSY_POLL 50 us and non-blocking spin recv. The server falls back to `poll()` after about 2 ms of spinning, so an idle connection does not burn a core.
  - **Heartbeat mode** (flag bit1): a pacer thread sends a 0x00 byte every 20 us while a request is being computed and stops before the reply is written; the client skips leading 0x00 bytes.
  - **Binding and loading:** `HF_BIND=<ip>` binds to the direct-link address only; `serve-weights <port> <dir> <layer list>` loads a subset of layers.
  - **Batching:** multi-token requests batch identical experts across tokens (one weight pass per expert), as before.
- **`layer_exchange_bench.c` (new).** A Spark-side bench that sends 42 sequential per-layer requests per step.
  - T tokens per request each select 8 experts from a window union of U, using the union sizes measured on the M1 traces. The CPU owns a fraction f of the union.
  - Requests are pre-generated outside the timed path.
  - `check` mode compares heartbeat+busy-poll replies against plain-socket replies bit for bit.
- **`link_pingpong.c` (extended).** Server compute delay, heartbeat, and several receive modes, used for the diagnosis below.

## Correctness [M]

| check | result |
|---|---|
| `expert_server selftest` (int4 batched vs scalar reference; int8 vs int4 path) | 4.9e-7 and 1.1e-6 relative L2 |
| real weights, layer 20, 4 experts x 3 tokens: server vs the same int4 weights in numpy | 1.57% (int4 RTN vs BF16: 21.1%) |
| real weights, layer 44, 2 experts x 3 tokens | 1.78% (int4 vs BF16: 21.8%) |
| busy-poll + heartbeat replies vs plain-socket replies, 30 requests (T = 1, 4, 8) | bit-identical |

## Finding: a ~300 us idle-gap penalty on this link, and the fix [M]

The busy-poll ping-pong measures 50 us for 8 KB. With real expert compute between request and reply, however, the transport overhead (rtt - server compute) was 315-430 us. `link_pingpong` with a server-side delay reproduces it without any expert code (8 KB each way, p50):

| server delay | busy-poll both sides | + heartbeat every 50 us | + heartbeat every 20 us |
|---|---|---|---|
| 0 us | 50 us | | |
| 100 us | 500 us | | |
| 300 us | 608 us | 402 us | 352 us |
| 600 us | 907 us | 805 us | 652 us |
| 1500 us | 1808 us | 1656 us | 1552 us |

What the diagnosis showed:

- **Any quiet gap of 100 us or more costs a constant ~300 us.**
  - It appears in both directions (Spark to workstation and the reverse), so it is not tied to one NIC.
  - It appears with blocking recv and with spin recv without busy-poll, which both give a ~310 us round trip even at zero delay.
  - SO_PREFER_BUSY_POLL would need CAP_NET_ADMIN.
- **A single early byte does not help; continuous 20 us heartbeats remove most of the penalty** (+52 us at p50; p99 still shows occasional +300 us).
- The root cause is in the kernel/NIC receive path and would need root tracing to confirm. This was deliberately not done, to stay sudo-less.

## Per-layer exchange with real experts [M]

Server settings: OMP active wait, 32 threads pinned to cores. Values are p50 over 40 steps x 42 layers.

| workload | T | U | f | CPU experts | rtt busy-poll | rtt + heartbeat | server compute | transport, busy-poll | **transport, + heartbeat** | CPU eff. GB/s | step (42 layers), busy-poll | **step, + heartbeat** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 1 | 8 | 0 | 0 | 84 | 85 | 21-24 | 64 | 61 | - | 3.5 ms | 4.6 ms |
| decode | 1 | 8 | 0.5 | 4 | 653 | 408 | 338-343 | 315 | **62** | 165-168 | 27.6 ms | **17.4 ms** |
| decode | 1 | 8 | 0.85 | 7 | 845 | 602 | 530-539 | 316 | **62** | 184-187 | 35.6 ms | **25.4 ms** |
| decode | 1 | 8 | 1.0 | 8 | 907 | 667 | 592-604 | 316 | **61** | 188-191 | 38.1 ms | **28.3 ms** |
| verify | 4 | 19 | 0.5 | 10 | 1118 | 893 | 758-784 | 355 | **105** | 181-187 | 47.3 ms | **37.8 ms** |
| verify | 4 | 19 | 1.0 | 19 | 1713 | 1518 | 1356-1407 | 356 | **109** | 191-198 | 72.0 ms | **63.9 ms** |
| verify | 4 | 23 | 0.85 | 20 | 1828 | 1569 | 1413-1457 | 415 | **109** | 194-200 | 76.7 ms | **66.1 ms** |
| verify | 8 | 26 | 0.5 | 13 | 1414 | 1215 | 994-1031 | 417 | **172** | 179-185 | 59.5 ms | **51.4 ms** |
| verify | 8 | 26 | 0.85 | 22 | 2020 | 1850 | 1598-1662 | 420 | **177** | 187-195 | 84.9 ms | **77.9 ms** |
| verify | 8 | 26 | 1.0 | 26 | 2293 | 2129 | 1865-1944 | 427 | **178** | 189-197 | 96.4 ms | **89.6 ms** |

Notes on the table:

- **OMP wait policy matters.** With passive waits, CPU bandwidth is lower (133-193 GB/s; `bench_*_omppassive.txt`).
- **Heartbeat cost.** The pacer thread spins on an SMT sibling and costs about 1-4% of server compute.
- **Where the T=8 transport goes.** About 105 us of the 172-178 us is bf16 activation serialisation: 64 KB each way at 10 Gb/s.

## Against the plan's modeled assumptions

| assumption [P] | measured [M] | verdict |
|---|---|---|
| CPU effective expert-read bandwidth ~182 GB/s | 165-202 GB/s with OMP active wait (higher for larger unions) | holds |
| decode sync <= ~3 ms per token (about 0.07 ms per layer) | 61-62 us per layer with heartbeat = 2.6 ms per 42 layers; 315 us per layer (13 ms) without it | holds only with the heartbeat fix |
| verify sync overlapped with Spark-side work | 105-178 us per layer (4.4-7.5 ms per step), not yet overlapped | open: needs int8 activations (~halves the T=8 cost) and overlap with GPU attention/shared-expert work |

## Not done here (out of scope or next)

- GPU-side integration and Spark/CPU overlap; this bench measures only the CPU leg.
- int8 activation transport.
- NUMA/SMT tuning of the pacer thread.
- Root-causing the idle-gap penalty, which needs privileged tracing.

## Files

Raw outputs are in this directory:

- `bench_plain_omppassive.txt`, `bench_busy_omppassive.txt`
- `bench_busy_ompactive.txt` (the tail may overlap the start of another job)
- `bench_busy_ompactive_v2.txt`, `bench_busy_ompactive_v3.txt`
- `bench_busyhb_ompactive.txt`
- `check_identity.txt`, `pingpong_recheck.txt`, `pingpong_delay.txt`
