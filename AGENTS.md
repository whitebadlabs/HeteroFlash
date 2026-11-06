# AGENTS.md: rules for any coding agent working on HeteroFlash

Read in order before changing anything: this file, `docs/CONVERSATION_CONTEXT.md`, `CURRENT_STATUS.md`, `ROADMAP.md`, `docs/SYSTEM_DESIGN.md`, `docs/EXPERIMENT_PROTOCOL.md`.

## Non-negotiable constraints
- Deployed inference = ONE DGX Spark (GB10 GPU + its Arm CPU, one shared 128 GB pool) + CPU/RAM (~230 GB) of the separate workstation. The workstation's two RTX PRO 6000 Max-Q GPUs never take part in deployed inference.
- The Max-Q GPUs may be used temporarily for teacher inference, training, calibration and evaluation, **but only with explicit permission from the owner for each use**. The same goes for expensive downloads or training runs.
- Hard targets (same checkpoint + config for all three): 120-130 committed prose tok/s; 45K-token cold prompt TTFT <= 10.8 s; 45K exact cached prefix + predeclared short suffix (provisionally +32 tokens) TTFT <= 0.2 s; quality noninferior to the strongest feasible teacher.
- The model stays general-purpose. Do not tune or distill on one application's transcripts.
- Never invent results. Every number in docs, configs or the paper is labelled measured / published / assumed / synthetic, with its source.

## Practice
- Keep `CURRENT_STATUS.md` current after each experiment: what was measured, how, and what is still synthetic.
- Pin model revisions, engine commits, driver/CUDA versions and image digests for every run.
- Tests: `python3 -m unittest discover -s tests -v` (stdlib only; torch-dependent paths are optional and imported lazily).
- Commits are authored as the project owner. Raw logs, hostnames, IP addresses and private data stay out of the repo.
