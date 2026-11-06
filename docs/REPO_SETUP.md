# Repository setup

HeteroFlash is developed in a local git repo (owner-authored commits). It is published in two steps.

## 1. Forgejo (private, now)
1. In Forgejo, create an **empty** repository `HeteroFlash` in the `whitebadlabs` organization: private, no README, no license, no .gitignore. (The automation token cannot create organization repositories, so the owner does this step.)
2. The assistant then pushes `main` from the working repo through the usual bundle -> agentbox path and sets `main` as the default branch.
3. From then on, work lands as PR branches that are fast-forward merged. Commits are authored by the owner only, with no co-author trailers.

## 2. GitHub (public or private, later)
1. On GitHub, create an **empty** repository `whitebadlabs/HeteroFlash` (no README/license).
2. Create a fine-grained token with *Contents: read and write* on that one repository only.
3. In Forgejo -> HeteroFlash -> Settings -> Repository -> Mirror settings, add a **push mirror**:
   - URL `https://github.com/whitebadlabs/HeteroFlash.git`
   - credentials: GitHub username + the token
   - leave "sync when commits are pushed" **off**, so the research gate decides when to sync.
4. Add `HeteroFlash` to the research gate's `REPOS` list and restart the gate. It scans each new commit (gitleaks + private-pattern rules) and syncs the mirror only when the scan is clean.

## Before the first public sync
- Choose a license for the project's own code and docs. Model weights and third-party runtimes keep their own licenses.
- Review `docs/CONVERSATION_CONTEXT.md`. It quotes the planning conversation verbatim; trim anything that shouldn't be public.
- Keep raw traces, logs, hostnames, addresses and private data out of the repo. Results go in as curated summaries.
