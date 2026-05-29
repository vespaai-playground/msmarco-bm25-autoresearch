# LLM agent loop

An autonomous agent that edits **Vespa rank profiles**, accepting changes with
the **same paired-rotation robustness check** the manual sweep
(`scripts/sweep_paired.py`) uses. So the agent and the manual sweep apply an
identical acceptance test — what differs is only human-steered vs autonomous.
(It's inspired by [softwaredoug's autoresearch loop](https://softwaredoug.com/blog/2026/05/17/autoresearching-a-better-msmarco-bm25),
but uses our paired-rotation gate rather than a literal port of his single
held-out commit check.)

`run_agent.py` drives `gpt-5.5` (OpenAI Responses API). The 10 rotations are the
same ones `sweep_paired.py` uses (seeds `1234..1243`, each a random 20% of the
dev queries):

- **Tune on one rotation, commit against all ten.** Each round the agent tunes
  against a *single* rotation (it rotates per round). `run_train_eval` scores the
  deployed config on **that one rotation** vs the committed best — its only
  tuning view, and a noisy one. So while tuning it sees only 1 of the 10
  rotations.
- **`commit_patch`** is the gate: it re-scores the config across **all 10**
  rotations vs the committed best (each rotation *paired*, so per-subset
  difficulty cancels) and commits only if the mean paired delta is
  `>= eval_margin (0.002)` **and** clearly above the rotation-to-rotation noise
  (`> 2` standard errors — not "all 10 positive"; even the blog's winning config
  is `+0.034` mean but 9/10). Because the agent only saw one rotation while
  tuning, a weight that merely fits that rotation fails here — the commit is a
  real held-out test, not a formality. One 543-query eval (parallelized) feeds
  all 10 rotations.
- **One signal, exactly one commit attempt per round.** Each round the agent
  adds (or retunes) one signal, tunes its weight on the round's rotation, then
  **must make a single `commit_patch` attempt** — the round ends whether it's
  accepted or rejected (the harness nudges it to attempt if it tries to stop
  early). The single-attempt rule stops it from probing the held-out gate
  (commit, read the rejected mean/std, nudge weight, re-commit); the must-attempt
  rule stops it from wasting a round by giving up — every round tests one genuine
  held-out guess.
  `AGENT_ROUNDS` is thus "up to N sequential, each-validated improvements" — a
  greedy forward-selection. Every commit must beat the prior across the
  rotations, so the committed best is **monotonic**: the last commit is the best,
  which is what `agent/harness/state.json` records for `final_eval.py best`.
- Each round is a **fresh agent context** starting from the committed schema.
  Reasoning state does not carry across rounds, so the **cross-round memory is the
  committed schema plus a compact summary of earlier rounds' commit attempts**
  (committed *and* rejected). The committed schema's accumulated first-phase
  expression is its running work log (each accepted round adds/retunes one signal,
  so the expression grows), while the attempts summary lets it build on near-misses
  and avoid re-trying rejected dead ends.

Tools the agent has (defined in `run_agent.py`):

- `read_schema`
- `write_rank_profile(first_phase)` — supply ONLY the candidate's first-phase
  expression; the harness wraps it as a profile inheriting `bm25` (k1/b always
  carry) and auto-declares the `query(...)` weights it references. The agent
  can't touch the document/fields/bm25 baseline.
- `run_train_eval(inputs, extra)` — advisory paired delta vs committed on the
  round's single rotation; commits nothing. `extra` is request parameters (e.g.
  `ranking.matching.*`) — a separate tuning lever.
- `commit_patch(inputs, extra)` — the 10-rotation paired-robustness gate
- `query_one(...)` — inspect a single query's top-10
- `search_vespa_docs(query)` — live search of api.search.vespa.ai
- `fetch_url(url)` — fetch a docs.vespa.ai page (markdown)

The agent's initial context includes a focused **rank-expression reference**
(`agent/vespa_rank_expression.md`, distilled from the
[Vespa skills pack](https://github.com/vespaai-playground/skills)) covering the
one thing it emits under this harness: a valid Vespa first-phase expression —
expression syntax, `query()` inputs, and the math functions available. It names
**no ranking features**, so the agent still has to *discover which* rank features
to use, which it does **live** via `search_vespa_docs` / `fetch_url`. This keeps
the experiment about finding good ranking signal, not about Vespa syntax.

## Run it

From the repo root (needs `OPENAI_API_KEY`, a running+deployed Vespa with the
minimarco subset fed — see the top-level `README.md`):

```bash
AGENT_ROUNDS=8 AGENT_REASONING=xhigh uv run python agent/run_agent.py
tail -f agent/run_state/transcript.txt        # follow along

# evaluate the committed best on minimarco (in-sample) + full MSMARCO (held-out).
# The run leaves the committed schema deployed and on-disk, so this just works:
uv run python agent/harness/final_eval.py best
# If you have since changed/redeployed passage.sd, final_eval.py best refuses to
# run rather than silently scoring the wrong config; restore the snapshot first:
#   cp agent/run_state/best_schema.sd vespa-app/schemas/passage.sd
#   (cd vespa-app && vespa deploy --wait 60)
```

Env knobs: `AGENT_MODEL` (default `gpt-5.5`), `AGENT_ROUNDS` (default 8),
`AGENT_REASONING` (default `high`; we used `xhigh`).

**Cost / safety notes:**

- There is no cost ceiling in the code — only a per-round wall-clock cap
  (`ROUND_DEADLINE_SECONDS = 600`) and a per-round turn cap
  (`MAX_TURNS_PER_ROUND = 18`). OpenAI enforces no hard cap either. Our
  8-round xhigh run cost ~$6, but a stuck round at xhigh can spend faster
  than that — watch the live token counts in the transcript.
- `deploy()` refuses to run unless the vespa CLI target is `local`, so the
  agent can't accidentally push the toy schema to a Cloud/production
  endpoint. (Run `vespa config set target local` first.)
- A run **starts the agent from `baseline_schema.sd` (bm25 only)** — it
  overwrites `vespa-app/schemas/passage.sd` with that bm25-only schema and the
  agent builds its own rank profiles from there. It never sees the repo's
  `lexical` profile or the winning weights, so it has to do the work. After a
  run, `passage.sd` holds the agent's final schema; restore the repo version
  with `git checkout vespa-app/schemas/passage.sd && (cd vespa-app && vespa deploy)`.

## Files

- `run_agent.py` — the orchestration loop and the agent's system prompt.
  Writes `harness/state.json` at end-of-run so `final_eval.py best` works.
- `baseline_schema.sd` — the bm25-only schema the agent starts from.
- `vespa_rank_expression.md` — focused rank-expression reference, prepended to the
  agent's system prompt (expression syntax only; names no ranking features).
- `harness/final_eval.py` — evaluates the committed best on minimarco
  (in-sample) and full MSMARCO (the held-out generalization test). Run by the
  experimenter after the agent finishes.
- `run_state/` — per-run artifacts (`last_accepted_schema.sd` /
  `best_schema.sd`, both the committed schema; `run.jsonl`; `transcript.txt`);
  git-ignored.
- To eval an arbitrary profile/weights by hand (outside the agent loop), use
  `scripts/sweep_paired.py` (paired-rotation sweep) or
  `agent/harness/final_eval.py PROFILE k=v ...`.
