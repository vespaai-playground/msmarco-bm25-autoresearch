# msmarco-bm25-autoresearch

Companion code for the blog post
[*Re-autoresearching MSMARCO BM25, on Vespa*](https://blog.vespa.ai/re-autoresearching-msmarco-bm25-on-vespa/).
The post has the motivation, the results, and the discussion; this repo is just
the code to reproduce them.

It reproduces three ways of improving MSMARCO passage-ranking MRR@10 over an
Anserini-tuned BM25 baseline, all on [Vespa](https://vespa.ai):

1. A **"manual" coding agent rank-feature sweep** (`scripts/sweep_paired.py`) — paired
   evaluation over 10 rotating train splits.
2. A **Custom LLM loop** (`agent/run_agent.py`) — gpt-5.5 editing Vespa rank
   profiles autonomously (inspired by [softwaredoug's autoresearch loop](https://softwaredoug.com/blog/2026/05/17/autoresearching-a-better-msmarco-bm25)).
   It accepts a change only if it clears the **same paired-rotation check** the
   manual sweep uses — so the two differ only in human-steered vs autonomous.
3. A **generalization test** (`scripts/eval_full_msmarco.py`) — does the
   minimarco lift survive on the full 8.84M-doc corpus?

The minimarco subset is deterministic (`collection.sample(n=650_000,
random_state=42)`), but don't expect the *exact* numbers reported here. Metrics
depend on your Vespa version (tokenizer/stemmer, BM25 implementation), and the
agent loop is LLM-driven (gpt-5.5), so its discovered config and scores vary
from run to run. Expect the same ballpark and the same overall story, not
identical figures.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) for Python deps
- A container runtime: `docker` or `podman`
- ~2 GB disk for the minimarco subset; ~20 GB if you also do the full
  8.84M-doc corpus
- For the LLM agent only: `OPENAI_API_KEY` in your environment

```bash
uv sync
```

## 1. Start Vespa and deploy the app

> **Heads-up:** every step here (`vespa deploy`, `vespa feed`) targets
> whatever your vespa CLI is already configured for, so point it somewhere safe first.

```bash
# docker also works; substitute 'docker' for 'podman'
podman run --detach --name vespa \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa

# wait for the config server, then deploy
until curl -sf http://localhost:19071/state/v1/health >/dev/null; do sleep 3; done
vespa config set target local
(cd vespa-app && vespa deploy --wait 120)
```

The schema (`vespa-app/schemas/passage.sd`) defines two rank profiles:
`bm25` (the Anserini-tuned baseline) and `lexical` (BM25 plus `nativeProximity`
and `fieldMatch.earliness` as query-time weights, both defaulting to 0 — so
`lexical` with no inputs is just BM25).

## 2. Get the data and build the minimarco subset

```bash
mkdir -p data && cd data
curl -L -O https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -xzf collectionandqueries.tar.gz   # collection.tsv, qrels.dev.small.tsv, queries.dev.small.tsv, ...
cd ..

# sample 650k passages (random_state=42), build the Vespa feed + filtered qrels
uv run python scripts/build_minimarco.py
```

This writes `data/minimarco_feed.jsonl` (650k docs), `data/minimarco_corpus.tsv`,
and the filtered `data/minimarco_queries.tsv` / `data/minimarco_qrels.tsv`
(543 queries / 545 qrels that survive the sample).

## 3. Feed minimarco and reproduce the baseline + manual sweep

```bash
# feed the 650k subset (each doc tagged subset=1)
(cd data && vespa feed --target local minimarco_feed.jsonl)

# paired-rotation sweep over rank-feature weights (the core result)
uv run python scripts/sweep_paired.py
```

`sweep_paired.py` evaluates each candidate weight on 10 rotating train
splits (seeds 1234-1243, 20% train each) and reports the mean paired delta
vs BM25 with a per-config stderr. This is what keeps the manual sweep honest —
a +0.005 that only shows up on one rotation gets rejected.

> Note: every doc in `minimarco_feed.jsonl` is tagged `subset=1`. The
> optional `build_full_feed.py` (step 4) is what adds the rest of the corpus
> with `subset=0`, so the `... and subset=1` filter the eval scripts use
> works whether or not you've fed the full corpus yet.

## 4. (Optional) Generalization test on full MSMARCO

This feeds all 8.84M passages (the 650k subset keeps `subset=1`, the rest get
`subset=0`), so you can query either view from one index. Needs ~20 GB and
~30-60 min to feed.

```bash
uv run python scripts/build_full_feed.py            # writes data/msmarco_full_feed.jsonl
(cd data && vespa feed --target local msmarco_full_feed.jsonl)

uv run python scripts/eval_full_msmarco.py          # BM25 + tuned, on minimarco AND full
```

## 5. (Optional) Run the LLM agent

An autonomous agent editing Vespa rank profiles. Each round it tunes one signal
against a single rotation, then `commit_patch` accepts it only if it clears the
**same 10-rotation paired-robustness check** the manual sweep uses (so a weight
that only fits the round's rotation is rejected). Needs `OPENAI_API_KEY`. ~$6 of
gpt-5.5 spend, ~30 min. See `agent/README.md` for the loop details.

```bash
# from the repo root; reads data/minimarco_*.tsv, writes agent/run_state/
AGENT_ROUNDS=8 AGENT_REASONING=xhigh uv run python agent/run_agent.py

# then evaluate the committed best on minimarco (in-sample) + full MSMARCO
# (held-out). The run leaves the committed schema deployed, so this just works:
uv run python agent/harness/final_eval.py best
```

Follow along live with `tail -f agent/run_state/transcript.txt`.

> Heads-up: the agent overwrites `vespa-app/schemas/passage.sd` (with
> `agent/baseline_schema.sd` at start, then with whatever it builds). After
> a run, that file is the agent's leftover, not the canonical schema with
> the `lexical` profile. If you go back to step 3 (manual sweep) or step 4
> (full-corpus eval) afterwards, restore the canonical schema first:
> `git checkout vespa-app/schemas/passage.sd && (cd vespa-app && vespa deploy --wait 60)`.

## Layout

```
vespa-app/                 clean Vespa app: bm25 + lexical rank profiles
scripts/
  build_minimarco.py       sample 650k, build feed + filtered qrels
  build_full_feed.py       full 8.84M feed with subset flag
  sweep_paired.py          manual paired-rotation weight sweep  ← core
  eval_full_msmarco.py     generalization eval (minimarco + full)
agent/
  run_agent.py             gpt-5.5 orchestration loop (OpenAI Responses API)
  harness/final_eval.py    minimarco (in-sample) + full-MSMARCO eval (experimenter only)
  README.md                how the agent loop works + how to run it
```
