"""Autonomous agent that tunes Vespa rank profiles using our paired-rotation check.

Acceptance is the same robustness test sweep_paired.py uses, run by the agent:
  - Every config is scored by MRR@10 on ROTATIONS paired rotations (seeds
    1234.., each a random 20% of the dev queries) against the currently
    committed config. Per-rotation pairing cancels per-subset difficulty.
  - run_train_eval shows that paired delta (advisory; commits nothing).
  - commit_patch commits only if the mean paired delta over the committed best
    is >= eval_margin (0.002) AND clearly above the rotation noise (> SE_MULT
    standard errors). Marginal/noisy gains are rejected. Each commit raises the
    bar, so the committed best is monotonic and the last commit IS the best.
  - Each round is a FRESH agent context that adds/retunes exactly ONE signal on
    top of the committed schema; combinations accumulate across rounds. Reasoning
    state does not carry across rounds; the cross-round memory is the committed
    schema (whose accumulated first-phase expression is the running work log) plus
    a compact summary of earlier rounds' commit attempts (committed AND rejected),
    both shown at the start of each round.
  - final_eval.py reports the committed best on minimarco (in-sample, like the
    manual sweep) and full MSMARCO (the held-out generalization test).

(This is the manual sweep's methodology run autonomously, not a literal port of
Doug's single-val-set commit_patch — the comparison to the manual sweep then
isolates human-steered vs autonomous, with acceptance rigor held identical.)

Tools exposed to the agent:
  - read_schema()
  - write_rank_profile(first_phase)  -> sets the candidate first-phase expression
      (harness wraps it as a profile inheriting bm25, auto-declares query inputs)
  - run_train_eval(inputs={}, extra={})  -> paired delta vs committed (advisory)
  - commit_patch(inputs={}, extra={})    -> paired-rotation-gated commit
  - query_one(query_text, profile, ...)  -> top-10 hits for inspection
  - search_vespa_docs(query) / fetch_url(url)  -> live Vespa docs lookup

Hard budgets (see constants below):
  - Wall-clock per round: ROUND_DEADLINE_SECONDS (10 min)
  - Per-round LLM turn cap: MAX_TURNS_PER_ROUND (18)
"""
import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from openai import OpenAI


AGENT = Path(__file__).resolve().parent          # code/agent
REPO = AGENT.parent                               # code/
APP = REPO / "vespa-app"
DATA = REPO / "data"
SCHEMA = APP / "schemas" / "passage.sd"
RUN_STATE = AGENT / "run_state"
RUN_STATE.mkdir(exist_ok=True)
# The committed schema (updated only on a successful commit_patch) and a copy
# of it for final_eval.py to verify it is scoring the config that was committed.
LAST_GOOD = RUN_STATE / "last_accepted_schema.sd"
BEST_SCHEMA = RUN_STATE / "best_schema.sd"
LOG = RUN_STATE / "run.jsonl"
TRANSCRIPT = RUN_STATE / "transcript.txt"
# Where final_eval.py expects to find the committed config to evaluate against
# minimarco (in-sample) and full MSMARCO (the held-out generalization test).
HARNESS_STATE = AGENT / "harness" / "state.json"

VESPA_URL = "http://localhost:8080"
BASE_SEED = 1234
TRAIN_FRAC = 0.20
ROTATIONS = 10            # paired rotations, seeds BASE_SEED..BASE_SEED+9 (same as sweep_paired.py)
EVAL_MARGIN = 0.002       # nominal floor on the mean paired delta over the committed config
SE_MULT = 2.0             # robustness: the mean paired delta must also exceed this many
                          # standard errors above zero, i.e. be clearly above the
                          # rotation-to-rotation noise (not "all rotations positive" — even
                          # the blog's winning config is +0.034 mean but 9/10 positive).
ROUNDS = int(os.environ.get("AGENT_ROUNDS", "8"))  # fresh-context passes; one signal each
MAX_TURNS_PER_ROUND = 18  # plenty now that fetch_url returns whole docs (research is ~1 fetch)
ROUND_DEADLINE_SECONDS = 60 * 10  # 10 minutes wall-clock per round
REASONING_EFFORT = os.environ.get("AGENT_REASONING", "high")  # low|medium|high|xhigh

MODEL = os.environ.get("AGENT_MODEL", "gpt-5.5")

# The agent writes only a first-phase expression; the harness wraps it as this
# profile (inheriting bm25 so the tuned k1/b carry). All agent evals use it.
CANDIDATE_PROFILE = "candidate"

SYSTEM_PROMPT = """\
You are an information-retrieval engineer improving a Vespa lexical ranking
profile. The corpus is a 650k-passage subset of MSMARCO. You are scored on
MRR@10.

Goal: beat the Anserini-tuned BM25 baseline by as much as possible, in a way
that is robust — it must hold up across many random resamplings of the queries,
not just look good on one.

How acceptance works:
- Each round you tune on ONE rotation: a single random 20% sample of the dev
  queries (it changes each round). run_train_eval scores the deployed config on
  THAT rotation vs the currently committed config — the PAIRED delta, your MRR
  minus the committed config's MRR on the same queries (per-sample difficulty
  cancels). This is your tuning view; it is a single noisy sample, and it
  commits nothing.
- commit_patch is the gate. It re-scores your config across ALL 10 rotations vs
  the committed best and commits only if the mean paired delta is >= the
  eval_margin (0.002) AND clearly above the rotation-to-rotation noise (the mean
  is large relative to its spread across rotations). Because you only saw ONE
  rotation while tuning, a weight that merely fits your rotation will fail here —
  you must find a change that holds across rotations. Each commit raises the bar.
- Each committed improvement carries forward. Reasoning state does NOT carry
  across rounds — your memory is the committed expression plus a summary of
  earlier rounds' attempts, both shown to you at the start of each round.

Directions to explore (you must discover the actual Vespa feature names,
spellings, and matching parameters yourself — this prompt names none of them):
- BM25 scores bag-of-words term overlap; it ignores where in the document terms
  match and how they are arranged relative to each other. Consider what signals
  could capture what it misses.
- Not all matching query terms are equally informative; consider how very common
  terms affect both matching and ranking.
- bm25 is your strongest single signal and has a wide dynamic range — when you
  combine it with others, mind their relative scales so you don't disorder good
  results. Prefer signals that hold up across all the rotations over weights
  that only help on some.
- Ranking is not only the first-phase expression: query-time REQUEST PARAMETERS
  (passed via the `extra` map on run_train_eval/commit_patch) also change
  retrieval and matching — for instance how the matching pipeline handles
  high-document-frequency terms. Treat these as a separate lever worth
  experimenting with; the full set is in the query API reference, which you can
  fetch: https://docs.vespa.ai/en/reference/api/query.html

Process:
- Use search_vespa_docs / fetch_url to find which rank features and request
  parameters Vespa exposes — this prompt does not name them.
- Set your ranking via write_rank_profile: you supply ONLY the candidate's
  first-phase EXPRESSION. The harness wraps it as a profile that inherits bm25
  (so the tuned k1/b always carry) and auto-declares every query(...) weight you
  reference — so you cannot break the document/fields/bm25 baseline. Form:
  `bm25(description) + query(w_foo) * <feature>`; a feature on a much smaller
  scale than bm25 must NOT replace it or you'll mis-order docs. Sweep the
  query() weights at query time via run_train_eval/commit_patch `inputs` (no
  redeploy needed), and try request parameters via `extra`.
- Each round, add or retune exactly ONE signal — not several at once. Start from
  the committed expression (shown to you each round), then write_rank_profile
  with that expression plus a single new term (or a retuned weight). run_train_eval
  (your single rotation) is free and commits nothing — use it to sweep that one
  signal's weight and find its best value. Then make exactly ONE commit_patch
  attempt with your best candidate. You MUST end every round with a commit
  attempt — even if you are not sure it clears the gate. A rejected attempt is
  fine and informative; giving up without attempting wastes the round. The round
  ends on that attempt (accepted or rejected); you cannot probe the gate by
  retrying.
- Don't over-tune: a handful of weights is enough, then commit. Combinations
  build up across rounds — keep the committed terms and add to them. The
  committed expression carries to your next round (along with the attempts
  summary), so build on it rather than starting over.
"""

# Give the agent a focused reference for the one thing it emits: a Vespa
# first-phase rank EXPRESSION (syntax, query() inputs, math functions) so
# expression mistakes don't dominate the result. It names no ranking features, so
# feature discovery stays on the agent. (Distilled from
# github.com/vespaai-playground/skills.)
SYSTEM_PROMPT += (
    "\n\n========== Vespa rank-expression reference ==========\n"
    + (AGENT / "vespa_rank_expression.md").read_text()
)


def load_queries_qrels():
    queries = {}
    with (DATA / "minimarco_queries.tsv").open() as fh:
        for qid, q in csv.reader(fh, delimiter="\t"):
            queries[qid] = q
    qrels = {}
    with (DATA / "minimarco_qrels.tsv").open() as fh:
        for qid, did, grade in csv.reader(fh, delimiter="\t"):
            if int(grade) > 0:
                qrels.setdefault(qid, set()).add(did)
    return queries, qrels


def build_rotations(all_qids):
    """The ROTATIONS paired rotations from sweep_paired.py: for seed
    BASE_SEED+i, shuffle the queries and take the first TRAIN_FRAC as that
    rotation's subset. Returns a list of qid-lists."""
    rotations = []
    for i in range(ROTATIONS):
        shuffled = sorted(all_qids)
        random.Random(BASE_SEED + i).shuffle(shuffled)
        rotations.append(shuffled[: round(len(shuffled) * TRAIN_FRAC)])
    return rotations


def paired_delta(cand_per_q, base_per_q, rotations):
    """Mean/std/positive-count of the per-rotation paired MRR delta
    (candidate - base) over the rotations. Pairing on the same subset cancels
    per-subset difficulty (exactly what sweep_paired.py reports)."""
    deltas = []
    for rot in rotations:
        c = sum(cand_per_q[q] for q in rot) / len(rot)
        b = sum(base_per_q[q] for q in rot) / len(rot)
        deltas.append(c - b)
    mean = statistics.mean(deltas)
    std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    n_pos = sum(1 for d in deltas if d > 0)
    return {"mean": mean, "std": std, "n_positive": n_pos, "per_rotation": deltas}


def gate_pass(d):
    """A paired_delta dict clears the gate iff its mean is at least EVAL_MARGIN
    AND at least SE_MULT standard errors above zero (robust, not noise)."""
    stderr = d["std"] / (ROTATIONS ** 0.5)
    return d["mean"] >= EVAL_MARGIN and (d["mean"] - SE_MULT * stderr) > 0


def assert_local_target():
    """Refuse to deploy anywhere but a local Vespa. `vespa deploy` targets
    whatever `vespa config` points at; without this guard, running with the
    CLI pointed at Vespa Cloud / a production endpoint would overwrite that
    app. This experiment only ever means to touch a local container."""
    p = subprocess.run(["vespa", "config", "get", "target"],
                       capture_output=True, text=True, timeout=30)
    out = (p.stdout or "").strip()
    target = out.split("=")[-1].strip() if "=" in out else out
    if target != "local":
        raise SystemExit(
            f"Refusing to deploy: vespa CLI target is '{target or '(unset)'}', not 'local'.\n"
            f"This script only deploys to a local Vespa container. "
            f"Run `vespa config set target local` first."
        )


def deploy():
    """Return (success: bool, message: str)."""
    assert_local_target()
    proc = subprocess.run(
        ["vespa", "deploy", "--wait", "60"],
        cwd=str(APP), capture_output=True, text=True, timeout=180,
    )
    msg = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode == 0, msg.strip()


# Request params the harness controls; `extra` must not override them, or the
# agent could score a different profile/yql/hits than the one it commits.
_RESERVED_PARAMS = {"yql", "q", "ranking.profile", "hits", "language",
                    "model.locale", "timeout"}


def vespa_query(yql, query_text, profile, inputs, extra, hits=10):
    body = {
        "yql": yql, "q": query_text, "ranking.profile": profile,
        "hits": hits, "language": "en", "model.locale": "en", "timeout": "30s",
    }
    for k, v in inputs.items():
        body[f"input.query({k})"] = v
    for k in extra:
        if k in _RESERVED_PARAMS or k.startswith("input.query("):
            raise ValueError(
                f"extra may not override the harness-controlled request param '{k}' "
                f"(use inputs for query(...) weights; extra is for other request "
                f"parameters like ranking.matching.*).")
    body.update(extra)
    r = requests.post(f"{VESPA_URL}/search/", json=body, timeout=35)
    r.raise_for_status()
    return r.json()


def eval_per_query(qids, queries, qrels, profile, inputs, extra):
    """Reciprocal rank @10 for each qid (0 if no relevant doc in the top 10).
    Queries run concurrently — Vespa handles parallel requests fine, and one
    543-query pass then feeds every rotation's paired delta."""
    yql = "select doc_id from passage where description contains ({language:'en'}text(@q)) and subset=1"

    def rr(qid):
        rel = qrels.get(qid, set())
        resp = vespa_query(yql, queries[qid], profile, inputs, extra, hits=10)
        for rank, hit in enumerate(resp.get("root", {}).get("children", []) or [], start=1):
            if str(hit["fields"]["doc_id"]) in rel:
                return 1.0 / rank
        return 0.0

    qids = list(qids)
    with ThreadPoolExecutor(max_workers=16) as ex:
        rrs = list(ex.map(rr, qids))
    return dict(zip(qids, rrs))


# ----- Tool implementations -----

def tool_read_schema():
    return SCHEMA.read_text()


# The candidate profile, indented to sit inside `schema { ... }`. The agent
# supplies only {expr}; {inputs} is the auto-generated declaration block for the
# query() weights the expression references. Inherits bm25 so k1/b always carry.
_CANDIDATE_PROFILE_TEMPLATE = """\
    rank-profile {name} inherits bm25 {{
{inputs}        first-phase {{
            expression {{
                {expr}
            }}
        }}
    }}
"""


def _query_inputs(expr: str) -> list:
    """Names of the query(...) weights referenced in a rank expression."""
    return sorted(set(re.findall(r"query\(\s*([A-Za-z_]\w*)\s*\)", expr)))


def assemble_schema(first_phase: str) -> str:
    """Render the candidate profile from the template and inject it into the
    fixed skeleton (baseline_schema.sd = document + fields + fieldset + bm25),
    before the schema-closing brace. The agent never writes the skeleton, so it
    can't break the document block or lose the bm25 baseline."""
    names = _query_inputs(first_phase)
    inputs = ""
    if names:
        decls = "".join(f"            query({n}) double: 0.0\n" for n in names)
        inputs = f"        inputs {{\n{decls}        }}\n"
    profile = _CANDIDATE_PROFILE_TEMPLATE.format(
        name=CANDIDATE_PROFILE, inputs=inputs, expr=first_phase.strip())

    skeleton = (AGENT / "baseline_schema.sd").read_text().rstrip()
    head, brace, _ = skeleton.rpartition("}")
    if not brace:
        raise RuntimeError("baseline_schema.sd must end with the schema-closing brace")
    return f"{head.rstrip()}\n\n{profile}}}\n"


def tool_write_rank_profile(first_phase: str):
    """Set the candidate first-phase expression. The harness wraps it (inherits
    bm25, declares the query inputs it references) and redeploys. Returns the
    declared inputs so the agent knows what it can sweep."""
    try:
        SCHEMA.write_text(assemble_schema(first_phase))
    except Exception as e:
        return {"error": str(e)[:600]}
    ok, msg = deploy()
    out = {"deployed": ok, "profile": CANDIDATE_PROFILE,
           "declared_query_inputs": _query_inputs(first_phase), "message": msg[-1200:]}
    if not ok:
        out["warning"] = ("DEPLOY FAILED — your new expression is NOT live. Fix it "
                          "and call write_rank_profile again; do NOT run_train_eval or "
                          "commit_patch yet (they would score the previously committed "
                          "config, not this one).")
    return out


def _paired_vs(inputs, extra, ctx):
    """Eval the deployed candidate once over all queries, return (per_q, delta-vs-
    committed, delta-vs-bm25). delta-* are paired_delta dicts over the rotations."""
    per_q = eval_per_query(ctx["all_qids"], ctx["queries"], ctx["qrels"],
                           CANDIDATE_PROFILE, inputs, extra)
    state = ctx["state"]
    return (per_q,
            paired_delta(per_q, state["committed_per_q"], ctx["rotations"]),
            paired_delta(per_q, ctx["bm25_per_q"], ctx["rotations"]))


def tool_run_train_eval(inputs=None, extra=None, *, ctx):
    """Advisory: score the deployed candidate on THIS ROUND'S single rotation (one
    random 20% sample) vs the committed best. This is the only slice you see
    while tuning; it is noisy. commit_patch checks robustness across all 10
    rotations. Commits nothing."""
    inputs = inputs or {}
    extra = extra or {}
    rot = ctx["round_rotation"]
    try:
        per_q = eval_per_query(rot, ctx["queries"], ctx["qrels"], CANDIDATE_PROFILE, inputs, extra)
    except Exception as e:
        return {"error": str(e)[:1500]}
    comm = ctx["state"]["committed_per_q"]
    cand_mrr = sum(per_q.values()) / len(per_q)
    comm_mrr = sum(comm[q] for q in rot) / len(rot)
    return {
        "rotation_seed": ctx["round_rotation_seed"],
        "n_queries": len(rot),
        "mrr@10": round(cand_mrr, 6),
        "committed_mrr@10": round(comm_mrr, 6),
        "paired_delta_vs_committed_this_rotation": round(cand_mrr - comm_mrr, 6),
        "note": ("single-rotation advisory — noisy. commit_patch checks all "
                 f"{ROTATIONS} rotations; a weight that only helps this rotation "
                 "will be rejected. Nothing committed."),
    }


def tool_commit_patch(inputs=None, extra=None, *, ctx):
    """The acceptance gate. Scores the deployed candidate on the 10 paired
    rotations vs the committed best and commits it iff the mean paired delta >=
    EVAL_MARGIN AND that mean is more than SE_MULT standard errors above zero
    (robust, not noise — see gate_pass). Committed best is unchanged on rejection."""
    inputs = inputs or {}
    extra = extra or {}
    state = ctx["state"]
    try:
        per_q, d_comm, d_bm25 = _paired_vs(inputs, extra, ctx)
    except Exception as e:
        return {"error": str(e)[:1500]}
    accepted = gate_pass(d_comm)
    result = {
        "mean_paired_delta_vs_committed": round(d_comm["mean"], 6),
        "std": round(d_comm["std"], 6),
        "n_rotations_positive": f"{d_comm['n_positive']}/{ROTATIONS}",
        "eval_margin": EVAL_MARGIN,
        "committed": accepted,
    }
    if accepted:
        state["committed_per_q"] = per_q
        state["n_commits"] += 1
        state["accepted"] = {
            "profile": CANDIDATE_PROFILE, "inputs": inputs, "extra": extra,
            "mean_paired_delta_vs_bm25": round(d_bm25["mean"], 6),
            "n_rotations_positive_vs_bm25": f"{d_bm25['n_positive']}/{ROTATIONS}",
        }
        # The committed schema is whatever is deployed right now.
        LAST_GOOD.write_text(SCHEMA.read_text())
        BEST_SCHEMA.write_text(SCHEMA.read_text())
        result["cumulative_mean_paired_delta_vs_bm25"] = round(d_bm25["mean"], 6)
        result["message"] = ("Committed. Mean paired delta over the committed best "
                             "clears eval_margin and is well above the rotation noise; "
                             "this config is now the committed best.")
    else:
        result["message"] = (
            f"Rejected: needs mean paired delta >= {EVAL_MARGIN} AND clearly above "
            f"the rotation noise (> {SE_MULT}x its standard error). Got mean "
            f"{d_comm['mean']:+.4f}, std {d_comm['std']:.4f}, "
            f"{d_comm['n_positive']}/{ROTATIONS} rotations positive — not robust "
            f"enough. The committed best is unchanged; adjust the weight/feature "
            f"and try again.")
    return result


def tool_query_one(query_text, profile, inputs=None, extra=None, *, queries):
    inputs = inputs or {}
    extra = extra or {}
    yql = "select doc_id, description from passage where description contains ({language:'en'}text(@q)) and subset=1"
    try:
        resp = vespa_query(yql, query_text, profile, inputs, extra, hits=10)
    except Exception as e:
        return {"error": str(e)[:1500]}
    out = []
    for rank, hit in enumerate(resp.get("root", {}).get("children", []) or [], start=1):
        out.append({
            "rank": rank,
            "score": hit.get("relevance"),
            "doc_id": hit["fields"].get("doc_id"),
            "snippet": (hit["fields"].get("description") or "")[:300],
        })
    return {"query": query_text, "profile": profile, "top10": out}


def tool_search_vespa_docs(query: str):
    try:
        r = requests.get("https://api.search.vespa.ai/search/",
                         params={"query": query, "hits": 5}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        return {"error": str(e)[:600]}
    out = []
    for c in r.json().get("root", {}).get("children", []) or []:
        f = c.get("fields", {})
        out.append({"path": f.get("path"), "title": f.get("title"),
                    "snippet": (f.get("content") or "")[:500]})
    return out


FETCH_CAP = 150_000  # safety cap on a single page (the rank-features ref is ~53k)


def tool_fetch_url(url: str):
    """Fetch a Vespa docs page and return the whole markdown (one call — no
    paging), so the agent can read a full reference without burning turns."""
    if not (url.startswith("https://docs.vespa.ai/") or url.startswith("https://api.search.vespa.ai/")):
        return {"error": "fetch_url is restricted to docs.vespa.ai and api.search.vespa.ai"}
    # Drop any #anchor (search results include them). Otherwise the .html -> .md
    # markdown trick misses and we fetch the raw multi-MB HTML page, which the
    # FETCH_CAP then truncates to nav/boilerplate.
    url = url.split("#", 1)[0]
    try:
        md_url = url + ".md" if url.endswith(".html") else url
        r = requests.get(md_url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        return {"error": str(e)[:600]}
    text = r.text
    out = {"url": md_url, "length": len(text), "content": text[:FETCH_CAP]}
    if len(text) > FETCH_CAP:
        out["note"] = f"page is {len(text)} chars; showing the first {FETCH_CAP}."
    return out


# ----- OpenAI tool schema -----

TOOLS = [
    {"type": "function",
     "name": "read_schema",
     "description": "Read the current contents of vespa-app/schemas/passage.sd",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "function",
     "name": "write_rank_profile",
     "description": "Set your candidate's first-phase ranking expression and redeploy. The harness wraps it as `rank-profile candidate inherits bm25 { ... }` — so the tuned BM25 (k1/b) is always inherited — and auto-declares every query(...) input you reference (default 0.0) so you can sweep their values via run_train_eval/commit_patch. You write ONLY the expression; the document, fields, fieldset, and bm25 baseline are fixed and not yours to edit. Example expression: 'bm25(description) + query(w_foo) * <someRankFeature>'. Returns the declared query inputs.",
     "parameters": {
         "type": "object",
         "properties": {"first_phase": {"type": "string", "description": "The first-phase rank expression for the candidate profile (may span lines; reference query(name) for tunable weights)."}},
         "required": ["first_phase"]}},
    {"type": "function",
     "name": "run_train_eval",
     "description": "Advisory: score your candidate (current first-phase + the given inputs/extra) on THIS ROUND'S single rotation (one random 20% sample) vs the committed best, returning the paired MRR@10 delta on that rotation. Noisy (one sample); commit_patch checks robustness across all 10 rotations. Commits nothing.",
     "parameters": {
         "type": "object",
         "properties": {
             "inputs": {"type": "object",
                        "description": "Query input weights for the features in your expression. E.g. {'w_foo': 10.0}.",
                        "additionalProperties": {"type": "number"}},
             "extra": {"type": "object",
                       "description": "Request parameters as a flat key->value map (NOT input.query(...)). These are query-API request properties — e.g. matching-pipeline knobs under ranking.matching.* — and are a real tuning lever; see https://docs.vespa.ai/en/reference/api/query.html.",
                       "additionalProperties": {"type": ["number", "string"]}}},
         "required": []}},
    {"type": "function",
     "name": "commit_patch",
     "description": "The acceptance gate. Scores your candidate (current first-phase + the given inputs/extra) on the 10 paired rotations vs the committed best and commits it only if the mean paired MRR@10 delta >= eval_margin (0.002) AND that mean is clearly above the rotation-to-rotation noise (more than ~2 standard errors above zero) — so a gain that only shows up on some rotations is rejected as noise. The committed best is unchanged on rejection. One attempt per round.",
     "parameters": {
         "type": "object",
         "properties": {
             "inputs": {"type": "object", "additionalProperties": {"type": "number"}},
             "extra": {"type": "object", "additionalProperties": {"type": ["number", "string"]}}},
         "required": []}},
    {"type": "function",
     "name": "query_one",
     "description": "Inspect a single query's top-10 hits. Useful for debugging.",
     "parameters": {
         "type": "object",
         "properties": {
             "query_text": {"type": "string"},
             "profile": {"type": "string"},
             "inputs": {"type": "object", "additionalProperties": {"type": "number"}},
             "extra": {"type": "object", "additionalProperties": {"type": ["number", "string"]}}},
         "required": ["query_text", "profile"]}},
    {"type": "function",
     "name": "search_vespa_docs",
     "description": "Search Vespa documentation. Returns top hits with path, title, snippet.",
     "parameters": {
         "type": "object",
         "properties": {"query": {"type": "string"}},
         "required": ["query"]}},
    {"type": "function",
     "name": "fetch_url",
     "description": "Fetch a Vespa docs page (https://docs.vespa.ai/...) and return its full markdown in one call (e.g. the whole rank-features reference). No paging needed.",
     "parameters": {
         "type": "object",
         "properties": {"url": {"type": "string"}},
         "required": ["url"]}},
]


def dispatch_tool(name, args, ctx):
    if name == "read_schema":
        return tool_read_schema()
    if name == "write_rank_profile":
        return tool_write_rank_profile(args["first_phase"])
    if name == "run_train_eval":
        return tool_run_train_eval(args.get("inputs"), args.get("extra"), ctx=ctx)
    if name == "commit_patch":
        return tool_commit_patch(args.get("inputs"), args.get("extra"), ctx=ctx)
    if name == "query_one":
        return tool_query_one(
            args["query_text"], args["profile"], args.get("inputs"),
            args.get("extra"), queries=ctx["queries"])
    if name == "search_vespa_docs":
        return tool_search_vespa_docs(args["query"])
    if name == "fetch_url":
        return tool_fetch_url(args["url"])
    return {"error": f"unknown tool: {name}"}


def log(entry):
    with LOG.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
        fh.flush()


def transcript(s):
    # Write+flush so `tail -f` follows in real time.
    with TRANSCRIPT.open("a") as fh:
        fh.write(s + "\n")
        fh.flush()
    # Also mirror to stdout so the parent terminal sees it.
    print(s, flush=True)


def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY not set", file=sys.stderr); sys.exit(2)
    assert_local_target()  # fail before spending any OpenAI tokens
    client = OpenAI()

    LOG.write_text("")
    TRANSCRIPT.write_text("")

    # Start from a bm25-ONLY schema (agent/baseline_schema.sd), not the repo's
    # bm25+lexical schema: the agent must build its own rank profiles and never
    # sees the `lexical` profile or the winning weights. Overwrites
    # vespa-app/schemas/passage.sd for the run; restore afterwards with
    #   git checkout vespa-app/schemas/passage.sd && (cd vespa-app && vespa deploy)
    baseline_schema = (AGENT / "baseline_schema.sd").read_text()
    SCHEMA.write_text(baseline_schema)
    LAST_GOOD.write_text(baseline_schema)
    BEST_SCHEMA.write_text(baseline_schema)
    ok, msg = deploy()
    if not ok:
        raise RuntimeError(f"Failed to deploy baseline schema: {msg[-400:]}")

    queries, qrels = load_queries_qrels()
    all_qids = sorted(queries.keys())
    rotations = build_rotations(all_qids)

    # bm25 per-query baseline over ALL queries; the committed config starts as bm25.
    bm25_per_q = eval_per_query(all_qids, queries, qrels, "bm25", {}, {})
    bm25_overall = sum(bm25_per_q.values()) / len(bm25_per_q)
    print(f"{len(all_qids)} queries, {ROTATIONS} paired rotations "
          f"(seeds {BASE_SEED}..{BASE_SEED + ROTATIONS - 1}, "
          f"{round(len(all_qids) * TRAIN_FRAC)} queries each).", flush=True)
    print(f"bm25 baseline mrr@10 over all queries = {bm25_overall:.4f}", flush=True)
    log({"event": "baseline", "n_queries": len(all_qids), "rotations": ROTATIONS,
         "bm25_mrr_all": bm25_overall})

    # Committed state carries across rounds. The gate is monotonic (each commit
    # must beat the prior committed by >= eval_margin on the rotation mean), so
    # state["accepted"] is always the best config so far.
    state = {
        "committed_per_q": bm25_per_q,
        "n_commits": 0,
        "accepted": {
            "profile": "bm25", "inputs": {}, "extra": {},
            "mean_paired_delta_vs_bm25": 0.0,
            "n_rotations_positive_vs_bm25": f"0/{ROTATIONS}",
        },
    }
    ctx = {"all_qids": all_qids, "rotations": rotations,
           "queries": queries, "qrels": qrels,
           "bm25_per_q": bm25_per_q, "state": state}

    total_in_tokens = 0
    total_out_tokens = 0
    # One entry per round's commit attempt (committed or rejected). Passed into
    # each round's prompt so the agent remembers what earlier rounds tried —
    # cross-round memory the fresh context would otherwise lose.
    history = []

    for round_idx in range(ROUNDS):
        # Fresh agent context, starting from the committed schema. Redeploy it in
        # case the prior round left a rejected candidate deployed.
        SCHEMA.write_text(LAST_GOOD.read_text())
        ok, msg = deploy()
        if not ok:
            raise RuntimeError(f"Failed to redeploy committed schema: {msg[-400:]}")

        # This round's tuning rotation (rotates each round). run_train_eval shows
        # only this rotation; commit_patch checks all ROTATIONS.
        rot_idx = round_idx % ROTATIONS
        ctx["round_rotation"] = rotations[rot_idx]
        ctx["round_rotation_seed"] = BASE_SEED + rot_idx

        cur_schema = SCHEMA.read_text()
        committed = state["accepted"]
        print(f"\n=== Round {round_idx}/{ROUNDS - 1} (rotation seed {BASE_SEED + rot_idx}; "
              f"committed: {committed['profile']}, mean paired delta vs bm25 "
              f"{committed['mean_paired_delta_vs_bm25']:+.4f}) ===", flush=True)
        transcript(f"\n===== ROUND {round_idx} =====\n")

        # Compact record of earlier rounds' commit attempts (committed AND
        # rejected) — the cross-round memory the fresh context would lose, so the
        # agent can build on near-misses and not re-try rejected dead ends.
        if history:
            lines = []
            for h in history:
                a = h["attempt"]
                if a:
                    verdict = "COMMITTED" if a["committed"] else "rejected"
                    outcome = (f"{verdict} {a['inputs']} {a['extra']} (paired delta vs "
                               f"then-committed {a['mean']:+.4f}, {a['n_positive']})")
                else:
                    outcome = "no commit attempt"
                lines.append(f"  round {h['round']}: explored {h['explored']}; "
                             f"committed/attempted: {outcome}")
            attempts_block = ("Earlier rounds (explored = features already tried — don't "
                              "re-sweep those or re-commit a rejected one; branch to "
                              "something new):\n" + "\n".join(lines) + "\n\n")
        else:
            attempts_block = ""

        # Only the dynamic per-round state goes here; the protocol (the tools, the
        # paired-rotation gate, one-improvement-per-round) is in SYSTEM_PROMPT,
        # which is re-sent each round.
        user_msg = (
            f"Round {round_idx} of {ROUNDS}.\n"
            f"This round you tune on rotation seed {BASE_SEED + rot_idx} "
            f"({len(ctx['round_rotation'])} queries) via run_train_eval; commit_patch "
            f"then checks all {ROTATIONS} rotations. bm25 mrr@10 over all queries = "
            f"{bm25_overall:.4f}.\n"
            f"Best committed so far: profile={committed['profile']}, "
            f"inputs={committed['inputs']}, extra={committed['extra']} "
            f"(mean paired delta vs bm25 {committed['mean_paired_delta_vs_bm25']:+.4f} "
            f"on {committed['n_rotations_positive_vs_bm25']} rotations).\n\n"
            f"{attempts_block}"
            f"Current committed schema (your work log so far):\n"
            f"```\n{cur_schema}\n```\n\n"
            f"Make one well-tuned improvement over this, following the protocol above. "
            f"Turn budget: {MAX_TURNS_PER_ROUND}."
        )
        # Responses API: reasoning state is carried across calls via
        # previous_response_id, so after the first turn we only append the new
        # function_call_output items.
        input_items = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        previous_response_id = None
        round_start = time.time()
        commit_attempted = False
        nudged = False
        explored_keys = set()  # feature/param keys touched this round (run_train_eval + commit_patch)
        round_attempt = None   # this round's commit attempt details (set when commit_patch runs)

        for turn in range(MAX_TURNS_PER_ROUND):
            if time.time() - round_start > ROUND_DEADLINE_SECONDS:
                print("  round wall-clock exceeded; stopping round", flush=True)
                break

            kwargs = {
                "model": MODEL, "input": input_items, "tools": TOOLS,
                "reasoning": {"effort": REASONING_EFFORT},
            }
            if previous_response_id is not None:
                kwargs["previous_response_id"] = previous_response_id
            try:
                resp = client.responses.create(**kwargs)
            except Exception as e:
                print(f"  API error: {e}", flush=True)
                time.sleep(2)
                continue
            previous_response_id = resp.id
            total_in_tokens += resp.usage.input_tokens
            total_out_tokens += resp.usage.output_tokens
            transcript(f"--- turn {turn} (in={resp.usage.input_tokens} "
                       f"out={resp.usage.output_tokens}; cumulative "
                       f"in={total_in_tokens} out={total_out_tokens}) ---")

            has_function_call = False
            next_inputs = []
            for item in resp.output:
                t = getattr(item, "type", None)
                if t == "reasoning":
                    continue
                if t == "message":
                    texts = [getattr(c, "text", None) or "" for c in (item.content or [])]
                    body = "\n".join(texts).strip()
                    if body:
                        transcript("ASSIST: " + body[:2000])
                    continue
                if t == "function_call":
                    has_function_call = True
                    name = item.name
                    try:
                        args = json.loads(item.arguments or "{}")
                    except Exception:
                        args = {}
                    transcript(f"TOOL CALL: {name} {json.dumps(args)[:600]}")
                    if name == "commit_patch" and commit_attempted:
                        # One commit attempt per round: a model can emit several
                        # function_calls in one turn, but the round ends on the
                        # first commit. Refuse a second so it can't mutate the
                        # committed state again or probe the gate within a turn.
                        # No "committed" key -> the block below skips it (the real
                        # first attempt's round_attempt/history entry is preserved).
                        result = {"error": "This round already made its one "
                                  "commit_patch attempt; ignoring. The round is ending."}
                    else:
                        result = dispatch_tool(name, args, ctx)
                    if name in ("run_train_eval", "commit_patch"):
                        explored_keys.update((args.get("inputs") or {}).keys())
                        explored_keys.update((args.get("extra") or {}).keys())
                    if (name == "commit_patch" and isinstance(result, dict)
                            and "committed" in result):
                        commit_attempted = True  # one attempt per round, accept or reject
                        round_attempt = {
                            "inputs": args.get("inputs") or {},
                            "extra": args.get("extra") or {},
                            "committed": result["committed"],
                            "mean": result.get("mean_paired_delta_vs_committed", 0.0),
                            "n_positive": result.get("n_rotations_positive", ""),
                        }
                        if result["committed"]:
                            print(f"  COMMIT (round {round_idx}): {state['accepted']}", flush=True)
                            log({"event": "commit", "round": round_idx, **state["accepted"]})
                        else:
                            print(f"  round {round_idx}: commit rejected "
                                  f"(mean {result.get('mean_paired_delta_vs_committed')}, "
                                  f"{result.get('n_rotations_positive')})", flush=True)
                            log({"event": "commit_rejected", "round": round_idx,
                                 "mean_paired_delta_vs_committed": result.get("mean_paired_delta_vs_committed"),
                                 "n_rotations_positive": result.get("n_rotations_positive")})
                    result_str = json.dumps(result)
                    if len(result_str) > 4000:
                        result_str = result_str[:4000] + " ...[truncated]"
                    transcript(f"TOOL RESULT: {result_str[:1500]}")
                    next_inputs.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result_str,
                    })

            if commit_attempted:
                # One commit attempt per round — the round ends whether it was
                # accepted or rejected. The agent tunes on its single rotation,
                # then makes one held-out commit; it can't probe the gate by
                # retrying with different weights.
                print(f"  round {round_idx}: commit attempt made; ending round", flush=True)
                break
            if not has_function_call:
                # Text-only with no commit yet: every round must test something,
                # so push once for a commit attempt rather than ending empty.
                if not nudged:
                    nudged = True
                    input_items = [{"role": "user", "content": (
                        "You haven't made your commit_patch attempt yet. Make exactly one "
                        "now with your best candidate this round — even if modest. Don't "
                        "end the round without attempting.")}]
                    continue
                print(f"  round {round_idx}: no commit attempt after nudge; ending round", flush=True)
                break
            input_items = next_inputs

        # Round done: record explored features + the commit attempt for the next
        # round's prompt (cross-round memory the fresh context would lose).
        history.append({"round": round_idx, "explored": sorted(explored_keys),
                        "attempt": round_attempt})

    # Lock the deployed + on-disk schema to the committed best, so final_eval
    # scores exactly what was committed. Fail loud if this final deploy fails —
    # otherwise final_eval's on-disk guard would pass while Vespa serves a stale
    # schema.
    SCHEMA.write_text(LAST_GOOD.read_text())
    ok, msg = deploy()
    if not ok:
        raise RuntimeError(f"Failed to deploy committed best at end of run: {msg[-400:]}")
    BEST_SCHEMA.write_text(LAST_GOOD.read_text())

    print("\n=== AGENT RUN COMPLETE ===", flush=True)
    print(f"Total tokens: in={total_in_tokens:,}  out={total_out_tokens:,}", flush=True)
    print(f"Commits: {state['n_commits']}  final mean paired delta vs bm25 = "
          f"{state['accepted']['mean_paired_delta_vs_bm25']:+.4f}", flush=True)
    print(f"Final committed config: {state['accepted']}", flush=True)
    print(f"Final schema:\n{SCHEMA.read_text()}", flush=True)
    log({"event": "complete", "total_in_tokens": total_in_tokens,
         "total_out_tokens": total_out_tokens, "n_commits": state["n_commits"],
         "final": state["accepted"]})

    # Write harness/state.json for final_eval.py best. The gate is monotonic
    # (each commit beats the prior by >= eval_margin on the rotation mean), so
    # the last committed config IS the best one the loop found.
    b = state["accepted"]
    HARNESS_STATE.parent.mkdir(exist_ok=True)
    HARNESS_STATE.write_text(json.dumps({
        "best": {
            "profile": b["profile"],
            "inputs": b["inputs"],
            "extra": b["extra"],
            "mean_paired_delta_vs_bm25": b["mean_paired_delta_vs_bm25"],
            "n_rotations_positive_vs_bm25": b["n_rotations_positive_vs_bm25"],
        }
    }, indent=2))
    print(f"Wrote {HARNESS_STATE} for final_eval.py best", flush=True)


if __name__ == "__main__":
    main()
