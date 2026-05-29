"""HARNESS-ONLY: final eval. The agent should NOT run this.

This script evaluates the committed best (or a given profile/params) on:
  - minimarco: all 543 scoreable dev queries, subset=1 filter (IN-SAMPLE — the
    agent gated on a validation slice of these; reported like Doug's in-sample
    minimarco number and like our manual-sweep row, for an apples-to-apples
    comparison).
  - full MSMARCO: all 6,980 dev queries over the 8.84M-doc corpus. This is the
    real held-out signal — does the minimarco lift survive the 13x bigger
    corpus?

It is for the experimenter to run AFTER the agent has finished iterating.

Usage (experimenter only):
    uv run python harness/final_eval.py best
    uv run python harness/final_eval.py PROFILE [k=v ...]
"""
import csv
import json
import sys
from pathlib import Path

import requests

AGENT = Path(__file__).resolve().parent.parent    # code/agent
REPO = AGENT.parent                               # code/
APP = REPO / "vespa-app"
DATA = REPO / "data"
STATE = AGENT / "harness" / "state.json"
SCHEMA = APP / "schemas" / "passage.sd"
# Snapshot run_agent.py wrote of the schema that produced state.json's best.
BEST_SCHEMA = AGENT / "run_state" / "best_schema.sd"
QUERIES = DATA / "minimarco_queries.tsv"
QRELS = DATA / "minimarco_qrels.tsv"
# Held-out / full sets used only here:
FULL_QUERIES = DATA / "queries.dev.small.tsv"
FULL_QRELS = DATA / "qrels.dev.small.tsv"

VESPA_URL = "http://localhost:8080"


def load_tsv(path, names):
    if not path.exists():
        return None
    out = {}
    with path.open() as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if names == "queries":
                out[row[0]] = row[1]
            elif names == "qrels_4col":
                if len(row) >= 4 and int(row[3]) > 0:
                    out.setdefault(row[0], set()).add(row[2])
            elif names == "qrels_3col":
                if len(row) >= 3 and int(row[2]) > 0:
                    out.setdefault(row[0], set()).add(row[1])
    return out


def evaluate(queries, qrels, profile, inputs, extra, scope_filter=None, hits=10):
    yql = "select doc_id from passage where description contains ({language:'en'}text(@q))"
    if scope_filter == "mini":
        yql += " and subset=1"
    rr10 = 0.0
    n = 0
    for qid, q in queries.items():
        rel = qrels.get(qid)
        if not rel:
            continue
        body = {
            "yql": yql, "q": q, "ranking.profile": profile,
            "hits": 10, "language": "en", "model.locale": "en",
            "timeout": "30s",
        }
        for k, v in inputs.items():
            body[f"input.query({k})"] = v
        body.update(extra)
        r = requests.post(f"{VESPA_URL}/search/", json=body, timeout=35)
        r.raise_for_status()
        children = r.json().get("root", {}).get("children", []) or []
        for rank, hit in enumerate(children, start=1):
            if str(hit["fields"]["doc_id"]) in rel:
                rr10 += 1.0 / rank
                break
        n += 1
    return {"mrr@10": rr10 / n, "n": n}


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(2)

    if args[0] == "best":
        state = json.loads(STATE.read_text())
        b = state["best"]
        profile = b["profile"]
        inputs = b["inputs"]
        extra = b["extra"]
        # Fail loud if the deployed schema isn't the one that produced `best`.
        # The run leaves the committed schema on disk + deployed, so right after
        # a run these match. But if passage.sd was changed/redeployed since (e.g.
        # the manual sweep, or `git checkout`), evaluating `best` against it would
        # silently score a different config (Vespa defaults unknown query inputs
        # to 0). We require the snapshot run_agent saved to be the schema on disk.
        if not BEST_SCHEMA.exists():
            raise SystemExit(
                f"{BEST_SCHEMA} is missing — re-run the agent to regenerate it "
                f"(it is written whenever a new best config is accepted)."
            )
        if not SCHEMA.exists() or SCHEMA.read_text() != BEST_SCHEMA.read_text():
            raise SystemExit(
                f"Deployed schema does not match the best-config snapshot.\n"
                f"`final_eval.py best` must evaluate the schema that produced "
                f"{STATE}, but {SCHEMA}\ndiffers from {BEST_SCHEMA}.\n\n"
                f"Deploy the snapshot first:\n"
                f"  cp {BEST_SCHEMA} {SCHEMA} && (cd {APP} && vespa deploy --wait 60)\n\n"
                f"Afterwards, restore the repo's canonical schema with "
                f"`git checkout {SCHEMA}` (and redeploy) if you want the "
                f"lexical profile back."
            )
        print(f"Loading best from state: {b}", flush=True)
    else:
        profile = args[0]
        inputs, extra = {}, {}
        for a in args[1:]:
            if "=" not in a:
                raise SystemExit(
                    f"Bad argument {a!r}: expected key=value "
                    f"(e.g. w_prox=10 w_fm_early=8 sw=0.05)."
                )
            k, v = a.split("=", 1)
            if k == "sw":
                extra["ranking.matching.weakand.stopwordLimit"] = float(v)
            else:
                inputs[k] = float(v)

    # 1) minimarco: all 543 scoreable queries, subset=1 (in-sample)
    queries = load_tsv(QUERIES, "queries")
    qrels = load_tsv(QRELS, "qrels_3col")
    mini_queries = {q: queries[q] for q in queries if q in qrels}

    r = evaluate(mini_queries, qrels, profile, inputs, extra, scope_filter="mini")
    print(f"\n=== minimarco (all {r['n']} scoreable queries, subset=1, in-sample) ===")
    print(f"  mrr@10  = {r['mrr@10']:.4f}")

    # 2) full MSMARCO (if available)
    full_q = load_tsv(FULL_QUERIES, "queries")
    full_qr = load_tsv(FULL_QRELS, "qrels_4col") if FULL_QRELS.exists() else load_tsv(FULL_QRELS, "qrels_3col")
    if full_q and full_qr:
        full_q = {qid: q for qid, q in full_q.items() if qid in full_qr}
        r2 = evaluate(full_q, full_qr, profile, inputs, extra, scope_filter=None)
        print(f"\n=== FULL MSMARCO (n={r2['n']}) ===")
        print(f"  mrr@10  = {r2['mrr@10']:.4f}")
    else:
        print("\n(full MSMARCO data not found at data/queries.dev.small.tsv + "
              "data/qrels.dev.small.tsv — skipping the full-corpus eval)")


if __name__ == "__main__":
    main()
