"""Evaluate on the FULL MSMARCO passage dev set (6,980 queries / 7,437 qrels).

Two evals per config:
  - mini: filter where subset=1 (our 650k sample, 543 queries with surviving qrels)
  - full: no filter (all 8.84M docs)

Reports MRR@10 for each (config, eval) pair. Compare against the README
numbers and Doug's blog for the autoresearch-Python-rewriter baseline.
"""
import csv
from pathlib import Path

from vespa.application import Vespa
from vespa.evaluation import VespaEvaluator


DATA = Path(__file__).resolve().parent.parent / "data"
QRELS = DATA / "qrels.dev.small.tsv"
QUERIES = DATA / "queries.dev.small.tsv"
# minimarco subset: the 543 dev queries whose relevant passage survived the
# 650k sample (built by build_minimarco.py). The mini scope MUST use these, not
# the full 6,980 — otherwise the ~6,437 queries whose answer isn't in the subset
# all score 0 and dilute the result (~0.038 instead of ~0.49).
MINI_QUERIES = DATA / "minimarco_queries.tsv"
MINI_QRELS = DATA / "minimarco_qrels.tsv"
VESPA_URL = "http://localhost:8080"


def load_full():
    queries = {}
    with QUERIES.open() as fh:
        for qid, q in csv.reader(fh, delimiter="\t"):
            queries[qid] = q
    qrels = {}
    with QRELS.open() as fh:
        for row in csv.reader(fh, delimiter="\t"):
            qid, _, doc_id, grade = row
            if int(grade) > 0:
                qrels.setdefault(qid, set()).add(doc_id)
    # only keep queries with at least one qrel (they all should, but safe)
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    return queries, qrels


def load_minimarco():
    """The 543 scoreable subset queries (3-col qrels written by build_minimarco)."""
    queries = {}
    with MINI_QUERIES.open() as fh:
        for qid, q in csv.reader(fh, delimiter="\t"):
            queries[qid] = q
    qrels = {}
    with MINI_QRELS.open() as fh:
        for qid, doc_id, grade in csv.reader(fh, delimiter="\t"):
            if int(grade) > 0:
                qrels.setdefault(qid, set()).add(doc_id)
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    return queries, qrels


def make_fn(profile, inputs, extra, scope="full"):
    yql_full = "select doc_id from passage where description contains ({language:'en'}text(@q))"
    yql_mini = "select doc_id from passage where description contains ({language:'en'}text(@q)) and subset=1"
    yql = yql_mini if scope == "mini" else yql_full

    def fn(q, top_k, qid=None):
        body = {
            "yql": yql, "q": q, "ranking.profile": profile,
            "hits": top_k, "language": "en", "model.locale": "en",
            "timeout": "30s",
        }
        for k, v in inputs.items():
            body[f"input.query({k})"] = v
        body.update(extra)
        return body
    return fn


def evaluate(profile, inputs, extra, scope, queries, qrels):
    app = Vespa(url=VESPA_URL)
    ev = VespaEvaluator(
        queries=queries, relevant_docs=qrels,
        vespa_query_fn=make_fn(profile, inputs, extra, scope), app=app,
        name=f"{scope}_{profile}", id_field="doc_id",
        accuracy_at_k=[1, 10], precision_recall_at_k=[10, 100],
        mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[100], write_csv=False,
    )
    return ev()


def report(label, r):
    print(f"  {label:35} mrr@10={r['mrr@10']:.4f}  ndcg@10={r['ndcg@10']:.4f}  "
          f"r@10={r['recall@10']:.4f}  r@100={r['recall@100']:.4f}  "
          f"t_avg={r['searchtime_avg']:.3f}s", flush=True)


def main():
    full_q, full_qr = load_full()
    mini_q, mini_qr = load_minimarco()
    print(f"full dev: {len(full_q):,} queries; minimarco: {len(mini_q):,} queries", flush=True)
    sets = {"mini": (mini_q, mini_qr), "full": (full_q, full_qr)}

    # Configs to evaluate
    configs = [
        ("BM25 baseline (k1=0.6 b=0.62)", "lexical", {}, {}),
        ("Manual sweep (prox=10 early=8 sw=0.05)", "lexical",
         {"w_prox": 10.0, "w_fm_early": 8.0},
         {"ranking.matching.weakand.stopwordLimit": 0.05}),
    ]

    for scope in ("mini", "full"):
        q, qr = sets[scope]
        print()
        print(f"=== scope: {scope} ({len(q):,} queries, "
              f"{'subset=1 filter' if scope == 'mini' else 'all 8.84M docs'}) ===", flush=True)
        for label, profile, inputs, extra in configs:
            r = evaluate(profile, inputs, extra, scope, q, qr)
            report(label, r)


if __name__ == "__main__":
    main()
