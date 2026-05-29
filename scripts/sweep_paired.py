"""Paired-rounds feature sweep.

For each round_idx i in 0..N-1, shuffle queries with seed=1234+i, take the
first 20% as train. Evaluate the baseline + every candidate config on that
same train set. Report the per-round MRR@10 delta vs baseline, then the
mean ± std of the paired delta (much tighter than unpaired comparison).

The baseline is pure BM25, and EVERY candidate's delta is measured against it
(not against a running best). So a candidate that bundles several features
(e.g. "+sw=0.05 +w_prox=10 +w_fm_early=8") reports that whole bundle's gain
over BM25. To read a feature's MARGINAL gain over another — e.g. the blog's
"earliness +0.0189 over the proximity-only anchor" — subtract the two relevant
rows: (prox+earliness combo Δ) − (prox-only Δ). Both are emitted below.

Numbers are environment-dependent (stemmer / Vespa version), so expect the same
story and ballpark as the blog, not identical figures (~±0.001).
"""
import csv
import random
import statistics
from pathlib import Path

from vespa.application import Vespa
from vespa.evaluation import VespaEvaluator


DATA = Path(__file__).resolve().parent.parent / "data"
QUERIES_TSV = DATA / "minimarco_queries.tsv"
QRELS_TSV = DATA / "minimarco_qrels.tsv"
VESPA_URL = "http://localhost:8080"

ROUNDS = 10
PROFILE = "lexical"
BASE_SEED = 1234
TRAIN_FRAC = 0.20


def load_all():
    queries, qrels = {}, {}
    with QUERIES_TSV.open() as fh:
        for qid, q in csv.reader(fh, delimiter="\t"):
            queries[qid] = q
    with QRELS_TSV.open() as fh:
        for qid, doc_id, grade in csv.reader(fh, delimiter="\t"):
            if int(grade) > 0:
                qrels.setdefault(qid, set()).add(doc_id)
    return queries, qrels


def train_split(all_qids, round_idx):
    """First TRAIN_FRAC of the queries shuffled with seed=BASE_SEED+round_idx."""
    shuffled = sorted(all_qids)
    random.Random(BASE_SEED + round_idx).shuffle(shuffled)
    return set(shuffled[: round(len(shuffled) * TRAIN_FRAC)])


def evaluate(app, qs, qrs, profile, inputs, extra, name):
    def query_fn(q, top_k, qid=None):
        body = {
            "yql": "select doc_id from passage where description contains ({language:'en'}text(@q)) and subset=1",
            "q": q, "ranking.profile": profile, "hits": top_k,
            "language": "en", "model.locale": "en",
        }
        for k, v in inputs.items():
            body[f"input.query({k})"] = v
        body.update(extra)
        return body

    ev = VespaEvaluator(
        queries=qs, relevant_docs=qrs, vespa_query_fn=query_fn, app=app,
        name=name, id_field="doc_id", mrr_at_k=[10], ndcg_at_k=[10], write_csv=False,
    )
    return ev()

# (label, inputs, extra)
# Baseline must be first.
BASELINE = ("baseline: bm25", {}, {})

CANDIDATES = [
    # stopword sweep (re-test with paired rounds)
    ("+sw=0.50", {}, {"ranking.matching.weakand.stopwordLimit": 0.50}),
    ("+sw=0.20", {}, {"ranking.matching.weakand.stopwordLimit": 0.20}),
    ("+sw=0.10", {}, {"ranking.matching.weakand.stopwordLimit": 0.10}),
    ("+sw=0.07", {}, {"ranking.matching.weakand.stopwordLimit": 0.07}),
    ("+sw=0.05", {}, {"ranking.matching.weakand.stopwordLimit": 0.05}),
    ("+sw=0.03", {}, {"ranking.matching.weakand.stopwordLimit": 0.03}),
    ("+sw=0.02", {}, {"ranking.matching.weakand.stopwordLimit": 0.02}),
    # proximity only (8-14 covers the plateau the blog reports)
    ("+w_prox=5",  {"w_prox": 5.0},  {}),
    ("+w_prox=8",  {"w_prox": 8.0},  {}),
    ("+w_prox=10", {"w_prox": 10.0}, {}),
    ("+w_prox=12", {"w_prox": 12.0}, {}),
    ("+w_prox=14", {"w_prox": 14.0}, {}),
    ("+w_prox=15", {"w_prox": 15.0}, {}),
    ("+w_prox=20", {"w_prox": 20.0}, {}),
    # combo of best so far (stopword + proximity)
    ("+sw=0.05 +w_prox=10", {"w_prox": 10.0}, {"ranking.matching.weakand.stopwordLimit": 0.05}),
    ("+sw=0.05 +w_prox=15", {"w_prox": 15.0}, {"ranking.matching.weakand.stopwordLimit": 0.05}),
    ("+sw=0.07 +w_prox=10", {"w_prox": 10.0}, {"ranking.matching.weakand.stopwordLimit": 0.07}),
    ("+sw=0.10 +w_prox=10", {"w_prox": 10.0}, {"ranking.matching.weakand.stopwordLimit": 0.10}),
    # earliness only (on top of bm25 baseline)
    ("+w_fm_early=4",  {"w_fm_early": 4.0},  {}),
    ("+w_fm_early=6",  {"w_fm_early": 6.0},  {}),
    ("+w_fm_early=8",  {"w_fm_early": 8.0},  {}),
    ("+w_fm_early=10", {"w_fm_early": 10.0}, {}),
    ("+w_fm_early=12", {"w_fm_early": 12.0}, {}),
    # full combo: stopword + proximity + earliness (the final config)
    ("+sw=0.05 +w_prox=10 +w_fm_early=6",  {"w_prox": 10.0, "w_fm_early": 6.0},
     {"ranking.matching.weakand.stopwordLimit": 0.05}),
    ("+sw=0.05 +w_prox=10 +w_fm_early=8",  {"w_prox": 10.0, "w_fm_early": 8.0},
     {"ranking.matching.weakand.stopwordLimit": 0.05}),
    ("+sw=0.05 +w_prox=10 +w_fm_early=10", {"w_prox": 10.0, "w_fm_early": 10.0},
     {"ranking.matching.weakand.stopwordLimit": 0.05}),
]


def fmt(xs):
    if len(xs) <= 1:
        return f"{xs[0]:+.4f}"
    return f"{statistics.mean(xs):+.4f} ± {statistics.stdev(xs):.4f}"


def main():
    all_queries, all_qrels = load_all()
    all_qids = sorted(all_queries.keys())
    app = Vespa(url=VESPA_URL)

    splits = [train_split(all_qids, i) for i in range(ROUNDS)]

    # 1) Baseline per round
    base_label, base_in, base_ex = BASELINE
    base_mrrs, base_ndcgs = [], []
    print(f"=== {base_label} ===", flush=True)
    for i, qids in enumerate(splits):
        qs = {q: all_queries[q] for q in qids if q in all_queries}
        qr = {q: all_qrels[q] for q in qids if q in all_qrels}
        r = evaluate(app, qs, qr, PROFILE, base_in, base_ex, f"base_r{i}")
        base_mrrs.append(r['mrr@10']); base_ndcgs.append(r['ndcg@10'])
        print(f"  round {i}: mrr@10={r['mrr@10']:.4f}  ndcg@10={r['ndcg@10']:.4f}", flush=True)
    print(f"  mean: mrr@10={statistics.mean(base_mrrs):.4f}  ndcg@10={statistics.mean(base_ndcgs):.4f}", flush=True)

    print()
    print(f"{'candidate':40} {'Δmrr@10 (paired)':25} {'Δndcg@10 (paired)':25}")
    print("-" * 95)

    # 2) Each candidate, paired diffs
    for label, ins, ex in CANDIDATES:
        diffs_m, diffs_n = [], []
        for i, qids in enumerate(splits):
            qs = {q: all_queries[q] for q in qids if q in all_queries}
            qr = {q: all_qrels[q] for q in qids if q in all_qrels}
            r = evaluate(app, qs, qr, PROFILE, ins, ex, f"{label}_r{i}")
            diffs_m.append(r['mrr@10'] - base_mrrs[i])
            diffs_n.append(r['ndcg@10'] - base_ndcgs[i])
        print(f"{label:40} {fmt(diffs_m):25} {fmt(diffs_n):25}", flush=True)


if __name__ == "__main__":
    main()
