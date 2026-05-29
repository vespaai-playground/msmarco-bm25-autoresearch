"""Build the minimarco subset exactly like cheat_at_search/minimarco_data.py:

- Load full MSMARCO passage collection.tsv (~8.84M passages)
- Sample 650,000 passages with random_state=42 (then reset_index(drop=True))
- Filter qrels.dev.small.tsv to qrels whose doc_id is in the sample
- Write Vespa-format JSONL feed file (one doc per line) and filtered queries/qrels

Outputs (under data/):
- minimarco_corpus.tsv       (doc_id, description)
- minimarco_feed.jsonl       (Vespa feed format)
- minimarco_qrels.tsv        (filtered qrels)
- minimarco_queries.tsv      (filtered queries: those that have >=1 qrel in subset)
"""
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
COLLECTION = DATA / "collection.tsv"
QRELS = DATA / "qrels.dev.small.tsv"
QUERIES = DATA / "queries.dev.small.tsv"

OUT_CORPUS = DATA / "minimarco_corpus.tsv"
OUT_FEED = DATA / "minimarco_feed.jsonl"
OUT_QRELS = DATA / "minimarco_qrels.tsv"
OUT_QUERIES = DATA / "minimarco_queries.tsv"

SAMPLE_SIZE = 650_000
SEED = 42


def main():
    print(f"Reading {COLLECTION} ...", flush=True)
    passages = pd.read_csv(
        COLLECTION, sep="\t", names=["doc_id", "description"], dtype={"doc_id": "int64", "description": "string"}
    )
    print(f"  loaded {len(passages):,} passages", flush=True)

    n = min(SAMPLE_SIZE, len(passages))
    sample = passages.sample(n=n, random_state=SEED).reset_index(drop=True)
    print(f"  sampled {len(sample):,} (seed={SEED})", flush=True)

    sample[["doc_id", "description"]].to_csv(OUT_CORPUS, sep="\t", index=False, header=False)
    print(f"  wrote {OUT_CORPUS}", flush=True)

    print(f"Writing Vespa feed {OUT_FEED} ...", flush=True)
    with OUT_FEED.open("w") as fh:
        for doc_id, description in zip(sample["doc_id"].to_numpy(), sample["description"].to_numpy()):
            doc = {
                "put": f"id:passage:passage::{int(doc_id)}",
                "fields": {
                    "doc_id": int(doc_id),
                    "description": "" if pd.isna(description) else str(description),
                    "subset": 1,   # every minimarco doc is in the subset
                },
            }
            fh.write(json.dumps(doc, ensure_ascii=False))
            fh.write("\n")
    print(f"  done", flush=True)

    print("Filtering qrels/queries to subset ...", flush=True)
    qrels = pd.read_csv(
        QRELS, sep="\t", usecols=[0, 2, 3], names=["query_id", "doc_id", "grade"],
        dtype={"query_id": "int64", "doc_id": "int64", "grade": "int64"},
    )
    queries = pd.read_csv(
        QUERIES, sep="\t", names=["query_id", "query"],
        dtype={"query_id": "int64", "query": "string"},
    )
    sub_doc_ids = set(sample["doc_id"].astype("int64").tolist())
    print(f"  full dev qrels: {len(qrels):,}, queries: {len(queries):,}", flush=True)
    qrels = qrels[qrels["doc_id"].isin(sub_doc_ids)].reset_index(drop=True)
    print(f"  filtered qrels: {len(qrels):,}", flush=True)

    queries = queries.merge(qrels[["query_id"]].drop_duplicates(), on="query_id", how="inner")
    print(f"  queries with >=1 qrel in subset: {len(queries):,}", flush=True)

    qrels.to_csv(OUT_QRELS, sep="\t", index=False, header=False)
    queries.to_csv(OUT_QUERIES, sep="\t", index=False, header=False)
    print(f"  wrote {OUT_QRELS}, {OUT_QUERIES}", flush=True)


if __name__ == "__main__":
    main()
