"""Build a Vespa feed JSONL for the full MSMARCO passage corpus (8.84M docs).
Tags each doc with `subset: 1` if its doc_id is in our 650k minimarco sample
(random_state=42), else `subset: 0`.
"""
import csv
import json
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
COLLECTION = DATA / "collection.tsv"
SUBSET_TSV = DATA / "minimarco_corpus.tsv"
OUT = DATA / "msmarco_full_feed.jsonl"

CHUNK = 500_000


def main():
    print(f"Loading minimarco subset doc_ids from {SUBSET_TSV} ...", flush=True)
    subset_ids: set[int] = set()
    with SUBSET_TSV.open() as fh:
        for row in csv.reader(fh, delimiter="\t"):
            subset_ids.add(int(row[0]))
    print(f"  loaded {len(subset_ids):,}", flush=True)

    print(f"Reading {COLLECTION} in chunks of {CHUNK:,} ...", flush=True)
    n_total = 0
    n_subset = 0
    with OUT.open("w") as fh:
        for chunk in pd.read_csv(
            COLLECTION, sep="\t", names=["doc_id", "description"],
            dtype={"doc_id": "int64", "description": "string"},
            chunksize=CHUNK, na_filter=False,
        ):
            for doc_id, description in zip(chunk["doc_id"].to_numpy(), chunk["description"].to_numpy()):
                in_subset = int(doc_id) in subset_ids
                if in_subset:
                    n_subset += 1
                doc = {
                    "put": f"id:passage:passage::{int(doc_id)}",
                    "fields": {
                        "doc_id": int(doc_id),
                        "description": "" if pd.isna(description) else str(description),
                        "subset": 1 if in_subset else 0,
                    },
                }
                fh.write(json.dumps(doc, ensure_ascii=False))
                fh.write("\n")
            n_total += len(chunk)
            print(f"  wrote {n_total:,}  (subset so far: {n_subset:,})", flush=True)
    print(f"done: {n_total:,} docs, {n_subset:,} flagged subset=1", flush=True)


if __name__ == "__main__":
    main()
