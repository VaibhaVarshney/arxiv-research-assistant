"""
Freeze the corpus the eval labels were made against.

    python scripts/eval_snapshot.py                         # from data/bm25_index.json.gz
    python scripts/eval_snapshot.py --until 20261101        # only papers published on or before a date

Run it from the local BM25 file you built BEFORE new papers were ingested (that's the
corpus you labelled against). After that, eval_retrieval.py ignores anything newer.
"""
import argparse
from pathlib import Path

from arxiv_rag.bm25 import from_bytes
from arxiv_rag.config import BM25_LOCAL_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--index", default=BM25_LOCAL_PATH)
parser.add_argument("--until", type=int,
                    help="keep papers with published_date <= this (YYYYMMDD)")
parser.add_argument("--out", default="evals/snapshot_ids.txt")
args = parser.parse_args()

idx = from_bytes(Path(args.index).read_bytes())
ids = [pid for pid, d in zip(idx["ids"], idx["dates"])
       if not args.until or d <= args.until]
Path(args.out).write_text("\n".join(sorted(ids)) + "\n")
print(f"Snapshot: {len(ids)} papers -> {args.out}")
