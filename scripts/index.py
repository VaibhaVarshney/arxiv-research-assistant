"""
Embed a local JSONL file into S3 Vectors.

    python scripts/index.py                    # newest file in data/
    python scripts/index.py --file data/papers_20261021.jsonl --limit 20
    python scripts/index.py --setup            # create bucket + index if missing
    python scripts/index.py --force            # re-embed even papers already indexed
"""
import argparse
import json
from pathlib import Path

from arxiv_rag.indexer import ensure_index, index_papers

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=Path)
parser.add_argument("--limit", type=int)
parser.add_argument("--setup", action="store_true")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

if args.setup:
    ensure_index()
    raise SystemExit

path = args.file or sorted(Path("data").glob("papers_*.jsonl"))[-1]
papers = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
papers = papers[: args.limit]
print(f"Indexing {len(papers)} papers from {path}")
print(index_papers(papers, skip_existing=not args.force, progress_every=100))
