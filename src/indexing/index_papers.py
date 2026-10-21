"""
Step 2b: Embed abstracts and store them in S3 Vectors.

Usage:
    python src/indexing/index_papers.py                 # newest file in data/
    python src/indexing/index_papers.py --file data/papers_20261020.jsonl
    python src/indexing/index_papers.py --limit 20      # quick test run

Re-running is safe: each paper's key is its arXiv ID, so it overwrites
instead of creating duplicates (idempotency).
"""
import argparse
import json
import time
from pathlib import Path

from aws_config import (INDEX_NAME, PRICE_PER_1K_TOKENS, VECTOR_BUCKET, embed,
                        paper_to_text, s3vectors)

BATCH_SIZE = 100       # S3 Vectors accepts up to 500 per call
MAX_AUTHORS = 10       # keep metadata small


def latest_file() -> Path:
    files = sorted(Path("data").glob("papers_*.jsonl"))
    if not files:
        raise SystemExit("No data/papers_*.jsonl found. Run fetch_arxiv.py first.")
    return files[-1]


def load_papers(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def date_as_int(iso: str) -> int:
    """'2026-10-17T14:02:00Z' -> 20261017, so we can do range filters like >= 20261001."""
    return int(iso[:10].replace("-", ""))


def build_vector(paper: dict, embedding: list[float]) -> dict:
    authors = paper["authors"][:MAX_AUTHORS]
    if len(paper["authors"]) > MAX_AUTHORS:
        authors.append("et al.")
    return {
        "key": paper["arxiv_id"],
        "data": {"float32": embedding},
        "metadata": {
            # Filterable (small, used in search filters)
            "primary_category": paper["primary_category"],
            "categories": paper["categories"],
            "published_date": date_as_int(paper["published"]),
            # Non-filterable (returned with results so we can show and cite them)
            "title": paper["title"],
            "abstract": paper["abstract"],
            "authors": ", ".join(authors),
            "abs_url": paper["abs_url"],
        },
    }


def flush(batch: list[dict]):
    if batch:
        s3vectors.put_vectors(vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME, vectors=batch)
        print(f"  wrote {len(batch)} vectors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, help="JSONL from fetch_arxiv.py")
    parser.add_argument("--limit", type=int, help="only index the first N papers")
    args = parser.parse_args()

    path = args.file or latest_file()
    papers = load_papers(path)[: args.limit]
    print(f"Indexing {len(papers)} papers from {path} into {VECTOR_BUCKET}/{INDEX_NAME}")

    batch, total_tokens, start = [], 0, time.time()
    for i, paper in enumerate(papers, 1):
        embedding, tokens = embed(paper_to_text(paper))
        total_tokens += tokens
        batch.append(build_vector(paper, embedding))
        if i % 25 == 0:
            print(f"  embedded {i}/{len(papers)}")
        if len(batch) >= BATCH_SIZE:
            flush(batch)
            batch = []
    flush(batch)

    cost = total_tokens / 1000 * PRICE_PER_1K_TOKENS
    print(f"\nDone in {time.time() - start:.0f}s")
    print(f"Tokens embedded: {total_tokens:,}  (approx cost ${cost:.4f})")


if __name__ == "__main__":
    main()
