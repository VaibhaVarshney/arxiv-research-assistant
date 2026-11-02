"""
Index full text from your laptop, using the HTML cached by probe_fulltext.py (no arXiv requests).

    python scripts/index.py --setup              # once: creates the chunks-v1 index
    python scripts/fulltext_local.py             # every paper cached in data/html/
    python scripts/fulltext_local.py 2610.31314  # just one

In AWS the FullText Lambda does this for every paper; this is for development and for
checking that chunks land in S3 Vectors correctly before deploying.
"""
import argparse
from pathlib import Path

from arxiv_rag.config import EMBED_PRICE_PER_1K_TOKENS, INDEX_NAME, VECTOR_BUCKET, s3vectors
from arxiv_rag.fulltext import paper_chunks
from arxiv_rag.fulltext_worker import index_chunks

CACHE = Path("data/html")

parser = argparse.ArgumentParser()
parser.add_argument("ids", nargs="*")
args = parser.parse_args()
ids = args.ids or sorted(p.stem for p in CACHE.glob("*.html"))

# Paper metadata (title, category, date) comes from the abstract index.
meta = {}
for i in range(0, len(ids), 100):
    resp = s3vectors.get_vectors(vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME,
                                 keys=ids[i:i + 100], returnData=False, returnMetadata=True)
    meta.update({v["key"]: v.get("metadata") or {}
                for v in resp.get("vectors", [])})

total_tokens = 0
for pid in ids:
    m = meta.get(pid, {})
    paper = {"arxiv_id": pid, "title": m.get("title", ""), "primary_category": m.get("primary_category", ""),
             "published_date": m.get("published_date", 0), "version": m.get("version", 1)}
    chunks = paper_chunks(
        pid, (CACHE / f"{pid}.html").read_text(encoding="utf-8"), paper["title"] or None)
    tokens = index_chunks(paper, chunks)
    total_tokens += tokens
    print(
        f"  {pid}  {len(chunks):>3} chunks  {tokens:>6,} tokens  {paper['title'][:60]}")

print(f"\nDone: {len(ids)} papers, {total_tokens:,} tokens, ~${total_tokens / 1000 * EMBED_PRICE_PER_1K_TOKENS:.4f}")
