"""
Embed papers and write them to S3 Vectors (same logic as Step 2, now reusable).
"""
import time

from botocore.exceptions import ClientError

from .config import (EMBED_DIM, EMBED_PRICE_PER_1K_TOKENS, INDEX_NAME,
                     NON_FILTERABLE_KEYS, VECTOR_BUCKET, log, s3vectors)
from .embeddings import embed, paper_to_text

BATCH_SIZE = 100
MAX_AUTHORS = 10


def date_as_int(iso: str) -> int:
    return int(iso[:10].replace("-", ""))


def build_vector(paper: dict, embedding: list[float]) -> dict:
    authors = paper["authors"][:MAX_AUTHORS]
    if len(paper["authors"]) > MAX_AUTHORS:
        authors.append("et al.")
    return {
        "key": paper["arxiv_id"],
        "data": {"float32": embedding},
        "metadata": {
            "primary_category": paper["primary_category"],
            "categories": paper["categories"],
            "published_date": date_as_int(paper["published"]),
            "title": paper["title"],
            "abstract": paper["abstract"],
            "authors": ", ".join(authors),
            "abs_url": paper["abs_url"],
        },
    }


def _flush(batch: list[dict]):
    if batch:
        s3vectors.put_vectors(vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME, vectors=batch)


def index_papers(papers: list[dict], progress_every: int = 0) -> dict:
    """Embed + upsert. Idempotent: keys are arXiv IDs, so re-runs overwrite."""
    batch, tokens, start = [], 0, time.time()
    for i, paper in enumerate(papers, 1):
        vector, used = embed(paper_to_text(paper))
        tokens += used
        batch.append(build_vector(paper, vector))
        if len(batch) >= BATCH_SIZE:
            _flush(batch)
            batch = []
        if progress_every and i % progress_every == 0:
            print(f"  embedded {i}/{len(papers)}")
    _flush(batch)
    return {
        "papers_indexed": len(papers),
        "tokens": tokens,
        "approx_cost_usd": round(tokens / 1000 * EMBED_PRICE_PER_1K_TOKENS, 5),
        "seconds": round(time.time() - start, 1),
    }


def ensure_index():
    """Create the vector bucket and index if missing. Safe to re-run."""
    def create(label, fn, **kwargs):
        try:
            fn(**kwargs)
            log("created", resource=label)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConflictException":
                raise
            log("already_exists", resource=label)

    create(VECTOR_BUCKET, s3vectors.create_vector_bucket, vectorBucketName=VECTOR_BUCKET)
    create(INDEX_NAME, s3vectors.create_index, vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME,
           dataType="float32", dimension=EMBED_DIM, distanceMetric="cosine",
           metadataConfiguration={"nonFilterableMetadataKeys": NON_FILTERABLE_KEYS})
