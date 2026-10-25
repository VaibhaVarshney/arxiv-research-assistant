"""
Embed papers and write them to S3 Vectors.

Step 5 adds incremental indexing: before embedding, we ask the index which
papers it already has, and only embed ones that are new or have a newer
arXiv version. The daily fetch overlaps 4 days, so this skips ~3/4 of the work.
"""
import time

from botocore.exceptions import ClientError

from .config import (EMBED_DIM, EMBED_PRICE_PER_1K_TOKENS, INDEX_NAME,
                     NON_FILTERABLE_KEYS, VECTOR_BUCKET, log, s3vectors)
from .embeddings import embed, paper_to_text

BATCH_SIZE = 100          # PutVectors accepts up to 500; GetVectors up to 100
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
            "version": paper.get("version", 1),
            "title": paper["title"],
            "abstract": paper["abstract"],
            "authors": ", ".join(authors),
            "abs_url": paper["abs_url"],
        },
    }


def existing_versions(keys: list[str]) -> dict[str, int]:
    """Which of these arXiv IDs are already indexed, and at what version?"""
    found = {}
    for i in range(0, len(keys), BATCH_SIZE):
        resp = s3vectors.get_vectors(vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME,
                                     keys=keys[i:i + BATCH_SIZE], returnData=False, returnMetadata=True)
        for v in resp.get("vectors", []):
            # Papers indexed before Step 5 have no "version" field: treat them as v1.
            found[v["key"]] = int((v.get("metadata") or {}).get("version", 1))
    return found


def select_new(papers: list[dict], already: dict[str, int]) -> list[dict]:
    return [p for p in papers
            if p["arxiv_id"] not in already or p.get("version", 1) > already[p["arxiv_id"]]]


def _flush(batch: list[dict]):
    if batch:
        s3vectors.put_vectors(vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME, vectors=batch)


def index_papers(papers: list[dict], skip_existing: bool = True, progress_every: int = 0) -> dict:
    start = time.time()
    to_embed = papers
    if skip_existing and papers:
        to_embed = select_new(papers, existing_versions([p["arxiv_id"] for p in papers]))

    batch, tokens = [], 0
    for i, paper in enumerate(to_embed, 1):
        vector, used = embed(paper_to_text(paper))
        tokens += used
        batch.append(build_vector(paper, vector))
        if len(batch) >= BATCH_SIZE:
            _flush(batch)
            batch = []
        if progress_every and i % progress_every == 0:
            print(f"  embedded {i}/{len(to_embed)}")
    _flush(batch)
    return {
        "papers_seen": len(papers),
        "skipped_existing": len(papers) - len(to_embed),
        "papers_indexed": len(to_embed),
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
