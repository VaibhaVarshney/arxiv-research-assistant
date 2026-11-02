from .bm25 import get_index
from .config import (BM25_KEY, BM25_LOCAL_PATH, HYBRID_EACH, INDEX_NAME, RAW_BUCKET, RRF_K,
                     VECTOR_BUCKET, log, s3vectors)
from .embeddings import embed_query


def build_filter(category: str | None, since: int | None) -> dict | None:
    conditions = []
    if category:
        conditions.append({"primary_category": {"$eq": category}})
    if since:
        conditions.append({"published_date": {"$gte": int(since)}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(query: str, k: int = 5, category: str | None = None, since: int | None = None) -> list[dict]:
    query_vector = list(embed_query(query))
    params = {
        "vectorBucketName": VECTOR_BUCKET,
        "indexName": INDEX_NAME,
        "queryVector": {"float32": query_vector},
        "topK": k,
        "returnDistance": True,
        "returnMetadata": True,
    }
    flt = build_filter(category, since)
    if flt:
        params["filter"] = flt
    return s3vectors.query_vectors(**params)["vectors"]


# ------------------------------------------------------------------ Phase 2, Step 3
def bm25_search(query: str, k: int = 50, category: str | None = None,
                since: int | None = None) -> list[tuple[str, float]]:
    index = get_index(RAW_BUCKET, BM25_KEY, BM25_LOCAL_PATH)
    if index is None:
        log("bm25_missing", where=RAW_BUCKET or BM25_LOCAL_PATH)
        return []
    return index.search(query, k, category, since)


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    Vector scores (cosine, ~0.3-0.6) and BM25 scores (unbounded, ~5-40) live on
    different scales, so adding them is meaningless. RRF ignores the scores and
    uses only each paper's RANK in each list:

        rrf(paper) = sum over lists of 1 / (k + rank)

    A paper ranked well by BOTH retrievers beats one ranked first by only one.
    k=60 is the value from the original paper (Cormack et al., 2009); it keeps a
    single #1 from dominating. No tuning, no score normalisation needed.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking, 1):
            scores[pid] = scores.get(pid, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _fetch(keys: list[str]) -> dict[str, dict]:
    """Metadata for papers that only BM25 found (vector hits already carry theirs)."""
    found = {}
    for i in range(0, len(keys), 100):           # GetVectors takes up to 100 keys
        resp = s3vectors.get_vectors(vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME,
                                     keys=keys[i:i + 100], returnData=False, returnMetadata=True)
        found.update({v["key"]: v for v in resp.get("vectors", [])})
    return found


def hybrid_search(query: str, k: int = 50, category: str | None = None,
                  since: int | None = None) -> list[dict]:
    """Vector top-N + BM25 top-N, fused with RRF. Returns hits shaped like search() results,
    plus 'rrf_score', 'vector_rank' and 'bm25_rank'. BM25-only hits have no 'distance'."""
    vec = search(query, HYBRID_EACH, category, since)
    kw = bm25_search(query, HYBRID_EACH, category, since)
    vec_ids, kw_ids = [h["key"] for h in vec], [pid for pid, _ in kw]
    fused = rrf([vec_ids, kw_ids])[:k]

    by_id = {h["key"]: h for h in vec}
    by_id.update(_fetch([pid for pid, _ in fused if pid not in by_id]))
    vrank = {pid: i for i, pid in enumerate(vec_ids, 1)}
    krank = {pid: i for i, pid in enumerate(kw_ids, 1)}
    return [{**by_id[pid], "rrf_score": score, "vector_rank": vrank.get(pid), "bm25_rank": krank.get(pid)}
            for pid, score in fused if pid in by_id]
