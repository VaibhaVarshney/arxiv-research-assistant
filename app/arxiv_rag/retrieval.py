from .config import INDEX_NAME, VECTOR_BUCKET, s3vectors
from .embeddings import embed


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
    query_vector, _ = embed(query)
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
