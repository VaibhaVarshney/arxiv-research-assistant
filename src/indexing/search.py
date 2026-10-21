"""
Step 2c: Semantic search over your index (retrieval only, no LLM yet).

Usage:
    python src/indexing/search.py "how to evaluate RAG systems"
    python src/indexing/search.py "agent memory" --k 3 --category cs.AI
    python src/indexing/search.py "long context vs retrieval" --since 20261015
"""
import argparse

from aws_config import INDEX_NAME, VECTOR_BUCKET, embed, s3vectors


def build_filter(category: str | None, since: int | None) -> dict | None:
    conditions = []
    if category:
        conditions.append({"primary_category": {"$eq": category}})
    if since:
        conditions.append({"published_date": {"$gte": since}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(query: str, k: int = 5, category: str | None = None, since: int | None = None) -> list[dict]:
    query_vector, _ = embed(query)   # SAME model as the papers, or the "map" won't match
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--category", help="e.g. cs.CL, cs.AI, cs.IR")
    parser.add_argument("--since", type=int, help="YYYYMMDD, e.g. 20261015")
    args = parser.parse_args()

    results = search(args.query, args.k, args.category, args.since)
    if not results:
        print("No results. Is the index empty, or is your filter too strict?")
        return

    for rank, r in enumerate(results, 1):
        m = r["metadata"]
        # Cosine distance: 0 = identical meaning, bigger = less related.
        similarity = 1 - r["distance"]
        print(f"\n{rank}. [{r['key']}] {m['title']}")
        print(f"   similarity {similarity:.3f} | {m['primary_category']} | {m['published_date']}")
        print(f"   {m['abs_url']}")
        print(f"   {m['abstract'][:200]}...")


if __name__ == "__main__":
    main()
