"""
Semantic search only (no LLM).

    python scripts/search.py "how to evaluate RAG systems" --k 5 --category cs.CL --since 20261015
"""
import argparse

from arxiv_rag.retrieval import search

parser = argparse.ArgumentParser()
parser.add_argument("query")
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--category")
parser.add_argument("--since", type=int)
args = parser.parse_args()

for rank, r in enumerate(search(args.query, args.k, args.category, args.since), 1):
    m = r["metadata"]
    print(f"\n{rank}. [{r['key']}] {m['title']}")
    print(f"   similarity {1 - r['distance']:.3f} | {m['primary_category']} | {m['published_date']}")
    print(f"   {m['abs_url']}")
