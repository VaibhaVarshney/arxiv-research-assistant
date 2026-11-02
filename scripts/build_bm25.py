"""
Build the BM25 keyword index from everything in S3 Vectors.

    python scripts/build_bm25.py              # writes data/bm25_index.json.gz (for local runs + eval)

In AWS the Index Lambda rebuilds it after every nightly run. To build it there
right after the first deploy (instead of waiting for tonight's run):

    sam remote invoke IndexFunction --event '{"rebuild_bm25": true}'
"""
from arxiv_rag.bm25 import BM25, from_bytes
from arxiv_rag.config import BM25_LOCAL_PATH
from arxiv_rag.indexer import rebuild_bm25

stats = rebuild_bm25(local_path=BM25_LOCAL_PATH)
print(f"Indexed {stats['bm25_papers']} papers in {stats['bm25_seconds']} s "
      f"-> {BM25_LOCAL_PATH} ({stats['bm25_bytes'] / 1e6:.1f} MB gzipped)")

bm = BM25(from_bytes(open(BM25_LOCAL_PATH, "rb").read()))
print(
    f"Vocabulary: {len(bm.postings):,} terms, avg {bm.avg_len:.0f} tokens per paper")
for term in ["model", "llm", "halting", "awareness"]:
    print(f"  idf({term!r}) = {bm.idf(term):.2f}")
