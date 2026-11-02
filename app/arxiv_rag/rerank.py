"""
Phase 2, Step 2: re-score vector search candidates with a cross-encoder.

Why this helps with bunched scores:
  An embedding model (bi-encoder) turns the question and each paper into vectors
  SEPARATELY, then compares them. Every RAG paper ends up in roughly the same
  neighbourhood, so a broad question sits ~equally close to all of them.

  A reranker (cross-encoder) reads the question and one paper TOGETHER, so every
  word of the question can attend to every word of the abstract. It can notice
  "this is about evaluating retrievers, but they asked about evaluating answers".
  That's far more accurate, but too slow to run over 3,000+ papers, so we do:

      vector search -> top 50 (fast, cheap, decent recall)
      reranker      -> re-order those 50 (slow-ish, precise)

  The reranker can only reorder what it's given. If vector search never surfaces
  the right paper, reranking can't save it. That's what hybrid search (Step 3) fixes.
"""
from .config import RERANK_MODEL_ARN, bedrock_agent


def doc_text(hit: dict) -> str:
    """What the reranker reads for each candidate. Same content we embed: title + abstract."""
    m = hit["metadata"]
    return f"{m['title']}\n\n{m['abstract']}"


def rerank(query: str, hits: list[dict], top_n: int) -> list[dict]:
    """Return hits re-ordered by relevance, each with a 'rerank_score' between 0 and 1."""
    if not hits:
        return []
    resp = bedrock_agent.rerank(
        queries=[{"type": "TEXT", "textQuery": {"text": query}}],
        sources=[{"type": "INLINE",
                  "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": doc_text(h)}}}
                 for h in hits],
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": RERANK_MODEL_ARN},
                "numberOfResults": min(top_n, len(hits)),
            },
        },
    )
    # Results come back best first. "index" points into the list we sent.
    return [{**hits[r["index"]], "rerank_score": r["relevanceScore"]} for r in resp["results"]]
