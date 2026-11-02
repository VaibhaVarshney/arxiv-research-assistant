"""
Phase 2, Step 4c: two-level retrieval.

    Level 1 (papers):    abstracts + reranker pick the top papers        (Steps 1-3, unchanged)
    Level 2 (passages):  search the full-text chunks of ONLY those papers,
                         rerank the passages, keep the best few per paper

Why two levels instead of searching all ~170k chunks directly?
  - Ranking stays on abstracts, which the eval shows work well (hit@1 0.95 on specific
    questions). Chunks from 3,700 papers would bring back the score-bunching problem,
    because many papers have near-identical "we evaluate on..." paragraphs.
  - A filter on arxiv_id makes level 2 a small, precise search: 5 papers x ~45 chunks.
  - The LLM gets the abstract (the paper's own summary) PLUS the passages with the
    specifics: numbers from result tables, method details, limitations.

Papers without full text (no HTML, or not processed yet) simply keep abstract-only evidence.
"""
import time

from .config import (CHUNK_CANDIDATES, CHUNK_INDEX_NAME, MAX_PASSAGES, MAX_PASSAGES_PER_PAPER,
                     MIN_PASSAGE_RELEVANCE, VECTOR_BUCKET, log, s3vectors)
from .embeddings import embed_query
from .rerank import rerank


def passage_text(hit: dict) -> str:
    """What the reranker reads for a chunk: where it sits in the paper, then the passage."""
    m = hit["metadata"]
    return f"{m.get('title', '')} > {m.get('section', '')}\n\n{m.get('text', '')}"


def chunk_search(question: str, paper_ids: list[str], k: int = CHUNK_CANDIDATES) -> list[dict]:
    if not paper_ids:
        return []
    return s3vectors.query_vectors(
        vectorBucketName=VECTOR_BUCKET, indexName=CHUNK_INDEX_NAME,
        queryVector={"float32": list(embed_query(question))}, topK=k,
        filter={"arxiv_id": {"$in": paper_ids}},
        returnDistance=True, returnMetadata=True,
    )["vectors"]


def select_passages(ranked: list[dict], max_total: int = MAX_PASSAGES,
                    per_paper: int = MAX_PASSAGES_PER_PAPER, min_score: float = MIN_PASSAGE_RELEVANCE) -> dict:
    """Best-first, with a per-paper cap. Returns {arxiv_id: [passage, ...]} in reading order."""
    picked: dict[str, list[dict]] = {}
    total = 0
    for h in ranked:
        score = h.get("rerank_score", 1 - h.get("distance", 1))
        if total >= max_total or score < min_score:
            break
        pid = h["metadata"]["arxiv_id"]
        if len(picked.get(pid, [])) >= per_paper:
            continue
        m = h["metadata"]
        picked.setdefault(pid, []).append({"chunk": int(m.get("chunk", 0)), "section": m.get("section", ""),
                                           "text": m.get("text", ""), "score": round(score, 3)})
        total += 1
    for passages in picked.values():
        # read them in the paper's own order
        passages.sort(key=lambda p: p["chunk"])
    return picked


def gather_evidence(question: str, sources: list[dict]) -> dict:
    """Returns {"passages": {arxiv_id: [...]}, "timings_ms": {...}, "chunk_hits": n}."""
    t0 = time.perf_counter()
    hits = chunk_search(question, [s["key"] for s in sources])
    t_search = round((time.perf_counter() - t0) * 1000)
    if not hits:
        return {"passages": {}, "chunk_hits": 0, "timings_ms": {"chunks": t_search}}
    try:
        ranked = rerank(question, hits, len(hits), text_fn=passage_text)
    except Exception as e:
        # fall back to vector order
        log("passage_rerank_failed", error=repr(e))
        ranked = sorted(hits, key=lambda h: h["distance"])
    return {"passages": select_passages(ranked), "chunk_hits": len(hits),
            "timings_ms": {"chunks": t_search,
                           "chunk_rerank": round((time.perf_counter() - t0) * 1000) - t_search}}
