"""
Phase 3, Step 2: passage nomination.

The problem (answer eval v2 baseline, 6 of ~10 misses per run):
    Level 1 judges a paper ONLY by its abstract. A question about a detail that lives in
    the body ("which model was the primary judge?", "how many fact paragraphs?") doesn't
    match the abstract, so the right paper either
      - never reaches the 50 vector candidates ("far": a04, a06, a07), or
      - is found but scored under MIN_RELEVANCE and dropped ("cut": b10, b24, b26).
    Asking for more candidates doesn't help: no gold paper was waiting at rank 6-50.
    The fix is a different SIGNAL, not a bigger net.

The idea:
    Run a second search, over the full-text chunks of ALL papers, alongside the abstract
    search. Rerank those chunks. If a paper has a passage that answers the question well,
    that passage can NOMINATE the paper into the sources, even if its abstract scored low.

        abstracts:  vector top 50 -> rerank -> abstract_score per paper   (as before)
        passages:   chunk  top 50 -> rerank -> passage_score per paper    (new, in parallel)
        merge:      a paper qualifies if   abstract_score >= MIN_RELEVANCE
                                       or  passage_score  >= MIN_NOMINATION
                    order by max(abstract_score, passage_score), keep 5

Guard rails:
    - MAX_NOMINATED: at most 2 of the 5 sources may get in through a passage alone. Broad
      questions ("what are recent approaches to X?") are better served by abstracts, and
      this caps how many abstract-qualified papers nomination can push out.
    - MIN_NOMINATION is calibrated separately from MIN_RELEVANCE (eval_retrieval.py
      --mode passages), because a passage is short and specific: Cohere scores it on a
      different curve than an abstract.
    - The nominating chunks are handed to the evidence step, so the passage that got the
      paper in is guaranteed to be considered when choosing what the LLM reads.
    - Off by default (USE_NOMINATION). If the chunk search or its rerank fails, retrieval
      carries on with abstracts only.
"""
from .config import (CHUNK_INDEX_NAME, INDEX_NAME, MAX_NOMINATED, MAX_SOURCES, MIN_NOMINATION,
                     MIN_RELEVANCE, NOMINATION_CHUNKS, VECTOR_BUCKET, s3vectors)
from .embeddings import embed_query
from .evidence import passage_text
from .rerank import rerank
from .retrieval import build_filter


def global_chunk_search(question: str, k: int = NOMINATION_CHUNKS, category: str | None = None,
                        since: int | None = None) -> list[dict]:
    """Like evidence.chunk_search, but over every paper's chunks, not just the chosen ones."""
    params = {"vectorBucketName": VECTOR_BUCKET, "indexName": CHUNK_INDEX_NAME,
              "queryVector": {"float32": list(embed_query(question))}, "topK": k,
              "returnDistance": True, "returnMetadata": True}
    flt = build_filter(category, since)     # chunks carry primary_category + published_date too
    if flt:
        params["filter"] = flt
    return s3vectors.query_vectors(**params)["vectors"]


def nominate(question: str, category: str | None = None, since: int | None = None,
             allowed: set[str] | None = None, k: int = NOMINATION_CHUNKS) -> dict[str, list[dict]]:
    """{arxiv_id: [chunk hits, best first]}. Every hit carries a passage rerank_score.

    allowed: eval-only snapshot restriction (same as graph.ALLOWED_IDS).
    """
    hits = global_chunk_search(question, k, category, since)
    if allowed is not None:
        hits = [h for h in hits if h["metadata"].get("arxiv_id") in allowed]
    if not hits:
        return {}
    by_paper: dict[str, list[dict]] = {}
    for h in rerank(question, hits, len(hits), text_fn=passage_text):
        by_paper.setdefault(h["metadata"]["arxiv_id"], []).append(h)
    return by_paper


def fetch_papers(ids: list[str]) -> dict[str, dict]:
    """Abstract-index records for papers that came in through a passage alone
    (the LLM still gets title + abstract for every source)."""
    if not ids:
        return {}
    resp = s3vectors.get_vectors(vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME,
                                 keys=ids[:100], returnData=False, returnMetadata=True)
    return {v["key"]: {"key": v["key"], "metadata": v.get("metadata") or {}}
            for v in resp.get("vectors", [])}


def merge(abstract_ranked: list[dict], nominated: dict[str, list[dict]], fetch=fetch_papers,
          max_sources: int = MAX_SOURCES, min_relevance: float = MIN_RELEVANCE,
          min_nomination: float = MIN_NOMINATION, max_nominated: int = MAX_NOMINATED):
    """Combine both signals. Returns (ordered, sources).

    ordered: every paper either search saw, sources first, then the rest by score.
    sources: what goes to the evidence step and the LLM.
    With no nominations this gives exactly the Phase 2 result: the top 5 abstracts
    scoring >= min_relevance.
    """
    entries: dict[str, dict] = {}
    for h in abstract_ranked:
        entries[h["key"]] = {**h, "abstract_score": h.get("rerank_score"), "passage_score": None}
    for pid, hits in nominated.items():
        e = entries.setdefault(pid, {"key": pid, "metadata": None, "abstract_score": None})
        e["passage_score"] = hits[0]["rerank_score"]
    for e in entries.values():
        a, p = e["abstract_score"] or 0.0, e["passage_score"] or 0.0
        e["score"] = max(a, p)
        e["via_abstract"] = a >= min_relevance
        e["via_passage"] = p >= min_nomination

    ranked = sorted(entries.values(), key=lambda e: -e["score"])
    sources, used = [], 0
    for e in ranked:
        if len(sources) >= max_sources:
            break
        if e["via_abstract"]:
            sources.append(e)
        elif e["via_passage"] and used < max_nominated:
            sources.append(e)
            used += 1
    picked = {e["key"] for e in sources}
    ordered = sources + [e for e in ranked if e["key"] not in picked]

    missing = [e["key"] for e in sources if e["metadata"] is None]
    if missing:
        found = fetch(missing)
        for e in sources:
            if e["metadata"] is None and e["key"] in found:
                e["metadata"] = found[e["key"]]["metadata"]
        # a chunk can outlive its paper's abstract vector (e.g. mid re-index): skip it
        sources = [e for e in sources if e["metadata"]]
    return ordered, sources
