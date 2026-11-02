"""
The RAG pipeline as a LangGraph graph.

    START -> retrieve --(enough evidence?)--> evidence -> generate -> check_citations -> END
                     \\--(no)--> no_evidence -> END

Step 4c: "evidence" searches the full-text chunks of the papers retrieve picked, and
hands the best passages to generate alongside each abstract (two-level retrieval).

Phase 2: retrieve is now multi-stage.
    Step 3: hybrid search casts the net. Vector top 50 + BM25 top 50, fused with RRF.
    Step 2: the reranker re-scores the fused 50; we keep those above MIN_RELEVANCE.
If the reranker is down, we fall back to the vector hits with the cosine floor, so the
API still answers (BM25-only hits have no comparable score, so they're dropped then).
"""
import re
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .config import (CANDIDATES_K, LLM_MODEL_ID, MAX_SOURCES, MIN_RELEVANCE, MIN_SIMILARITY,
                     RERANK_CANDIDATES, USE_FULLTEXT, USE_HYBRID, USE_RERANK, log)
from .evidence import gather_evidence
from .llm import generate_answer
from .rerank import rerank
from .retrieval import hybrid_search, search

CITATION_PATTERN = re.compile(r"\[(\d{4}\.\d{4,5})\]")


class RAGState(TypedDict, total=False):
    question: str
    category: str | None
    since: int | None
    candidates: list[dict]
    sources: list[dict]
    answer: str
    cited_ids: list[str]
    invalid_citations: list[str]
    usage: dict
    timings_ms: dict
    reranked: bool
    # {arxiv_id: [{chunk, section, text, score}]} from the full text
    passages: dict


def _similarity(hit: dict) -> float | None:
    # BM25-only hits have no cosine score
    return 1 - hit["distance"] if "distance" in hit else None


def _gather(q, cat, since, k):
    if USE_HYBRID:
        try:
            return hybrid_search(q, k, cat, since)
        except Exception as e:
            log("hybrid_failed", error=repr(e))
    return search(q, k, cat, since)


def retrieve(state: RAGState) -> RAGState:
    t0 = time.perf_counter()
    q, cat, since = state["question"], state.get(
        "category"), state.get("since")
    if USE_RERANK:
        candidates = _gather(q, cat, since, RERANK_CANDIDATES)
        t_search = round((time.perf_counter() - t0) * 1000)
        try:
            ranked = rerank(q, candidates, MAX_SOURCES)
            sources = [h for h in ranked if h["rerank_score"] >= MIN_RELEVANCE]
            return {"candidates": ranked, "sources": sources, "reranked": True,
                    "timings_ms": {"search": t_search,
                                   "rerank": round((time.perf_counter() - t0) * 1000) - t_search}}
        except Exception as e:
            # Degrade, don't fail: a vector-only answer beats a 500 error.
            log("rerank_failed", error=repr(e))
    else:
        candidates = search(q, CANDIDATES_K, cat, since)
    candidates = sorted((c for c in candidates if "distance" in c),
                        key=lambda c: c["distance"])[:CANDIDATES_K]
    sources = [c for c in candidates if _similarity(
        c) >= MIN_SIMILARITY][:MAX_SOURCES]
    return {"candidates": candidates, "sources": sources, "reranked": False,
            "timings_ms": {"search": round((time.perf_counter() - t0) * 1000)}}


def route_after_retrieve(state: RAGState) -> str:
    return "evidence" if state["sources"] else "no_evidence"


def evidence(state: RAGState) -> RAGState:
    """Level 2: best full-text passages from the chosen papers. Never fatal: abstracts still work."""
    if not USE_FULLTEXT:
        return {"passages": {}}
    try:
        ev = gather_evidence(state["question"], state["sources"])
    except Exception as e:
        log("evidence_failed", error=repr(e))
        return {"passages": {}}
    return {"passages": ev["passages"], "timings_ms": {**state["timings_ms"], **ev["timings_ms"]}}


def generate(state: RAGState) -> RAGState:
    t0 = time.perf_counter()
    result = generate_answer(
        state["question"], state["sources"], state.get("passages") or {})
    return {
        "answer": result["text"],
        "usage": {k: result[k] for k in ("input_tokens", "output_tokens", "stop_reason")},
        "timings_ms": {**state["timings_ms"], "generate": round((time.perf_counter() - t0) * 1000)},
    }


def check_citations(state: RAGState) -> RAGState:
    cited = list(dict.fromkeys(CITATION_PATTERN.findall(state["answer"])))
    allowed = {s["key"] for s in state["sources"]}
    return {"cited_ids": cited, "invalid_citations": [c for c in cited if c not in allowed]}


def no_evidence(state: RAGState) -> RAGState:
    return {
        "answer": ("I couldn't find papers in the index that cover this well enough to answer. "
                   "Try rephrasing, widening your filters, or indexing more days of papers."),
        "cited_ids": [], "invalid_citations": [],
    }


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("evidence", evidence)
    g.add_node("generate", generate)
    g.add_node("check_citations", check_citations)
    g.add_node("no_evidence", no_evidence)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, [
                            "evidence", "no_evidence"])
    g.add_edge("evidence", "generate")
    g.add_edge("generate", "check_citations")
    g.add_edge("check_citations", END)
    g.add_edge("no_evidence", END)
    return g.compile()


_app = None


def run_query(question: str, category: str | None = None, since: int | None = None) -> dict:
    """Run the graph and return a JSON-friendly result (used by Lambda and the CLI)."""
    global _app
    if _app is None:            # compile once per container, reuse when warm
        _app = build_graph()
    t0 = time.perf_counter()
    state = _app.invoke(
        {"question": question, "category": category, "since": since})
    cited = set(state.get("cited_ids", []))
    passages = state.get("passages") or {}

    def summarize(hit):
        m = hit["metadata"]
        return {"id": hit["key"], "title": m["title"], "url": m["abs_url"],
                "published_date": m["published_date"],
                "similarity": round(_similarity(hit), 3) if "distance" in hit else None,
                "found_by": [name for name, key in (("vector", "vector_rank"), ("bm25", "bm25_rank"))
                             if hit.get(key)] or None,
                "relevance": round(hit["rerank_score"], 3) if "rerank_score" in hit else None,
                # Which parts of the full text backed the answer (short snippets, not the paper).
                "passages": [{"section": p["section"], "score": p["score"], "snippet": p["text"][:240]}
                             for p in passages.get(hit["key"], [])],
                "cited": hit["key"] in cited}

    return {
        "question": question,
        "answer": state["answer"],
        "sources": [summarize(s) for s in state.get("sources", [])],
        "closest_below_threshold": [summarize(c) for c in state.get("candidates", [])[:3]]
        if not state.get("sources") else [],
        "invalid_citations": state.get("invalid_citations", []),
        "usage": state.get("usage", {}),
        "timings_ms": {**state.get("timings_ms", {}), "total": round((time.perf_counter() - t0) * 1000)},
        "model": LLM_MODEL_ID,
        "reranked": state.get("reranked", False),
        "fulltext_papers": sum(1 for s in state.get("sources", []) if passages.get(s["key"])),
        "min_similarity": MIN_SIMILARITY,
        "min_relevance": MIN_RELEVANCE,
    }
