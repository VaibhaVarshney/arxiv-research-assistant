"""
The RAG pipeline as a LangGraph graph.

    START -> retrieve --(enough evidence?)--> generate -> check_citations -> END
                     \\--(no)--> no_evidence -> END

Why a graph for something this simple? In Phase 3 the agent will loop
(search, read, search again). Starting as a graph means we add nodes and
edges later instead of rewriting.
"""
import os
import re
import time
from typing import TypedDict

from llm import generate_answer  # also sets up the import path to src/indexing
from search import search

MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.30"))  # tuned from Step 2 results
CANDIDATES_K = 8      # how many we pull from the vector store
MAX_SOURCES = 5       # how many we let the LLM read

CITATION_PATTERN = re.compile(r"\[(\d{4}\.\d{4,5})\]")


class RAGState(TypedDict, total=False):
    # inputs
    question: str
    category: str | None
    since: int | None
    # filled in by nodes
    candidates: list[dict]
    sources: list[dict]
    answer: str
    cited_ids: list[str]
    invalid_citations: list[str]
    usage: dict
    timings_ms: dict


def _similarity(hit: dict) -> float:
    return 1 - hit["distance"]


def retrieve(state: RAGState) -> RAGState:
    t0 = time.perf_counter()
    candidates = search(state["question"], CANDIDATES_K, state.get("category"), state.get("since"))
    sources = [c for c in candidates if _similarity(c) >= MIN_SIMILARITY][:MAX_SOURCES]
    return {
        "candidates": candidates,
        "sources": sources,
        "timings_ms": {"retrieve": round((time.perf_counter() - t0) * 1000)},
    }


def route_after_retrieve(state: RAGState) -> str:
    return "generate" if state["sources"] else "no_evidence"


def generate(state: RAGState) -> RAGState:
    t0 = time.perf_counter()
    result = generate_answer(state["question"], state["sources"])
    timings = {**state["timings_ms"], "generate": round((time.perf_counter() - t0) * 1000)}
    return {
        "answer": result["text"],
        "usage": {k: result[k] for k in ("input_tokens", "output_tokens", "stop_reason")},
        "timings_ms": timings,
    }


def check_citations(state: RAGState) -> RAGState:
    """Every [id] in the answer must be a paper we actually gave the model."""
    cited = list(dict.fromkeys(CITATION_PATTERN.findall(state["answer"])))
    allowed = {s["key"] for s in state["sources"]}
    invalid = [c for c in cited if c not in allowed]
    return {"cited_ids": cited, "invalid_citations": invalid}


def no_evidence(state: RAGState) -> RAGState:
    return {
        "answer": (
            "I couldn't find papers in the index that cover this well enough to answer. "
            "Try rephrasing, widening your filters, or indexing more days of papers."
        ),
        "cited_ids": [],
        "invalid_citations": [],
    }


def build_graph():
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("check_citations", check_citations)
    g.add_node("no_evidence", no_evidence)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, ["generate", "no_evidence"])
    g.add_edge("generate", "check_citations")
    g.add_edge("check_citations", END)
    g.add_edge("no_evidence", END)
    return g.compile()
