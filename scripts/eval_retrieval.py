"""
Step 1b: measure retrieval quality, so every Phase 2 change is proven, not guessed.

    python scripts/eval_retrieval.py evals/retrieval_v1.jsonl
    python scripts/eval_retrieval.py evals/retrieval_v1.jsonl --mode rerank
    python scripts/eval_retrieval.py evals/retrieval_v1.jsonl --mode vector --k 50   # reranker's ceiling

Modes:  vector | bm25 | hybrid (vector+bm25, RRF) | rerank (vector -> rerank)
        hybrid_rerank (vector+bm25 -> RRF -> rerank: the Step 3 production pipeline)
        passages        (Phase 3 Step 2) papers ranked by their best full-text passage alone.
                        Its threshold check calibrates MIN_NOMINATION.
        rerank_nominate (Phase 3 Step 2) abstract rerank + passage nomination, merged exactly
                        as graph.py does it. Compare against --mode rerank.
BM25 modes need the local index first:  python scripts/build_bm25.py

Metrics (higher is better unless noted):
  hit@1      did the right paper come first?
  recall@5   share of relevant papers that made the top 5 (what the LLM sees)
  mrr        1 / rank of the first relevant paper, averaged. Rank 1 = 1.0, rank 4 = 0.25
  margin     score of best relevant paper minus best non-relevant one.
             Positive = the right paper stands out. This is the number your
             "everything scores 0.42-0.47" problem shows up in.
  spread     top-1 score minus top-5 score. Small spread = scores are bunched.

Question types: "specific" (one paper, paraphrased), "broad" (many papers),
"keyword" (uses the paper's own name or term, the case BM25 is built for),
and "none".

Questions with "type": "none" have no answer in the index (off-topic on purpose).
They don't count toward the metrics above. Instead the script compares the top
score on those against the best relevant score on real questions, which tells
you where to set the "is this relevant enough to answer?" threshold.

Snapshot: the live index grows every night, and new (unjudged) papers can push labelled
ones down, so scores drift even when nothing changed in the code. If evals/snapshot_ids.txt
exists, any paper not in it is dropped from results before scoring, so the eval always
judges the same corpus the labels were made against. Create it once with:
    python scripts/eval_snapshot.py

Each run is saved to evals/results/ so you can compare modes side by side.
"""
import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from arxiv_rag.config import RERANK_CANDIDATES
from arxiv_rag.nominate import merge, nominate
from arxiv_rag.rerank import rerank as rerank_hits
from arxiv_rag.retrieval import bm25_search, hybrid_search, search


# ---- Retrievers. Each returns [(paper_id, score), ...] best first.
# Phase 2 adds "bm25" and "hybrid" here too, so the eval itself never changes.
# Note: scores are on different scales per mode (cosine vs rerank 0-1), so compare
# margin/spread only WITHIN a mode. Compare hit@1, recall@5 and mrr ACROSS modes.
def vector(question: str, k: int) -> list[tuple[str, float]]:
    return [(h["key"], 1 - h["distance"]) for h in search(question, k)]


def rerank(question: str, k: int) -> list[tuple[str, float]]:
    candidates = search(question, RERANK_CANDIDATES)
    return [(h["key"], h["rerank_score"]) for h in rerank_hits(question, candidates, k)]


def bm25(question: str, k: int) -> list[tuple[str, float]]:
    return bm25_search(question, k)


def hybrid(question: str, k: int) -> list[tuple[str, float]]:
    return [(h["key"], h["rrf_score"]) for h in hybrid_search(question, k)]


def hybrid_rerank(question: str, k: int) -> list[tuple[str, float]]:
    candidates = hybrid_search(question, RERANK_CANDIDATES)
    return [(h["key"], h["rerank_score"]) for h in rerank_hits(question, candidates, k)]


def passages(question: str, k: int) -> list[tuple[str, float]]:
    by_paper = nominate(question, allowed=snapshot)
    ranked = sorted(((pid, hits[0]["rerank_score"]) for pid, hits in by_paper.items()),
                    key=lambda x: -x[1])
    return ranked[:k]


def rerank_nominate(question: str, k: int) -> list[tuple[str, float]]:
    candidates = [h for h in search(question, RERANK_CANDIDATES)
                  if snapshot is None or h["key"] in snapshot]
    ranked = rerank_hits(question, candidates, len(candidates))
    # fetch=no-op: the eval only needs the order, not the abstracts
    ordered, _ = merge(ranked, nominate(question, allowed=snapshot), fetch=lambda ids: {})
    return [(e["key"], e["score"]) for e in ordered[:k]]


RETRIEVERS = {"vector": vector, "bm25": bm25, "hybrid": hybrid, "rerank": rerank,
              "hybrid_rerank": hybrid_rerank, "passages": passages,
              "rerank_nominate": rerank_nominate}


def score_one(ranked: list[tuple[str, float]], relevant: set[str]) -> dict:
    top_score = ranked[0][1] if ranked else None
    # "none" question: only the top score matters (it should be low)
    if not relevant:
        return {"hit@1": None, "recall@5": None, "found": None, "mrr": None, "margin": None,
                "spread": None, "first_relevant_rank": None, "top_score": top_score,
                "best_relevant_score": None,
                "top": [{"id": pid, "score": round(s, 4), "relevant": False} for pid, s in ranked[:5]]}
    ids = [pid for pid, _ in ranked]
    first = next((i for i, pid in enumerate(ids, 1) if pid in relevant), None)
    rel_scores = [s for pid, s in ranked if pid in relevant]
    other_scores = [s for pid, s in ranked if pid not in relevant]
    return {
        "hit@1": float(first == 1),
        "recall@5": len(relevant & set(ids[:5])) / len(relevant),
        # relevant paper anywhere in the top k
        "found": float(first is not None),
        "mrr": 1 / first if first else 0.0,
        "margin": (max(rel_scores) - max(other_scores)) if rel_scores and other_scores else None,
        "spread": ranked[0][1] - ranked[min(4, len(ranked) - 1)][1] if ranked else None,
        "first_relevant_rank": first,
        "top_score": top_score,
        # Only papers in the top 5 can become sources, so a relevant paper at rank 11
        # shouldn't drag the threshold down: it's out of reach whatever the threshold.
        "best_relevant_score": max((s for pid, s in ranked[:5] if pid in relevant), default=None),
        "top": [{"id": pid, "score": round(s, 4), "relevant": pid in relevant} for pid, s in ranked[:5]],
    }


def mean(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return round(statistics.mean(vals), 3) if vals else None


def load(path: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("question", "").strip()]


parser = argparse.ArgumentParser()
parser.add_argument("evalset")
parser.add_argument("--mode", default="vector", choices=sorted(RETRIEVERS))
parser.add_argument("--k", type=int, default=20,
                    help="how deep to look for the relevant paper")
parser.add_argument("--snapshot", default="evals/snapshot_ids.txt",
                    help="file of arXiv IDs the labels were made against ('' to disable)")
args = parser.parse_args()

snapshot = None
if args.snapshot and Path(args.snapshot).exists():
    snapshot = set(Path(args.snapshot).read_text().split())

questions = load(args.evalset)
if not questions:
    raise SystemExit(
        "No questions filled in yet. Write them in the eval file first.")

retrieve = RETRIEVERS[args.mode]
results, latencies = [], []
for q in questions:
    t0 = time.perf_counter()
    ranked = retrieve(q["question"], args.k)
    if snapshot:
        ranked = [(pid, s) for pid, s in ranked if pid in snapshot]
    latencies.append((time.perf_counter() - t0) * 1000)
    results.append({"id": q["id"], "type": q.get("type", "specific"), "question": q["question"],
                    **score_one(ranked, set(q["relevant"]))})

print(f"\n{'id':<5} {'type':<9} {'rank':>4} {'margin':>7}  question   (for 'none' rows, rank = top score)")
for r in results:
    rank = r["first_relevant_rank"] or "-"
    if r["type"] == "none":
        rank = f"{r['top_score']:.2f}" if r["top_score"] is not None else "-"
    margin = f"{r['margin']:+.3f}" if r["margin"] is not None else "   n/a"
    print(f"{r['id']:<5} {r['type']:<9} {rank:>4} {margin:>7}  {r['question'][:60]}")

summary = {}
answerable = [r for r in results if r["type"] != "none"]
for group in ["all", "specific", "broad", "keyword"]:
    rows = answerable if group == "all" else [
        r for r in results if r["type"] == group]
    if rows:
        summary[group] = {"n": len(rows), **{m: mean(rows, m) for m in
                                             ["hit@1", "recall@5", "mrr", "found", "margin", "spread"]}}
summary["median_latency_ms"] = round(statistics.median(latencies))

# ---- Threshold calibration: a good threshold sits above every "none" top score
# and below the best relevant score of (almost) every real question.
negatives = sorted(r["top_score"] for r in results if r["type"]
                   == "none" and r["top_score"] is not None)
positives = sorted(r["best_relevant_score"]
                   for r in answerable if r["best_relevant_score"] is not None)
if negatives and positives:
    summary["calibration"] = {
        "off_topic_top_scores": [round(s, 3) for s in negatives],
        "lowest_relevant_scores": [round(s, 3) for s in positives[:5]],
        "suggested_threshold": round((max(negatives) + positives[0]) / 2, 3)
        if max(negatives) < positives[0] else None,
    }

print(f"\nmode={args.mode}  k={args.k}  snapshot={len(snapshot) if snapshot else 'off'} papers")
for group, s in summary.items():
    if group in ("all", "specific", "broad", "keyword"):
        print(f"  {group:<9} n={s['n']:<3} hit@1={s['hit@1']}  recall@5={s['recall@5']}  "
              f"mrr={s['mrr']}  found@{args.k}={s['found']}  margin={s['margin']}  spread={s['spread']}")
print(f"  median latency {summary['median_latency_ms']} ms")
if "calibration" in summary:
    c = summary["calibration"]
    print(f"\n  threshold check ({args.mode} scores)")
    print(f"    off-topic questions, top score : {c['off_topic_top_scores']}")
    print(
        f"    real questions, 5 lowest scores of their best top-5 relevant paper : {c['lowest_relevant_scores']}")
    if c["suggested_threshold"] is not None:
        print(
            f"    clean gap. Suggested threshold: {c['suggested_threshold']}")
    else:
        print("    overlap: no threshold separates them perfectly. Pick one that rejects all off-topic "
              "questions and check which real questions it would drop.")

out = Path("evals/results") / \
    f"{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"mode": args.mode, "k": args.k, "evalset": args.evalset,
                           "summary": summary, "questions": results}, indent=2))
print(f"  saved {out}")
