"""
Ask questions locally (same pipeline the Lambda runs).

    python scripts/ask.py "What's new in RAG evaluation?"
    python scripts/ask.py                       # interactive, 'quit' to exit
    python scripts/ask.py --graph               # print the graph as Mermaid
"""
import argparse

from arxiv_rag.graph import build_graph, run_query


def show(r: dict):
    print("\n" + "=" * 70 + f"\n{r['answer']}\n" + "=" * 70)
    if r["invalid_citations"]:
        print(
            f"\nWARNING: cites papers it was never given: {r['invalid_citations']}")
    key, floor = ("relevance", r["min_relevance"]) if r["reranked"] else (
        "similarity", r["min_similarity"])
    if r["sources"]:
        print(f"\nSources ({key} >= {floor}):")
        for s in r["sources"]:
            via = "+".join(s.get("found_by") or []) or "-"
            print(
                f"  [{s['id']}] {s[key]:.3f}  {s['title'][:62]}  ({'cited' if s['cited'] else 'not cited'}, via {via})")
            for p in s.get("passages") or []:
                print(f"      {p['score']:.2f}  {p['section'][:80]}")
    else:
        print("\nClosest papers (all below threshold):")
        for s in r["closest_below_threshold"]:
            print(f"  [{s['id']}] {s[key]:.3f}  {s['title'][:70]}")
    t, u = r["timings_ms"], r["usage"]
    line = (f"\nsearch {t.get('search', '-')} ms | rerank {t.get('rerank', '-')} ms | "
            f"chunks {t.get('chunks', '-')} ms | passage rerank {t.get('chunk_rerank', '-')} ms | "
            f"generate {t.get('generate', '-')} ms | total {t['total']} ms")
    if r.get("sources"):
        line += f"\nfull text used for {r.get('fulltext_papers', 0)} of {len(r['sources'])} papers"
    if u:
        line += f" | tokens in {u['input_tokens']:,} / out {u['output_tokens']:,}"
    print(line)


parser = argparse.ArgumentParser()
parser.add_argument("question", nargs="?")
parser.add_argument("--category")
parser.add_argument("--since", type=int)
parser.add_argument("--graph", action="store_true")
args = parser.parse_args()

if args.graph:
    print(build_graph().get_graph().draw_mermaid())
elif args.question:
    show(run_query(args.question, args.category, args.since))
else:
    print("Ask about recent arXiv papers. Type 'quit' to exit.")
    while (q := input("\n> ").strip()).lower() not in {"quit", "exit", "q"}:
        if q:
            show(run_query(q, args.category, args.since))
