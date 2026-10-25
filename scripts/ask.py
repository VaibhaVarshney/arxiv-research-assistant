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
        print(f"\nWARNING: cites papers it was never given: {r['invalid_citations']}")
    if r["sources"]:
        print(f"\nSources (similarity >= {r['min_similarity']}):")
        for s in r["sources"]:
            print(f"  [{s['id']}] {s['similarity']:.3f}  {s['title'][:70]}  ({'cited' if s['cited'] else 'not cited'})")
    else:
        print("\nClosest papers (all below threshold):")
        for s in r["closest_below_threshold"]:
            print(f"  [{s['id']}] {s['similarity']:.3f}  {s['title'][:70]}")
    t, u = r["timings_ms"], r["usage"]
    line = f"\nretrieve {t.get('retrieve', '-')} ms | generate {t.get('generate', '-')} ms | total {t['total']} ms"
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
