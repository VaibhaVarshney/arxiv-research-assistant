"""
Step 3: Ask questions, get answers grounded in arXiv papers, with citations.

Usage:
    python src/rag/ask.py "What's new in RAG evaluation?"
    python src/rag/ask.py "How do agents avoid bad tool calls?" --category cs.AI
    python src/rag/ask.py                      # interactive mode, type 'quit' to exit
    python src/rag/ask.py --graph              # print the pipeline as a Mermaid diagram
"""
import argparse
import time

from graph import MIN_SIMILARITY, build_graph

app = build_graph()


def show(result: dict, total_ms: int):
    print("\n" + "=" * 70)
    print(result["answer"])
    print("=" * 70)

    if result.get("invalid_citations"):
        print(f"\nWARNING: answer cites papers it was never given: {result['invalid_citations']}")

    sources = result.get("sources", [])
    cited = set(result.get("cited_ids", []))
    if sources:
        print(f"\nSources given to the model (similarity >= {MIN_SIMILARITY}):")
        for s in sources:
            mark = "cited" if s["key"] in cited else "not cited"
            print(f"  [{s['key']}] {1 - s['distance']:.3f}  {s['metadata']['title'][:70]}  ({mark})")
            print(f"           {s['metadata']['abs_url']}")
    else:
        print("\nClosest papers (all below the similarity threshold):")
        for c in result.get("candidates", [])[:3]:
            print(f"  [{c['key']}] {1 - c['distance']:.3f}  {c['metadata']['title'][:70]}")

    t = result.get("timings_ms", {})
    u = result.get("usage", {})
    line = f"\nretrieve {t.get('retrieve', '-')} ms | generate {t.get('generate', '-')} ms | total {total_ms} ms"
    if u:
        line += f" | tokens in {u['input_tokens']:,} / out {u['output_tokens']:,}"
        if u["stop_reason"] == "max_tokens":
            line += " | WARNING: answer was cut off"
    print(line)


def ask(question: str, category: str | None, since: int | None):
    t0 = time.perf_counter()
    result = app.invoke({"question": question, "category": category, "since": since})
    show(result, round((time.perf_counter() - t0) * 1000))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--category", help="e.g. cs.CL, cs.AI, cs.IR")
    parser.add_argument("--since", type=int, help="YYYYMMDD")
    parser.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    args = parser.parse_args()

    if args.graph:
        print(app.get_graph().draw_mermaid())
        return

    if args.question:
        ask(args.question, args.category, args.since)
        return

    print("Ask about recent arXiv papers. Type 'quit' to exit.")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            ask(q, args.category, args.since)


if __name__ == "__main__":
    main()
