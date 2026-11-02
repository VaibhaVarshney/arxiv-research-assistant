"""
Step 4c: does full text make ANSWERS better? (The retrieval eval can't tell: ranking is unchanged.)

    python scripts/eval_answers.py evals/answers_v1.jsonl

Each question asks for a specific detail (a score, a count, a setting) and lists the
strings a correct answer must ALL contain. An answer that says the detail isn't there
("does not specify", "cannot be answered", ...) always fails, even if it happens to
mention the expected number in passing (e.g. "only reports a range of 62.4% to 100%"). Every question runs twice through the real
pipeline: abstracts only, then abstracts + full-text passages.

    pass     the answer contains an expected string
    found    the source paper was among the retrieved papers (if not, it's a retrieval miss)

Add your own: pick a paper with full text, read a results table or method detail that is
NOT in its abstract, and write the question + expected value. Details that ARE in the
abstract test nothing here, because both modes see the abstract.
"""
import argparse
import json
import re
from pathlib import Path

import arxiv_rag.graph as graph
from arxiv_rag.graph import run_query


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


REFUSAL = re.compile(r"(does|do) not (specify|mention|report|provide|state|include)|"
                     r"not (specified|mentioned|reported|provided|included|named)|"
                     r"cannot be answered|can't be answered|couldn't find|unnamed|unspecified|no information", re.I)


def passes(answer: str, expect: list[str]) -> bool:
    if REFUSAL.search(answer):
        return False
    a = norm(answer)
    return all(norm(e) in a for e in expect)


parser = argparse.ArgumentParser()
parser.add_argument("evalset")
parser.add_argument("--show", action="store_true", help="print every answer")
args = parser.parse_args()
path = Path(args.evalset)
if not path.exists():
    raise SystemExit(f"{path} not found.")
# utf-8-sig: also accepts files that Windows editors saved with a BOM marker
rows = [json.loads(l) for l in path.read_text(
    encoding="utf-8-sig").splitlines() if l.strip()]
if not rows:
    raise SystemExit(
        f"{path} has no questions in it ({path.stat().st_size} bytes). Re-save the file.")
print(f"{len(rows)} questions, each run twice (abstracts only, then with full text). ~1 min...")

results = {}
for mode, flag in (("abstracts", False), ("fulltext", True)):
    graph.USE_FULLTEXT = flag
    out = []
    for r in rows:
        res = run_query(r["question"])
        out.append({"id": r["id"], "pass": passes(res["answer"], r["expect"]),
                    "found": any(s["id"] == r["paper"] for s in res["sources"]),
                    "passages": sum(len(s.get("passages") or []) for s in res["sources"]),
                    "tokens_in": res["usage"].get("input_tokens", 0), "ms": res["timings_ms"]["total"],
                    "answer": res["answer"]})
    results[mode] = out

print(f"\n{'id':<5} {'found':>5}  {'abstracts':>9}  {'fulltext':>8}  question")
for i, r in enumerate(rows):
    a, f = results["abstracts"][i], results["fulltext"][i]
    def mark(x): return "PASS" if x["pass"] else "-"
    print(f"{r['id']:<5} {'yes' if f['found'] else 'NO':>5}  {mark(a):>9}  {mark(f):>8}  {r['question'][:70]}")
    if args.show:
        print(
            f"      abstracts: {a['answer'][:300]!r}\n      fulltext:  {f['answer'][:300]!r}")

for mode, out in results.items():
    n = len(out)
    print(f"\n{mode:<10} pass {sum(o['pass'] for o in out)}/{n}   "
          f"avg input tokens {sum(o['tokens_in'] for o in out) // n:,}   "
          f"median latency {sorted(o['ms'] for o in out)[n // 2]} ms")
