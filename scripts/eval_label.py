"""
Step 1c: label broad questions, where many papers could be a correct answer.

You can't scan 3,000+ papers by hand, so this uses "pooling" (the trick TREC uses):
show you the top candidates, you mark the ones that genuinely answer the question.

    python scripts/eval_label.py evals/retrieval_v1.jsonl

For each row with "type": "broad" it prints the top 25 papers. Type the numbers of
the relevant ones (e.g. 1 4 7), press Enter to keep the current labels, or q to quit.
Your seed paper stays in "relevant" unless you remove it by editing the file.

Pooling bias: if the pool comes from one retriever, papers it misses can never be
labelled relevant, and the eval quietly favours that retriever. So from Step 3 on,
pool from BOTH and skip what's already been judged:

    python scripts/eval_label.py evals/retrieval_v1.jsonl --sources vector,bm25 \
        --skip evals/pools.jsonl --dump evals/pools_v2.jsonl
"""
import argparse
import json
from pathlib import Path

from arxiv_rag.retrieval import bm25_search, search

POOL = 25

parser = argparse.ArgumentParser()
parser.add_argument("evalset")
parser.add_argument(
    "--dump", help="write candidate pools to this file instead of asking interactively")
parser.add_argument("--sources", default="vector",
                    help="comma list: vector,bm25")
parser.add_argument(
    "--skip", help="earlier pools file: candidates already judged there are left out")
args = parser.parse_args()
sources = args.sources.split(",")

judged: dict[str, set[str]] = {}
if args.skip:
    for line in Path(args.skip).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            judged[r["id"]] = {p["id"] for p in r["pool"]}


def pool_for(row) -> list[dict]:
    """Top POOL from each source, interleaved, de-duplicated, minus already-judged papers."""
    lists = []
    if "vector" in sources:
        lists.append([h["key"] for h in search(row["question"], POOL)])
    if "bm25" in sources:
        lists.append([pid for pid, _ in bm25_search(row["question"], POOL)])
    seen, keys = judged.get(row["id"], set()) | set(row["relevant"]), []
    for i in range(POOL):
        for lst in lists:
            if i < len(lst) and lst[i] not in seen:
                seen.add(lst[i])
                keys.append(lst[i])
    if not keys:
        return []
    from arxiv_rag.retrieval import _fetch
    meta = _fetch(keys)
    return [{"key": k, "metadata": meta[k]["metadata"]} for k in keys if k in meta]


path = Path(args.evalset)
rows = [json.loads(l) for l in path.read_text(
    encoding="utf-8").splitlines() if l.strip()]

if args.dump:
    # Non-interactive: save each broad question's pool (with abstracts) so someone else can label it.
    with open(args.dump, "w", encoding="utf-8") as f:
        for row in rows:
            if row.get("type") == "broad" and row.get("question", "").strip():
                hits = pool_for(row)
                f.write(json.dumps({"id": row["id"], "question": row["question"], "pool": [
                    {"id": h["key"], "title": h["metadata"]["title"],
                     "abstract": h["metadata"]["abstract"][:600]} for h in hits]},
                    ensure_ascii=False) + "\n")
    raise SystemExit(f"Wrote pools to {args.dump}")


def save():
    path.write_text("".join(json.dumps(r, ensure_ascii=False) +
                    "\n" for r in rows), encoding="utf-8")


for row in rows:
    if row.get("type") != "broad" or not row.get("question", "").strip():
        continue
    hits = pool_for(row)
    if not hits:
        continue
    print(f"\n=== {row['id']}: {row['question']}")
    for i, h in enumerate(hits, 1):
        mark = "*" if h["key"] in row["relevant"] else " "
        print(f" {mark}{i:>2}. [{h['key']}] {h['metadata']['title'][:90]}")
    answer = input(
        "Relevant numbers (Enter = keep, q = quit): ").strip().lower()
    if answer == "q":
        break
    if answer:
        picked = [hits[int(n) - 1]["key"] for n in answer.replace(",", " ").split()
                  if n.isdigit() and 1 <= int(n) <= len(hits)]
        row["relevant"] = list(dict.fromkeys(row["relevant"] + picked))
        save()
        print(f"  saved {len(row['relevant'])} relevant")

save()
print("\nDone. Now run: python scripts/eval_retrieval.py", path)
