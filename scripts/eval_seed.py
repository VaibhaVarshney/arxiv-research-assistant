"""
Step 1a: draft a retrieval eval set from papers already in the index.

Picks random papers and writes a template where YOU write the question.
The paper you picked from becomes the "gold" answer for that question.

    python scripts/eval_seed.py --n 30 --out evals/retrieval_v1.jsonl

Rules for writing good questions (this is what makes the eval honest):
  - Ask what a researcher would ask, not "what does this paper say".
  - Don't copy distinctive words from the title. Paraphrase. Copying words
    makes keyword search (BM25) look better than it really is.
  - Mix in ~1/3 broad questions ("what's new in RAG evaluation?") and list
    every paper in the index you'd accept as a top answer under "relevant".
"""
import argparse
import json
import random
from pathlib import Path

from arxiv_rag.config import INDEX_NAME, VECTOR_BUCKET, s3vectors


def all_papers() -> list[dict]:
    papers, token = [], None
    while True:
        params = {"vectorBucketName": VECTOR_BUCKET, "indexName": INDEX_NAME,
                  "maxResults": 1000, "returnMetadata": True}
        if token:
            params["nextToken"] = token
        resp = s3vectors.list_vectors(**params)
        papers += resp.get("vectors", [])
        token = resp.get("nextToken")
        if not token:
            return papers


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=30)
parser.add_argument("--out", default="evals/retrieval_v1.jsonl")
parser.add_argument("--seed", type=int, default=7)
args = parser.parse_args()

out = Path(args.out)
if out.exists():
    raise SystemExit(f"{out} already exists. Delete it or pick another --out (don't lose your labels).")

papers = all_papers()
random.Random(args.seed).shuffle(papers)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for i, p in enumerate(papers[: args.n], 1):
        m = p["metadata"]
        f.write(json.dumps({
            "id": f"q{i:02d}",
            "type": "specific",            # change to "broad" for broad questions
            "question": "",                # <- you write this
            "relevant": [p["key"]],        # add more IDs for broad questions
            "_title": m["title"],          # keys starting with _ are notes, ignored by the eval
            "_abstract": m["abstract"][:400],
        }, ensure_ascii=False) + "\n")

print(f"Indexed papers: {len(papers)}. Wrote {min(args.n, len(papers))} templates to {out}")
print("Fill in every 'question' (delete rows you can't write a good question for), then run:")
print(f"  python scripts/eval_retrieval.py {out}")
