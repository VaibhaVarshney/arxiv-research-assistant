"""
Where does the time go? Per-stage latency with passage nomination off vs on.

    python scripts/time_stages.py evals/answers_v2.jsonl            # first 10 questions
    python scripts/time_stages.py evals/answers_v2.jsonl --n 20

Each question runs twice, once per setting, in alternating order (off-on, then on-off, ...)
so warm-up and network drift don't favour either side. The embedding cache is cleared
before every run, so neither side gets a free embedding.

Stages (ms): search + rerank = abstract level, nominate = passage level (runs in parallel
with the abstract level when on), retrieve = wall time of both, chunks / passage rerank =
evidence step, generate = LLM, total = end to end.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import arxiv_rag.graph as graph
from arxiv_rag.embeddings import embed_query
from arxiv_rag.graph import run_query

parser = argparse.ArgumentParser()
parser.add_argument("evalset")
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--snapshot", default="evals/snapshot_ids.txt")
args = parser.parse_args()

rows = [json.loads(l) for l in Path(args.evalset).read_text(encoding="utf-8-sig").splitlines()
        if l.strip()][:args.n]
if args.snapshot and Path(args.snapshot).exists():
    graph.ALLOWED_IDS = set(Path(args.snapshot).read_text().split())

stages = {False: defaultdict(list), True: defaultdict(list)}
for i, r in enumerate(rows):
    for flag in ((False, True) if i % 2 == 0 else (True, False)):
        graph.USE_NOMINATION = flag
        getattr(embed_query, "cache_clear", lambda: None)()
        res = run_query(r["question"])
        for k, v in res["timings_ms"].items():
            stages[flag][k].append(v)
        stages[flag]["input_tokens"].append(res["usage"].get("input_tokens", 0))
        stages[flag]["sources"].append(len(res["sources"]))
    print(f"\r  {i + 1}/{len(rows)}", end="", flush=True)
print()

keys = sorted(set(stages[False]) | set(stages[True]),
              key=lambda k: (k in ("total", "input_tokens", "sources"), k))
print(f"\n{'stage (median)':<16} {'off':>8} {'on':>8} {'diff':>8}")
for k in keys:
    off = statistics.median(stages[False][k]) if stages[False][k] else None
    on = statistics.median(stages[True][k]) if stages[True][k] else None
    diff = f"{on - off:+8.0f}" if off is not None and on is not None else f"{'':>8}"
    fmt = lambda v: f"{v:8.0f}" if v is not None else f"{'-':>8}"
    print(f"{k:<16} {fmt(off)} {fmt(on)} {diff}")
