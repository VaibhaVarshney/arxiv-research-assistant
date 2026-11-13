"""
Answer eval: does the pipeline give the RIGHT detail, and if not, WHERE did it break?

    python scripts/eval_answers.py evals/answers_v2.jsonl                    # full pipeline, once
    python scripts/eval_answers.py evals/answers_v2.jsonl --repeat 3         # baseline (see below)
    python scripts/eval_answers.py evals/answers_v2.jsonl --modes abstracts,fulltext
    python scripts/eval_answers.py evals/answers_v2.jsonl --ids b01,b02 --show
    python scripts/eval_answers.py evals/answers_v2.jsonl --repeat 3 --nominate     # Phase 3 Step 2

Each question asks for a specific detail and lists strings a correct answer must ALL contain
(an item can be a list of alternatives, any one of which counts).
An answer that says the detail isn't there ("does not specify", ...) always fails, even if
it mentions the expected value in passing.

Every miss is put in exactly one bucket, in pipeline order:

    paper_miss    the gold paper wasn't in the top 5 sent to the LLM (level 1)
                    cut:  it ranked in the top 5 but scored under the relevance threshold
                    near: it was reranked but landed at rank 6-50 -> a wider net / nomination fixes it
                    far:  it wasn't among the 50 candidates at all -> a different query is needed
    passage_miss  the paper was there, but no passage holding the answer was (level 2)
    gen_miss      the answer was in front of the LLM and it still got it wrong

Passage buckets need "gold_chunks" in each row. scripts/eval_draft.py check --write adds them.

Snapshot: retrieval is restricted to evals/snapshot_ids.txt (same as the retrieval eval),
so newly indexed papers can't change the result overnight. --snapshot "" turns it off.

Repeats: the LLM isn't deterministic, so one run can flip a borderline question.
--repeat 3 runs everything 3 times and reports the mean and the range. Use that for any
number you'll compare against later.
"""
import argparse
import json
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import arxiv_rag.graph as graph
from arxiv_rag.config import LLM_MODEL_ID, MAX_SOURCES, MIN_RELEVANCE
from arxiv_rag.graph import run_query

REFUSAL = re.compile(r"(does|do) not (specify|mention|report|provide|state|include)|"
                     r"not (specified|mentioned|reported|provided|included|named)|"
                     r"cannot be answered|can't be answered|couldn't find|unnamed|unspecified|no information", re.I)
BUCKETS = ["pass", "paper_miss", "passage_miss", "gen_miss"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def passes(answer: str, expect: list) -> bool:
    """Every item must appear. An item can be a list of alternatives: ["62%", "62 percent"]."""
    if REFUSAL.search(answer):
        return False
    a = norm(answer)
    return all(any(norm(alt) in a for alt in (e if isinstance(e, list) else [e])) for e in expect)


def diagnose(row: dict, res: dict, mode: str) -> dict:
    ranking = [r["id"] for r in res.get("ranking", [])]
    rank = ranking.index(row["paper"]) + 1 if row["paper"] in ranking else None
    src = next((s for s in res["sources"] if s["id"] == row["paper"]), None)
    got_chunks = [p["chunk"] for p in (src or {}).get("passages", [])]
    gold = row.get("gold_chunks")
    ok = passes(res["answer"], row["expect"])

    if ok:
        bucket = "pass"
    elif src is None:
        bucket = "paper_miss"
    elif mode == "fulltext" and gold is not None and not row.get("in_abstract") \
            and not set(gold) & set(got_chunks):
        bucket = "passage_miss"
    elif mode == "abstracts" and not row.get("in_abstract"):
        bucket = "passage_miss"      # by design: the answer isn't in the abstract
    else:
        bucket = "gen_miss"
    return {"id": row["id"], "bucket": bucket, "pass": ok, "paper_rank": rank,
            "via": (src or {}).get("via"),
            "miss_kind": None if bucket != "paper_miss" else
            ("far" if rank is None else "cut" if rank <= MAX_SOURCES else "near"),
            "gold_chunks": gold, "got_chunks": got_chunks,
            "tokens_in": res["usage"].get("input_tokens", 0), "ms": res["timings_ms"]["total"],
            "answer": res["answer"]}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or "?"
    except OSError:
        return "?"


parser = argparse.ArgumentParser()
parser.add_argument("evalset")
parser.add_argument("--modes", default="fulltext", help="comma list: abstracts,fulltext")
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--ids", help="comma list of question ids to run")
parser.add_argument("--snapshot", default="evals/snapshot_ids.txt")
parser.add_argument("--show", action="store_true", help="print every failing answer")
parser.add_argument("--nominate", action="store_true",
                    help="turn on passage nomination (USE_NOMINATION) for this run")
args = parser.parse_args()

path = Path(args.evalset)
if not path.exists():
    raise SystemExit(f"{path} not found.")
# utf-8-sig: also accepts files that Windows editors saved with a BOM marker
rows = [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
if args.ids:
    keep = set(args.ids.split(","))
    rows = [r for r in rows if r["id"] in keep]
if not rows:
    raise SystemExit("No questions to run.")
if not any("gold_chunks" in r for r in rows):
    print("Note: no gold_chunks yet, so passage_miss can't be told apart from gen_miss.\n"
          "      Run: python scripts/eval_draft.py check", args.evalset, "--write\n")

if args.snapshot and Path(args.snapshot).exists():
    graph.ALLOWED_IDS = set(Path(args.snapshot).read_text().split())
if args.nominate:
    graph.USE_NOMINATION = True
modes = args.modes.split(",")
n_calls = len(rows) * len(modes) * args.repeat
print(f"{len(rows)} questions x {len(modes)} mode(s) x {args.repeat} repeat(s) = {n_calls} queries "
      f"(~{n_calls * 4 // 60 + 1} min). snapshot={'on' if graph.ALLOWED_IDS else 'off'}  "
      f"nomination={'on' if graph.USE_NOMINATION else 'off'}")

runs = {m: [] for m in modes}          # runs[mode][repeat] = [diagnosis per row]
for mode in modes:
    graph.USE_FULLTEXT = mode == "fulltext"
    for rep in range(args.repeat):
        out = []
        for i, r in enumerate(rows, 1):
            print(f"\r  {mode} run {rep + 1}/{args.repeat}: {i}/{len(rows)}", end="", flush=True)
            out.append(diagnose(r, run_query(r["question"], debug=True), mode))
        runs[mode].append(out)
    print()

# ------------------------------------------------------------------ per question
for mode in modes:
    print(f"\n=== {mode}")
    print(f"{'id':<5} {'paper':<11} {'anchor':<9} {'where':<8} {'rank':>4}  {'passed':<7} outcome")
    for i, r in enumerate(rows):
        diags = [run[i] for run in runs[mode]]
        n_pass = sum(d["pass"] for d in diags)
        buckets = Counter(d["bucket"] for d in diags)
        rank = diags[0]["paper_rank"]
        outcome = ", ".join(f"{b}" + (f" x{n}" if args.repeat > 1 else "")
                            for b, n in buckets.most_common())
        if diags[0]["miss_kind"]:
            outcome += f"  ({diags[0]['miss_kind']})"
        if any(d["via"] == "passage" for d in diags):
            outcome += "  [paper got in via passage]"
        print(f"{r['id']:<5} {r['paper']:<11} {r.get('anchor', '?'):<9} {r.get('where', '?'):<8} "
              f"{rank or '-':>4}  {n_pass}/{args.repeat:<5} {outcome}")
        if args.show and n_pass < args.repeat:
            bad = next(d for d in diags if not d["pass"])
            print(f"      gold {bad['gold_chunks']} got {bad['got_chunks']}: {bad['answer'][:300]!r}")

# ------------------------------------------------------------------ summary


def rate(diags: list[dict]) -> float:
    return sum(d["pass"] for d in diags) / len(diags)


summary = {}
for mode in modes:
    per_run = [rate(run) for run in runs[mode]]
    flat = [d for run in runs[mode] for d in run]
    buckets = Counter(d["bucket"] for d in flat)
    kinds = Counter(d["miss_kind"] for d in flat if d["miss_kind"])
    s = {"pass_rate_mean": round(statistics.mean(per_run), 3),
         "pass_rate_min": round(min(per_run), 3), "pass_rate_max": round(max(per_run), 3),
         "buckets": {b: round(buckets[b] / args.repeat, 1) for b in BUCKETS},
         "paper_miss_kinds": {k: round(kinds[k] / args.repeat, 1) for k in ("cut", "near", "far")},
         "avg_tokens_in": sum(d["tokens_in"] for d in flat) // len(flat),
         "median_ms": statistics.median(d["ms"] for d in flat),
         "via_passage": round(sum(d["via"] == "passage" for d in flat) / args.repeat, 1)}
    for key in ("anchor", "where", "paper"):
        groups = defaultdict(list)
        for run in runs[mode]:
            for r, d in zip(rows, run):
                groups[r.get(key, "?")].append(d)
        s[f"by_{key}"] = {g: {"n": len(ds) // args.repeat, "pass": round(rate(ds), 2)}
                          for g, ds in sorted(groups.items())}
    summary[mode] = s

    n = len(rows)
    rng = f" (range {s['pass_rate_min']:.0%}-{s['pass_rate_max']:.0%})" if args.repeat > 1 else ""
    print(f"\n{mode}: pass {s['pass_rate_mean']:.0%} of {n}{rng}   "
          f"avg input tokens {s['avg_tokens_in']:,}   median latency {s['median_ms']:.0f} ms")
    b = s["buckets"]
    k = s["paper_miss_kinds"]
    if graph.USE_NOMINATION:
        print(f"  gold paper got in via a passage: {s['via_passage']} per run")
    print(f"  misses per run: paper {b['paper_miss']} (cut {k['cut']}, near {k['near']}, far {k['far']}), "
          f"passage {b['passage_miss']}, generation {b['gen_miss']}")
    for key in ("anchor", "where", "paper"):
        print(f"  by {key:<7} " + "   ".join(f"{g} {v['pass']:.0%} (n={v['n']})"
                                           for g, v in s[f"by_{key}"].items()))

out_dir = Path("evals/results")
out_dir.mkdir(parents=True, exist_ok=True)
tag = "+".join(modes) + ("_nominate" if graph.USE_NOMINATION else "")
out_file = out_dir / f"answers_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
out_file.write_text(json.dumps({
    "evalset": str(path), "commit": git_commit(), "model": LLM_MODEL_ID,
    "flags": {"USE_RERANK": graph.USE_RERANK, "USE_HYBRID": graph.USE_HYBRID,
              "MIN_RELEVANCE": MIN_RELEVANCE, "MAX_SOURCES": MAX_SOURCES,
              "snapshot": bool(graph.ALLOWED_IDS), "USE_NOMINATION": graph.USE_NOMINATION,
              "MIN_NOMINATION": graph.MIN_NOMINATION},
    "repeat": args.repeat, "summary": summary,
    "runs": {m: [[{k: v for k, v in d.items()} for d in run] for run in runs[m]] for m in modes},
}, indent=2), encoding="utf-8")
print(f"\nSaved {out_file}")
