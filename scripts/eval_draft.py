"""
Phase 3, Step 1: tools for growing the answer eval (evals/answers_v2.jsonl).

Everything here reads from S3 Vectors, i.e. exactly the text the pipeline can see.
No arXiv requests, no crawl delay.

    python scripts/eval_draft.py pick 25              # random snapshot papers that have full text
    python scripts/eval_draft.py sheet 2610.12345     # worksheet: abstract, candidate facts, every chunk
    python scripts/eval_draft.py check evals/answers_v2.jsonl           # validate questions
    python scripts/eval_draft.py check evals/answers_v2.jsonl --write   # ...and save gold labels

Workflow
  1. pick:  choose ~5 papers that differ from each other (see "Picking papers" below).
  2. sheet: open data/drafts/<id>.md. "Candidate facts" lists body sentences with numbers
            that are NOT in the abstract. Read the paper on arXiv too; the sheet is a shortcut,
            not a substitute for knowing what the paper says.
  3. write: 5-7 questions per paper into evals/answers_v2.jsonl (format below).
  4. check: confirms every expected string really is in the indexed text, and records
            WHERE (gold_chunks). That label is what lets eval_answers.py say whether a miss
            happened at the paper level, the passage level, or in generation.

Row format
  {"id": "b03", "paper": "2610.12345", "question": "...", "expect": ["41.2"],
   "anchor": "named", "where": "table"}

  expect   ALL items must appear in the answer. An item can be a list of alternatives:
           ["DeepSeek-v4-Flash", ["100%", "100.0"]] = the model name AND either form of 100.

  anchor   named      question uses the paper's own name for something ("StateTrack-Bench")
           described  question only describes the topic, the way a real user would
  where    table | method | results | setup | appendix   (where the answer lives)
  evidence (optional) a short phrase from the passage that holds the answer, e.g.
           "primary automated judge". Needed when the expected string appears in many chunks
           (a model name, a common number), so gold_chunks points at the RIGHT passage.
  check --write adds:  gold_chunks (chunk numbers containing every expected string),
                       in_abstract (true = both pipeline modes can see it; a weak test)
"""
import argparse
import json
import random
import re
from pathlib import Path

from arxiv_rag.config import CHUNK_INDEX_NAME, INDEX_NAME, VECTOR_BUCKET, s3vectors

SNAPSHOT = Path("evals/snapshot_ids.txt")
DRAFTS = Path("data/drafts")
BATCH = 100                  # S3 Vectors get_vectors takes at most 100 keys per call
NUMBER = re.compile(r"\d+(?:\.\d+)?%?")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def get(index: str, keys: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(keys), BATCH):
        resp = s3vectors.get_vectors(vectorBucketName=VECTOR_BUCKET, indexName=index,
                                     keys=keys[i:i + BATCH], returnData=False, returnMetadata=True)
        out.update({v["key"]: v.get("metadata") or {} for v in resp.get("vectors", [])})
    return out


def abstracts(ids: list[str]) -> dict[str, dict]:
    return get(INDEX_NAME, ids)


def chunks(arxiv_id: str) -> list[dict]:
    """All chunks of one paper, in order. Keys are <id>#000, #001, ..."""
    found, start = [], 0
    while True:
        keys = [f"{arxiv_id}#{i:03d}" for i in range(start, start + BATCH)]
        got = get(CHUNK_INDEX_NAME, keys)
        found += [got[k] for k in keys if k in got]
        if len(got) < BATCH:
            return sorted(found, key=lambda m: int(m.get("chunk", 0)))
        start += BATCH


def has_fulltext(ids: list[str]) -> set[str]:
    got = get(CHUNK_INDEX_NAME, [f"{i}#000" for i in ids])
    return {k.split("#")[0] for k in got}


# ------------------------------------------------------------------ pick
def cmd_pick(args):
    ids = SNAPSHOT.read_text().split()
    used = {json.loads(l)["paper"] for f in Path("evals").glob("answers_*.jsonl")
            for l in f.read_text(encoding="utf-8-sig").splitlines() if l.strip()}
    random.Random(args.seed).shuffle(ids)
    sample = [i for i in ids if i not in used][:args.n * 2]   # ~7% have no full text
    meta, full = abstracts(sample), has_fulltext(sample)
    rows = [(i, meta[i]) for i in sample if i in full and i in meta][:args.n]
    print(f"{len(rows)} snapshot papers with full text (seed {args.seed}; skipping {len(used)} already used)\n")
    for i, m in rows:
        print(f"{i}  {m.get('primary_category', ''):<6}  {m.get('title', '')[:95]}")


# ------------------------------------------------------------------ sheet
def candidate_facts(abstract: str, paper_chunks: list[dict]) -> list[tuple[int, str, str]]:
    """Body sentences containing a number that the abstract doesn't mention."""
    in_abs = set(NUMBER.findall(abstract))
    out = []
    for c in paper_chunks:
        for sent in re.split(r"(?<=[.;])\s+|\n", c.get("text", "")):
            nums = [n for n in NUMBER.findall(sent) if n not in in_abs and len(n.rstrip("%")) >= 2]
            # skip years and citation-only lines
            nums = [n for n in nums if not re.fullmatch(r"(19|20)\d\d", n)]
            if nums and len(sent) > 25:
                out.append((int(c.get("chunk", 0)), c.get("section", ""), sent.strip()[:400]))
    return out


def cmd_sheet(args):
    DRAFTS.mkdir(parents=True, exist_ok=True)
    meta = abstracts(args.ids)
    for pid in args.ids:
        if pid not in meta:
            print(f"{pid}: not in the abstract index")
            continue
        m, cs = meta[pid], chunks(pid)
        if not cs:
            print(f"{pid}: no full-text chunks (no HTML, or not processed yet). Pick another.")
            continue
        facts = candidate_facts(m.get("abstract", ""), cs)
        lines = [f"# {m.get('title', '')}", "",
                 f"{pid} · {m.get('primary_category', '')} · {len(cs)} chunks · "
                 f"https://arxiv.org/html/{pid}", "",
                 "## Abstract", "", m.get("abstract", ""), "",
                 f"## Candidate facts ({len(facts)}): numbers in the body, not in the abstract", ""]
        lines += [f"- `#{n:03d}` *{sec[:60]}*: {s}" for n, sec, s in facts]
        lines += ["", "## All chunks", ""]
        for c in cs:
            lines += [f"### #{int(c.get('chunk', 0)):03d} · {c.get('section', '')}", "",
                      c.get("text", ""), ""]
        out = DRAFTS / f"{pid}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"{pid}: {len(cs)} chunks, {len(facts)} candidate facts -> {out}")


# ------------------------------------------------------------------ check
def cmd_check(args):
    path = Path(args.evalset)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    snapshot = set(SNAPSHOT.read_text().split()) if SNAPSHOT.exists() else None
    papers = sorted({r["paper"] for r in rows})
    meta = abstracts(papers)
    paper_chunks = {p: chunks(p) for p in papers}
    problems = 0
    ids = [r["id"] for r in rows]
    for dup in {i for i in ids if ids.count(i) > 1}:
        print(f"!! duplicate id {dup}")
        problems += 1

    print(f"{'id':<5} {'paper':<11} {'anchor':<9} {'where':<8} {'gold chunks':<14} notes")
    for r in rows:
        notes, block = [], []
        p = r["paper"]
        # each expect item is a string or a list of alternatives
        exp = [[norm(a) for a in (e if isinstance(e, list) else [e])] for e in r["expect"]]

        def has_all(text: str) -> bool:
            return all(any(a in text for a in alts) for alts in exp)

        if snapshot is not None and p not in snapshot:
            block.append("not in snapshot (eval_answers restricts to it, so this can never pass)")
        if p not in meta:
            block.append("paper not indexed")
        if not paper_chunks[p]:
            block.append("no full text")
        for k in ("anchor", "where"):
            if k not in r:
                block.append(f"missing '{k}'")
        abstract = norm(meta.get(p, {}).get("abstract", "") + " " + meta.get(p, {}).get("title", ""))
        in_abs = has_all(abstract)
        ev = norm(r.get("evidence", ""))     # optional phrase that pins the right passage
        gold = [int(c.get("chunk", 0)) for c in paper_chunks[p]
                if has_all(norm(c.get("text", ""))) and ev in norm(c.get("text", ""))]
        if not gold and not in_abs:
            block.append("expected text not found anywhere: check spelling/format (88 vs 88.0)"
                         + (" or the evidence phrase" if ev else ""))
        elif len(gold) > 3:
            notes.append(f"{len(gold)} chunks match: fine if all answer it, else add \"evidence\"")
        if in_abs:
            notes.append("in abstract (weak test)")
        for e in (a for alts in exp for a in alts):
            # Short bare numbers like "62" or "100" turn up in wrong answers by accident.
            if re.fullmatch(r"\d{1,2}|100", e):
                notes.append(f"'{e}' is too short to be distinctive (try '62%' or '62.0')")
        problems += bool(block)
        notes = [f"!! {b}" for b in block] + notes
        r["gold_chunks"], r["in_abstract"] = gold, in_abs
        print(f"{r['id']:<5} {p:<11} {r.get('anchor', '?'):<9} {r.get('where', '?'):<8} "
              f"{','.join(map(str, gold))[:14]:<14} {'; '.join(notes)}")

    by_paper = {p: sum(r['paper'] == p for r in rows) for p in papers}
    print(f"\n{len(rows)} questions, {len(papers)} papers: " +
          ", ".join(f"{p} x{n}" for p, n in by_paper.items()))
    for k in ("anchor", "where"):
        counts = {}
        for r in rows:
            counts[r.get(k, "?")] = counts.get(r.get(k, "?"), 0) + 1
        print(f"{k}: " + ", ".join(f"{v} {n}" for v, n in sorted(counts.items())))
    print(f"\n{problems} blocking problem(s).")
    if args.write:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
        print(f"Saved gold_chunks and in_abstract into {path}")


parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
sub = parser.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("pick")
p.add_argument("n", type=int, nargs="?", default=25)
p.add_argument("--seed", type=int, default=3)
p.set_defaults(fn=cmd_pick)
p = sub.add_parser("sheet")
p.add_argument("ids", nargs="+")
p.set_defaults(fn=cmd_sheet)
p = sub.add_parser("check")
p.add_argument("evalset")
p.add_argument("--write", action="store_true")
p.set_defaults(fn=cmd_check)
args = parser.parse_args()
args.fn(args)
