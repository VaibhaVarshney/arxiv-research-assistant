"""
Step 4a check: fetch a few real arXiv HTML papers and show what the chunker makes of them.

    python scripts/probe_fulltext.py                       # 8 papers from the eval set
    python scripts/probe_fulltext.py 2610.31314 2610.24951 # specific papers
    python scripts/probe_fulltext.py --show 2610.31314 --out chunks.txt   # every chunk of one paper, saved as UTF-8

Pages are cached in data/html/, so re-running never re-downloads. New downloads wait
15 s apart (arXiv's robots.txt Crawl-delay), so 8 papers take about 2 minutes.
"""
import argparse
import statistics
import time
from pathlib import Path

from arxiv_rag.fulltext import NoHTML, fetch_html, paper_chunks, parse_html

CACHE = Path("data/html")
DELAY_S = 15
DEFAULT_IDS = ["2610.31314", "2610.24951", "2610.21261", "2610.38570", "2610.36918",
               "2610.24598", "2610.33180", "2610.26542"]

parser = argparse.ArgumentParser()
parser.add_argument("ids", nargs="*", default=DEFAULT_IDS)
parser.add_argument("--show", help="print every chunk of this (cached) paper")
parser.add_argument(
    "--out", help="with --show: write to this file (UTF-8) instead of the console")
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)


def get(arxiv_id: str, last_fetch: list) -> str | None:
    path = CACHE / f"{arxiv_id}.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    wait = DELAY_S - (time.time() - last_fetch[0])
    if wait > 0:
        print(f"  (waiting {wait:.0f} s for arXiv's crawl delay)")
        time.sleep(wait)
    last_fetch[0] = time.time()
    try:
        html = fetch_html(arxiv_id)
    except NoHTML as e:
        print(f"  no HTML: {e}")
        return None
    path.write_text(html, encoding="utf-8")
    return html


if args.show:
    html = (CACHE / f"{args.show}.html").read_text(encoding="utf-8")
    text = "".join(f"\n--- {c['key']}  [{c['section']}]  {len(c['text'])} chars\n{c['text']}\n"
                   for c in paper_chunks(args.show, html))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    raise SystemExit

last = [0.0]
rows = []
for pid in args.ids:
    print(f"\n=== {pid}")
    html = get(pid, last)
    if html is None:
        rows.append((pid, None))
        continue
    parsed = parse_html(html)
    chunks = paper_chunks(pid, html)
    sections = list(dict.fromkeys(" > ".join(
        b["path"][:2]) for b in parsed["blocks"]))
    print(f"  {parsed['title'][:90]}")
    print(
        f"  {len(parsed['blocks'])} blocks -> {len(chunks)} chunks across {len(sections)} sections")
    for s in sections:
        n = sum(c["section"] == s for c in chunks)
        print(f"    {n:>3}  {s[:90]}")
    rows.append((pid, chunks))

ok = [c for _, c in rows if c]
print("\n" + "=" * 60)
print(f"HTML available: {len(ok)}/{len(rows)}")
if ok:
    sizes = [len(ch["text"]) for c in ok for ch in c]
    words = sum(len(ch["text"].split()) for c in ok for ch in c)
    print(
        f"chunks per paper: median {statistics.median(len(c) for c in ok)}, max {max(len(c) for c in ok)}")
    print(
        f"chunk size (chars): median {statistics.median(sizes):.0f}, min {min(sizes)}, max {max(sizes)}")
    print(f"~embedding tokens per paper: {words * 1.3 / len(ok):,.0f}")
print("\nLook at one paper in full:  python scripts/probe_fulltext.py --show",
      (ok and rows[0][0]) or "<id>", "--out chunks.txt")
