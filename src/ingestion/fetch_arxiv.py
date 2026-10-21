"""
Step 1: Fetch recent arXiv abstracts and save them as JSONL.

Usage:
    python fetch_arxiv.py --days 3 --max 500

Output:
    data/papers_<YYYYMMDD>.jsonl   (one paper per line)
"""
import argparse
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests

API_URL = "https://export.arxiv.org/api/query"
CATEGORIES = ["cs.CL", "cs.AI", "cs.IR"]
PAGE_SIZE = 100          # papers per request
DELAY_SECONDS = 3.1      # arXiv asks for max 1 request every 3 seconds
MAX_RETRIES = 5

# Identify yourself. Put your real email here; it's polite and helps if arXiv needs to reach you.
HEADERS = {
    "User-Agent": "arxiv-research-assistant/0.1 (mailto:varshney2vaibhav@gmail.com)"}

ID_PATTERN = re.compile(r"/abs/(.+?)(?:v(\d+))?$")


def build_query(days: int) -> str:
    """One query for all categories, so cross-listed papers come back once."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    cats = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    return f"({cats}) AND submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"


def fetch_page(query: str, start: int):
    params = {
        "search_query": query,
        "start": start,
        "max_results": PAGE_SIZE,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    for attempt in range(MAX_RETRIES):
        resp = requests.get(API_URL, params=params,
                            headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            return feedparser.parse(resp.text)
        # 429 / 503 = arXiv is busy. Back off exponentially instead of hammering it.
        wait = DELAY_SECONDS * (2 ** attempt)
        print(f"  HTTP {resp.status_code}, retrying in {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError(f"arXiv kept failing at start={start}")


def clean(text: str) -> str:
    """arXiv titles/abstracts contain hard line breaks. Collapse all whitespace."""
    return " ".join(text.split())


def parse_entry(entry) -> dict:
    match = ID_PATTERN.search(entry.id)
    arxiv_id, version = match.group(1), int(match.group(2) or 1)
    pdf_url = next(
        (l.href for l in entry.links if l.get("title") == "pdf"), None)
    return {
        "arxiv_id": arxiv_id,                       # versionless: our stable key
        "version": version,
        "title": clean(entry.title),
        "abstract": clean(entry.summary),
        "authors": [a.name for a in entry.authors],
        "primary_category": entry.arxiv_primary_category["term"],
        "categories": [t.term for t in entry.tags],
        "published": entry.published,               # first version date
        "updated": entry.updated,                   # latest version date
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": pdf_url,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3,
                        help="look back this many days")
    parser.add_argument("--max", type=int, default=500,
                        help="cap on papers fetched")
    args = parser.parse_args()

    query = build_query(args.days)
    print(f"Query: {query}")

    papers: dict[str, dict] = {}   # keyed by arxiv_id, so duplicates collapse
    start, total = 0, None

    while start < args.max:
        feed = fetch_page(query, start)
        if total is None:
            total = int(feed.feed.get("opensearch_totalresults", 0))
            print(f"arXiv reports {total} matching papers")
        if not feed.entries:
            break
        for entry in feed.entries:
            paper = parse_entry(entry)
            papers[paper["arxiv_id"]] = paper
        start += PAGE_SIZE
        print(f"  fetched {len(papers)} so far")
        if start >= total:
            break
        time.sleep(DELAY_SECONDS)

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"papers_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for p in papers.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved {len(papers)} papers to {out_path}")


if __name__ == "__main__":
    main()
