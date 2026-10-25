"""
Fetch recent abstracts from the arXiv API (same logic as Step 1, now reusable).
"""
import re
import time
from datetime import datetime, timedelta, timezone

import feedparser
import requests

from .config import CATEGORIES, CONTACT_EMAIL, log

API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 100
DELAY_SECONDS = 3.1          # arXiv: max 1 request every 3 seconds
MAX_RETRIES = 5
ID_PATTERN = re.compile(r"/abs/(.+?)(?:v(\d+))?$")


def build_query(days: int) -> str:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    cats = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    return f"({cats}) AND submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"


def _fetch_page(query: str, start: int):
    params = {"search_query": query, "start": start, "max_results": PAGE_SIZE,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    headers = {"User-Agent": f"arxiv-research-assistant/0.2 (mailto:{CONTACT_EMAIL})"}
    for attempt in range(MAX_RETRIES):
        resp = requests.get(API_URL, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            return feedparser.parse(resp.text)
        wait = DELAY_SECONDS * (2 ** attempt)
        log("arxiv_retry", status=resp.status_code, wait_s=wait, start=start)
        time.sleep(wait)
    raise RuntimeError(f"arXiv kept failing at start={start}")


def _clean(text: str) -> str:
    return " ".join(text.split())


def parse_entry(entry) -> dict:
    match = ID_PATTERN.search(entry.id)
    arxiv_id, version = match.group(1), int(match.group(2) or 1)
    pdf_url = next((l.href for l in entry.links if l.get("title") == "pdf"), None)
    return {
        "arxiv_id": arxiv_id,
        "version": version,
        "title": _clean(entry.title),
        "abstract": _clean(entry.summary),
        "authors": [a.name for a in entry.authors],
        "primary_category": entry.arxiv_primary_category["term"],
        "categories": [t.term for t in entry.tags],
        "published": entry.published,
        "updated": entry.updated,
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": pdf_url,
    }


def fetch_papers(days: int = 2, max_papers: int = 1000) -> list[dict]:
    query = build_query(days)
    papers: dict[str, dict] = {}
    start, total = 0, None
    while start < max_papers:
        feed = _fetch_page(query, start)
        if total is None:
            total = int(feed.feed.get("opensearch_totalresults", 0))
            log("arxiv_query", query=query, total_available=total)
        if not feed.entries:
            break
        for entry in feed.entries:
            paper = parse_entry(entry)
            papers[paper["arxiv_id"]] = paper
        start += PAGE_SIZE
        if start >= total:
            break
        time.sleep(DELAY_SECONDS)
    return list(papers.values())[:max_papers]
