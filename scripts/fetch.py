"""
Fetch recent arXiv abstracts to a local JSONL file.

    python scripts/fetch.py --days 3 --max 300
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from arxiv_rag.arxiv_client import fetch_papers

parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=3)
parser.add_argument("--max", type=int, default=500)
args = parser.parse_args()

papers = fetch_papers(days=args.days, max_papers=args.max)
Path("data").mkdir(exist_ok=True)
out = Path("data") / f"papers_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
out.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in papers), encoding="utf-8")
print(f"Saved {len(papers)} papers to {out}")
