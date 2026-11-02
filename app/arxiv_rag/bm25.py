"""
Phase 2, Step 3: keyword search with BM25.

Why add keyword search when we have embeddings?
  Embeddings capture meaning but blur exact terms. "evaluation awareness",
  "RLVR", "TinyDialogues" or a model name are precise signals, and an embedding
  of a whole abstract can wash them out. BM25 scores papers by the exact words
  they share with the question, weighted by how rare each word is:

      score(paper) = sum over query words w of
                     idf(w) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))

      tf      how often w appears in this paper (more is better, with diminishing returns: k1)
      idf     rare words count more: "halting" says a lot, "model" says almost nothing
      len     long abstracts get a small penalty so they don't win just by having more words (b)

  The two retrievers fail differently, which is exactly why combining them works.

How it's stored:
  An inverted index: for every word, the list of papers containing it and how often.
  A query only touches the papers that share at least one word with it, so search
  is fast even in pure Python. The index is one gzipped JSON file in S3, rebuilt
  nightly by the Index Lambda and cached in memory by the Query Lambda.

Scale note: this comfortably handles tens of thousands of papers. Past ~100k you'd
move keyword search to a real engine (OpenSearch Serverless, or tantivy in a container).
"""
import gzip
import heapq
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

K1, B = 1.2, 0.75
TITLE_WEIGHT = 2          # title words count twice: titles are dense with key terms

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set("""
a an and are as at be been being but by can could did do does for from had has have how if in into
is it its it's may might more most no not of on or our over such than that the their them then there
these they this those through to under up via was we were what when where which while who why will
with within without would you your also both each between however only other same so some using use
used based new show shows shown propose proposed paper study approach method methods results result
""".split())


def _stem(w: str) -> str:
    """Very light stemming so 'memories'/'memory' and 'agents'/'agent' match."""
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def paper_tokens(title: str, abstract: str) -> list[str]:
    return tokenize(title) * TITLE_WEIGHT + tokenize(abstract)


# ------------------------------------------------------------------ build
def build_index(papers) -> dict:
    """papers: iterable of dicts with key, title, abstract, primary_category, published_date."""
    ids, lengths, cats, dates = [], [], [], []
    # term -> flat [doc, tf, doc, tf, ...] (compact JSON)
    postings: dict[str, list[int]] = {}
    for p in papers:
        doc = len(ids)
        ids.append(p["key"])
        cats.append(p.get("primary_category"))
        dates.append(int(p.get("published_date") or 0))
        tokens = paper_tokens(p.get("title", ""), p.get("abstract", ""))
        lengths.append(len(tokens))
        for term, tf in Counter(tokens).items():
            postings.setdefault(term, []).extend((doc, tf))
    return {"version": 1, "built_at": int(time.time()), "ids": ids, "lengths": lengths,
            "categories": cats, "dates": dates, "postings": postings}


def to_bytes(index: dict) -> bytes:
    return gzip.compress(json.dumps(index, separators=(",", ":")).encode("utf-8"))


def from_bytes(blob: bytes) -> dict:
    return json.loads(gzip.decompress(blob))


# ------------------------------------------------------------------ search
class BM25:
    def __init__(self, index: dict):
        self.ids = index["ids"]
        self.lengths = index["lengths"]
        self.categories = index["categories"]
        self.dates = index["dates"]
        self.postings = index["postings"]
        self.n = len(self.ids)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 1.0
        self.built_at = index.get("built_at")

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ())) // 2
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 50, category: str | None = None,
               since: int | None = None) -> list[tuple[str, float]]:
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf(term)
            for i in range(0, len(plist), 2):
                doc, tf = plist[i], plist[i + 1]
                norm = tf + K1 * (1 - B + B * self.lengths[doc] / self.avg_len)
                scores[doc] = scores.get(doc, 0.0) + idf * tf * (K1 + 1) / norm
        if category or since:
            scores = {d: s for d, s in scores.items()
                      if (not category or self.categories[d] == category)
                      and (not since or self.dates[d] >= since)}
        best = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
        return [(self.ids[d], s) for d, s in best]


# ------------------------------------------------------------------ load + cache
_cache = {"bm25": None, "etag": None, "checked": 0.0}
# warm Lambdas re-check S3 every 5 min, so a nightly rebuild is picked up
REFRESH_SECONDS = 300


def get_index(bucket: str | None, key: str, local_path: str) -> BM25 | None:
    """Return the BM25 index, loading from S3 (Lambda) or a local file (laptop). None if missing."""
    now = time.time()
    if _cache["bm25"] is not None and now - _cache["checked"] < REFRESH_SECONDS:
        return _cache["bm25"]
    _cache["checked"] = now
    if bucket:
        from botocore.exceptions import ClientError
        from .config import s3
        try:
            etag = s3.head_object(Bucket=bucket, Key=key)["ETag"]
            if etag != _cache["etag"]:
                blob = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                _cache.update(bm25=BM25(from_bytes(blob)), etag=etag)
        except ClientError:
            # keep serving the old index (or None) rather than fail
            return _cache["bm25"]
    else:
        path = Path(local_path)
        if path.exists():
            etag = str(path.stat().st_mtime)
            if etag != _cache["etag"]:
                _cache.update(bm25=BM25(from_bytes(
                    path.read_bytes())), etag=etag)
    return _cache["bm25"]
