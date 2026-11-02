"""
Phase 2, Step 4b: the full-text pipeline.

    Index Lambda ──(new paper IDs)──► FullTextQueue (SQS) ──► FullText Lambda, every 15 min
                                          │ fails 3x                │ one paper at a time,
                                          ▼                         │ 15 s apart (arXiv's crawl delay)
                                     FullTextDLQ ──► email alarm    ▼
                                                     S3: fulltext/html/<id>v<n>.html.gz  (raw page, kept)
                                                     S3: fulltext/status/<id>.json       (done / no_html)
                                                     S3 Vectors: chunks-v1               (one vector per chunk)

Why a queue instead of a loop in the Index Lambda?
  ~285 papers x 15 s = ~70 min per night, and the backfill is ~3,700 papers (~16 h).
  A Lambda runs at most 15 min. The queue holds the backlog durably; each scheduled
  run takes what it can in ~13 min and stops. If a paper keeps failing it moves to a
  dead-letter queue (and you get an email) instead of blocking everything behind it.

Why keep the raw HTML in S3?
  So we never have to ask arXiv twice. When the chunker improves (it will), we re-chunk
  from S3 for free instead of re-crawling 3,700 pages at 15 s each.

Idempotent by design: a status file per paper means re-queueing (backfill, retries,
duplicate messages) never re-fetches or double-indexes.
"""
import gzip
import json
import time

from botocore.exceptions import ClientError

from .config import (CHUNK_INDEX_NAME, EMBED_PRICE_PER_1K_TOKENS, FULLTEXT_QUEUE_URL, RAW_BUCKET,
                     VECTOR_BUCKET, log, s3, s3vectors, sqs)
from .embeddings import embed
from .fulltext import NoHTML, RateLimited, fetch_html, paper_chunks

CRAWL_DELAY_S = 15            # arXiv robots.txt: Crawl-delay: 15
# leave time to finish the current paper before Lambda's timeout
STOP_WITH_MS_LEFT = 90_000
VISIBILITY_S = 300            # a message stays hidden this long while we work on it
STATUS_PREFIX = "fulltext/status/"
HTML_PREFIX = "fulltext/html/"
PUT_BATCH = 100


# ------------------------------------------------------------------ S3 helpers
def _status_key(arxiv_id: str) -> str:
    return f"{STATUS_PREFIX}{arxiv_id}.json"


def _html_key(arxiv_id: str, version: int) -> str:
    return f"{HTML_PREFIX}{arxiv_id}v{version}.html.gz"


def _get_json(key: str) -> dict | None:
    try:
        return json.loads(s3.get_object(Bucket=RAW_BUCKET, Key=key)["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def _put_status(arxiv_id: str, **fields):
    s3.put_object(Bucket=RAW_BUCKET, Key=_status_key(arxiv_id), ContentType="application/json",
                  Body=json.dumps({"arxiv_id": arxiv_id, "at": int(time.time()), **fields}).encode())


def _cached_html(arxiv_id: str, version: int) -> str | None:
    try:
        blob = s3.get_object(Bucket=RAW_BUCKET, Key=_html_key(
            arxiv_id, version))["Body"].read()
        return gzip.decompress(blob).decode("utf-8")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def _save_html(arxiv_id: str, version: int, html: str):
    s3.put_object(Bucket=RAW_BUCKET, Key=_html_key(arxiv_id, version), ContentType="application/gzip",
                  Body=gzip.compress(html.encode("utf-8")))


# ------------------------------------------------------------------ indexing
def index_chunks(paper: dict, chunks: list[dict], previous_count: int = 0) -> int:
    """Embed chunks and upsert them. Returns embedding tokens used."""
    tokens, batch = 0, []
    for c in chunks:
        vector, used = embed(c["embed_text"])
        tokens += used
        batch.append({"key": c["key"], "data": {"float32": vector}, "metadata": {
            "arxiv_id": paper["arxiv_id"],
            "chunk": c["chunk"],
            "published_date": int(paper.get("published_date") or 0),
            "primary_category": paper.get("primary_category") or "",
            "title": paper.get("title") or "",
            "section": c["section"],
            "text": c["text"],
        }})
        if len(batch) >= PUT_BATCH:
            s3vectors.put_vectors(
                vectorBucketName=VECTOR_BUCKET, indexName=CHUNK_INDEX_NAME, vectors=batch)
            batch = []
    if batch:
        s3vectors.put_vectors(vectorBucketName=VECTOR_BUCKET,
                              indexName=CHUNK_INDEX_NAME, vectors=batch)
    # A new version can have fewer chunks: remove the leftovers so stale text can't be retrieved.
    stale = [f"{paper['arxiv_id']}#{i:03d}" for i in range(
        len(chunks), previous_count)]
    for i in range(0, len(stale), PUT_BATCH):
        s3vectors.delete_vectors(vectorBucketName=VECTOR_BUCKET, indexName=CHUNK_INDEX_NAME,
                                 keys=stale[i:i + PUT_BATCH])
    return tokens


class _Politeness:
    """Guarantees >= CRAWL_DELAY_S between real requests to arXiv (cached pages don't count)."""

    def __init__(self):
        self.last = 0.0

    def fetch(self, arxiv_id: str, version: int) -> str:
        wait = CRAWL_DELAY_S - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        # fail fast: the queue is our retry
        return fetch_html(arxiv_id, version, max_retries=1)


def process_paper(paper: dict, polite: _Politeness) -> dict:
    """One paper end to end. Returns {"outcome": ..., ...}. Raises on retryable errors."""
    arxiv_id, version = paper["arxiv_id"], int(paper.get("version") or 1)
    status = _get_json(_status_key(arxiv_id))
    if status and status.get("status") in ("ok", "no_html") and int(status.get("version", 1)) >= version:
        return {"outcome": "skipped"}

    html, fetched = _cached_html(arxiv_id, version), False
    if html is None:
        try:
            html = polite.fetch(arxiv_id, version)
        except NoHTML as e:
            _put_status(arxiv_id, status="no_html",
                        version=version, reason=str(e))
            return {"outcome": "no_html"}
        _save_html(arxiv_id, version, html)
        fetched = True

    chunks = paper_chunks(arxiv_id, html, paper.get("title"))
    if not chunks:
        _put_status(arxiv_id, status="no_html", version=version,
                    reason="parsed to zero chunks")
        return {"outcome": "no_html", "fetched": fetched}
    tokens = index_chunks(paper, chunks, previous_count=int(
        (status or {}).get("chunks", 0)))
    _put_status(arxiv_id, status="ok", version=version,
                chunks=len(chunks), tokens=tokens)
    return {"outcome": "ok", "chunks": len(chunks), "tokens": tokens, "fetched": fetched}


# ------------------------------------------------------------------ the scheduled worker
def run_worker(time_left_ms) -> dict:
    """Drain the queue until it's empty or the Lambda is nearly out of time."""
    stats = {"ok": 0, "no_html": 0, "skipped": 0, "errors": 0, "chunks": 0, "tokens": 0,
             "fetched": 0, "rate_limited": False}
    polite = _Politeness()
    while time_left_ms() > STOP_WITH_MS_LEFT:
        resp = sqs.receive_message(QueueUrl=FULLTEXT_QUEUE_URL, MaxNumberOfMessages=1,
                                   WaitTimeSeconds=2, VisibilityTimeout=VISIBILITY_S)
        messages = resp.get("Messages", [])
        if not messages:
            break                                  # queue empty: finish early, costs ~nothing
        msg = messages[0]
        paper = json.loads(msg["Body"])
        try:
            result = process_paper(paper, polite)
        except RateLimited as e:
            # arXiv wants us to back off. Leave the message; try again next run (15 min later).
            log("fulltext_rate_limited", arxiv_id=paper.get(
                "arxiv_id"), error=str(e))
            stats["rate_limited"] = True
            break
        except Exception as e:
            # Leave the message: it reappears after VISIBILITY_S, and after 3 failures it goes
            # to the dead-letter queue (and the alarm emails you) instead of blocking the queue.
            log("fulltext_failed", arxiv_id=paper.get("arxiv_id"), error=repr(e))
            stats["errors"] += 1
            continue
        sqs.delete_message(QueueUrl=FULLTEXT_QUEUE_URL,
                           ReceiptHandle=msg["ReceiptHandle"])
        stats[result["outcome"]] += 1
        stats["chunks"] += result.get("chunks", 0)
        stats["tokens"] += result.get("tokens", 0)
        stats["fetched"] += int(result.get("fetched", False))

    attrs = sqs.get_queue_attributes(QueueUrl=FULLTEXT_QUEUE_URL,
                                     AttributeNames=["ApproximateNumberOfMessages"])["Attributes"]
    stats["queue_remaining"] = int(attrs.get("ApproximateNumberOfMessages", 0))
    stats["approx_cost_usd"] = round(
        stats["tokens"] / 1000 * EMBED_PRICE_PER_1K_TOKENS, 5)
    return stats


# ------------------------------------------------------------------ producers
def _message(p: dict) -> dict:
    # Papers from the nightly fetch carry "published" (ISO); papers from S3 Vectors carry published_date.
    date = p.get("published_date") or int(
        (p.get("published") or "0")[:10].replace("-", "") or 0)
    return {"arxiv_id": p["arxiv_id"], "version": int(p.get("version") or 1), "title": p.get("title", ""),
            "primary_category": p.get("primary_category", ""), "published_date": int(date)}


def enqueue(papers: list[dict]) -> int:
    """Queue papers for full text. Each needs arxiv_id, version, title, primary_category, published_date."""
    sent = 0
    for i in range(0, len(papers), 10):              # SendMessageBatch takes up to 10
        entries = [{"Id": str(j), "MessageBody": json.dumps(_message(p))}
                   for j, p in enumerate(papers[i:i + 10])]
        resp = sqs.send_message_batch(
            QueueUrl=FULLTEXT_QUEUE_URL, Entries=entries)
        sent += len(resp.get("Successful", []))
        for f in resp.get("Failed", []):
            log("fulltext_enqueue_failed", **f)
    return sent


def done_ids() -> set[str]:
    """Papers that already have a status file (ok or no_html)."""
    ids, token = set(), None
    while True:
        params = {"Bucket": RAW_BUCKET, "Prefix": STATUS_PREFIX}
        if token:
            params["ContinuationToken"] = token
        resp = s3.list_objects_v2(**params)
        ids.update(o["Key"][len(STATUS_PREFIX):-len(".json")]
                   for o in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            return ids
        token = resp["NextContinuationToken"]


def backfill() -> dict:
    """Queue every indexed paper that doesn't have full text yet. Safe to run repeatedly."""
    from .indexer import all_paper_metadata
    papers = [{"arxiv_id": p["key"], **p} for p in all_paper_metadata()]
    done = done_ids()
    todo = [p for p in papers if p["arxiv_id"] not in done]
    return {"indexed": len(papers), "already_done": len(done), "queued": enqueue(todo)}
