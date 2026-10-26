"""
Lambda entry points. Each function in template.yaml points at one of these.

A handler receives (event, context):
  event   = the input (a test event, an S3 notification, an API request...)
  context = info about this invocation (request ID, time remaining, ...)
"""
import base64
import json
import time
from datetime import datetime, timezone
from urllib.parse import unquote_plus

from arxiv_rag.arxiv_client import fetch_papers
from arxiv_rag.config import RAW_BUCKET, log, s3
from arxiv_rag.indexer import index_papers


def fetch_handler(event, context):
    """Pull recent abstracts from arXiv and save them as one JSONL file in S3."""
    event = event or {}
    days = int(event.get("days", 2))          # 2-day window overlaps safely: re-runs are idempotent
    max_papers = int(event.get("max", 1000))

    papers = fetch_papers(days=days, max_papers=max_papers)
    if not papers:
        # Nothing new (e.g. a weekend). Writing an empty file would trigger a pointless index run.
        log("fetch_skipped", request_id=context.aws_request_id, reason="no papers", days=days)
        return {"bucket": RAW_BUCKET, "key": None, "papers": 0}

    now = datetime.now(timezone.utc)
    key = f"raw/{now:%Y/%m/%d}/papers_{now:%H%M%S}.jsonl"
    body = "\n".join(json.dumps(p, ensure_ascii=False) for p in papers)
    s3.put_object(Bucket=RAW_BUCKET, Key=key, Body=body.encode("utf-8"),
                  ContentType="application/x-ndjson")

    result = {"bucket": RAW_BUCKET, "key": key, "papers": len(papers)}
    log("fetch_done", request_id=context.aws_request_id, **result)
    return result


def _keys_from_event(event) -> list[tuple[str, str]]:
    """Accept a manual test event ({"key": ...}) or a real S3 notification (Step 5)."""
    if "Records" in event:
        return [(r["s3"]["bucket"]["name"], unquote_plus(r["s3"]["object"]["key"]))
                for r in event["Records"]]
    return [(event.get("bucket", RAW_BUCKET), event["key"])]


def index_handler(event, context):
    """Read a JSONL file from S3, embed every paper, upsert into S3 Vectors."""
    results = []
    for bucket, key in _keys_from_event(event):
        obj = s3.get_object(Bucket=bucket, Key=key)
        lines = obj["Body"].read().decode("utf-8").splitlines()
        papers = [json.loads(line) for line in lines if line.strip()]
        stats = index_papers(papers)
        log("index_done", request_id=context.aws_request_id, key=key, **stats)
        results.append({"key": key, **stats})
    return {"files": results}


MAX_QUESTION_CHARS = 500
ALLOWED_CATEGORIES = {"cs.CL", "cs.AI", "cs.IR", "cs.LG"}


def _response(status: int, payload: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload)}


def _parse_query_event(event) -> dict:
    """Accept a direct invoke ({"question": ...}) or an API Gateway HTTP request."""
    if "body" not in event:
        return event
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    return json.loads(body)


def _validate(params: dict) -> tuple[dict | None, str | None]:
    """Never trust input from the internet. Returns (clean_params, error_message)."""
    question = str(params.get("question") or "").strip()
    if not question:
        return None, "question is required"
    if len(question) > MAX_QUESTION_CHARS:
        return None, f"question must be under {MAX_QUESTION_CHARS} characters"

    category = params.get("category") or None
    if category and category not in ALLOWED_CATEGORIES:
        return None, f"category must be one of {sorted(ALLOWED_CATEGORIES)}"

    since = params.get("since") or None
    if since:
        since = str(since).replace("-", "")       # accept 2026-10-15 or 20261015
        if not (since.isdigit() and len(since) == 8):
            return None, "since must be a date like 2026-10-15"
        since = int(since)
    return {"question": question, "category": category, "since": since}, None


def query_handler(event, context):
    from arxiv_rag.graph import run_query   # heavy import, only this function needs it

    t0 = time.perf_counter()
    try:
        params = _parse_query_event(event)
    except (ValueError, UnicodeDecodeError):
        return _response(400, {"error": "request body must be JSON"})

    clean, error = _validate(params)
    if error:
        return _response(400, {"error": error})

    try:
        result = run_query(clean["question"], clean["category"], clean["since"])
    except Exception as e:
        # Log the details for you; give the public a generic message (no internals leaked).
        log("query_failed", request_id=context.aws_request_id, error=repr(e))
        return _response(500, {"error": "Something went wrong answering that. Please try again."})

    log("query_done", request_id=context.aws_request_id, question=clean["question"],
        n_sources=len(result["sources"]), invalid_citations=result["invalid_citations"],
        handler_ms=round((time.perf_counter() - t0) * 1000), **result["timings_ms"], **result["usage"])
    return _response(200, result)
