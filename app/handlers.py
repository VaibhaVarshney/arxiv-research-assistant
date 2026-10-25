"""
Lambda entry points. Each function in template.yaml points at one of these.

A handler receives (event, context):
  event   = the input (a test event, an S3 notification, an API request...)
  context = info about this invocation (request ID, time remaining, ...)
"""
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


def _parse_query_event(event) -> dict:
    """Accept a direct invoke ({"question": ...}) or an API Gateway request (Step 6)."""
    if isinstance(event.get("body"), str):
        return json.loads(event["body"] or "{}")
    return event


def query_handler(event, context):
    from arxiv_rag.graph import run_query   # heavy import, only this function needs it

    t0 = time.perf_counter()
    params = _parse_query_event(event)
    question = (params.get("question") or "").strip()
    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": "question is required"})}

    result = run_query(question, params.get("category"), params.get("since"))
    log("query_done", request_id=context.aws_request_id, question=question,
        n_sources=len(result["sources"]), invalid_citations=result["invalid_citations"],
        handler_ms=round((time.perf_counter() - t0) * 1000), **result["timings_ms"], **result["usage"])
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result)}
