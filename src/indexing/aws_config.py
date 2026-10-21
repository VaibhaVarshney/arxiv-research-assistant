"""
Shared settings and helpers for indexing and search.

Everything AWS-related lives here so the other scripts stay short,
and so moving to Lambda later means changing one file.
"""
import json
import os

import boto3
from botocore.config import Config

# ---- Settings (override any of these with environment variables) ----
REGION = os.getenv("AWS_REGION", "us-east-1")
VECTOR_BUCKET = os.getenv("VECTOR_BUCKET", "vv-arxiv-vectors")
# Versioned index name: if you ever switch embedding models, create "abstracts-v2"
# instead of mixing vectors from two models in one index.
INDEX_NAME = os.getenv("VECTOR_INDEX", "abstracts-v1")

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024
PRICE_PER_1K_TOKENS = 0.00002   # Titan Text Embeddings V2, on-demand (approx.)

# Metadata we store but never filter on. Keeping big text fields non-filterable
# is required: filterable metadata is capped at 2 KB per vector.
NON_FILTERABLE_KEYS = ["title", "abstract", "authors", "abs_url"]

# Retries with backoff for when Bedrock throttles us.
_retry = Config(retries={"max_attempts": 10, "mode": "adaptive"})
bedrock = boto3.client("bedrock-runtime", region_name=REGION, config=_retry)
s3vectors = boto3.client("s3vectors", region_name=REGION, config=_retry)


def embed(text: str) -> tuple[list[float], int]:
    """Turn text into a 1024-number vector. Returns (vector, tokens_used)."""
    body = json.dumps({"inputText": text, "dimensions": EMBED_DIM, "normalize": True})
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    out = json.loads(resp["body"].read())
    return out["embedding"], out.get("inputTextTokenCount", 0)


def paper_to_text(paper: dict) -> str:
    """What we actually embed: title + abstract. The title is a strong summary signal."""
    return f"{paper['title']}\n\n{paper['abstract']}"
