"""
All settings and AWS clients in one place.

Locally these come from defaults below. In Lambda, the SAM template sets them
as environment variables, so the same code runs in both places unchanged.
"""
import json
import os

import boto3
from botocore.config import Config

REGION = os.getenv("AWS_REGION", "us-east-1")

# Storage
VECTOR_BUCKET = os.getenv("VECTOR_BUCKET", "vv-arxiv-vectors")
INDEX_NAME = os.getenv("VECTOR_INDEX", "abstracts-v1")
RAW_BUCKET = os.getenv("RAW_BUCKET")            # only set in Lambda

# Models
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024
EMBED_PRICE_PER_1K_TOKENS = 0.00002              # approx, Titan V2 on-demand
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
MAX_OUTPUT_TOKENS = 2048
# Reranker (Phase 2, Step 2). Cohere Rerank 3.5 is the only Bedrock reranker in us-east-1.
RERANK_MODEL_ID = os.getenv("RERANK_MODEL_ID", "cohere.rerank-v3-5:0")
RERANK_MODEL_ARN = f"arn:aws:bedrock:{REGION}::foundation-model/{RERANK_MODEL_ID}"
# how many vector hits the reranker re-scores
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "50"))

# arXiv
CATEGORIES = ["cs.CL", "cs.AI", "cs.IR"]
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "varshney2vaibhav@gmail.com")

# Retrieval
# cosine floor, used only if reranking fails
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.30"))
USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"
# rerank score floor, calibrated with the eval (Step 2)
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.15"))
CANDIDATES_K = 8
MAX_SOURCES = 5
NON_FILTERABLE_KEYS = ["title", "abstract", "authors", "abs_url"]

# Clients are created once when the module loads. In Lambda that means once per
# container, then reused by every warm invocation (faster than creating per call).
_aws_config = Config(retries={"max_attempts": 10, "mode": "adaptive"},
                     connect_timeout=5, read_timeout=60)
bedrock = boto3.client(
    "bedrock-runtime", region_name=REGION, config=_aws_config)
s3vectors = boto3.client("s3vectors", region_name=REGION, config=_aws_config)
s3 = boto3.client("s3", region_name=REGION, config=_aws_config)
# The Rerank API lives in bedrock-agent-runtime, not bedrock-runtime.
bedrock_agent = boto3.client(
    "bedrock-agent-runtime", region_name=REGION, config=_aws_config)


def log(event: str, **fields):
    """One JSON object per line. CloudWatch can search and graph these fields."""
    print(json.dumps({"event": event, **fields}, default=str))
