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
# Phase 2, Step 4: full-text chunks
CHUNK_INDEX_NAME = os.getenv("CHUNK_INDEX", "chunks-v1")
FULLTEXT_QUEUE_URL = os.getenv(
    "FULLTEXT_QUEUE_URL")          # only set in Lambda
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
# how many candidates the reranker re-scores
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "50"))

# Two-level retrieval (Phase 2, Step 4c): papers first, then the best passages inside them
USE_FULLTEXT = os.getenv("USE_FULLTEXT", "true").lower() == "true"
# chunk hits pulled from the top papers
CHUNK_CANDIDATES = int(os.getenv("CHUNK_CANDIDATES", "40"))
# passages handed to the LLM in total
MAX_PASSAGES = int(os.getenv("MAX_PASSAGES", "8"))
# so one long paper can't take every slot
MAX_PASSAGES_PER_PAPER = 3
MIN_PASSAGE_RELEVANCE = float(os.getenv("MIN_PASSAGE_RELEVANCE", "0.10"))

# Passage nomination (Phase 3, Step 2): full-text chunks of ANY paper can pull it into the
# sources when its abstract doesn't match a detail question. See nominate.py.
USE_NOMINATION = os.getenv("USE_NOMINATION", "false").lower() == "true"   # off until the evals say so
# chunks pulled from the whole chunk index, then reranked
NOMINATION_CHUNKS = int(os.getenv("NOMINATION_CHUNKS", "50"))
# passage rerank score a paper needs to get in WITHOUT its abstract qualifying.
# Starting point only: calibrate with  eval_retrieval.py --mode passages
MIN_NOMINATION = float(os.getenv("MIN_NOMINATION", "0.40"))
# at most this many of the MAX_SOURCES papers may enter through a passage alone
MAX_NOMINATED = int(os.getenv("MAX_NOMINATED", "2"))

# Hybrid search (Phase 2, Step 3)
USE_HYBRID = os.getenv("USE_HYBRID", "false").lower(
) == "true"   # off by default: see Step 3 eval
# take this many from vector AND from BM25 before fusing
HYBRID_EACH = int(os.getenv("HYBRID_EACH", "50"))
# standard Reciprocal Rank Fusion constant
RRF_K = 60
BM25_KEY = "bm25/index.json.gz"                     # in RAW_BUCKET (Lambda)
BM25_LOCAL_PATH = os.getenv(
    "BM25_LOCAL_PATH", "data/bm25_index.json.gz")   # on your laptop

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
CHUNK_NON_FILTERABLE_KEYS = ["title", "section", "text"]

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
sqs = boto3.client("sqs", region_name=REGION, config=_aws_config)


def log(event: str, **fields):
    """One JSON object per line. CloudWatch can search and graph these fields."""
    print(json.dumps({"event": event, **fields}, default=str))
