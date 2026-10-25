import json

from .config import EMBED_DIM, EMBED_MODEL_ID, bedrock


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
    """What we embed: title + abstract."""
    return f"{paper['title']}\n\n{paper['abstract']}"
