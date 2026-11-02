"""
Phase 2, Step 4a: turn an arXiv paper's HTML into section-aware chunks.

Where the text comes from:
  arXiv converts most LaTeX submissions to HTML with a tool called LaTeXML and
  serves it at https://arxiv.org/html/<id>v<version>. That HTML keeps the paper's
  structure as tags we can rely on:

      <section class="ltx_section">      1  Introduction       (h2)
        <section class="ltx_subsection"> 1.1 Setup             (h3)
          <div class="ltx_para"><p class="ltx_p">...</p></div>
      <section class="ltx_appendix">     A  Extra results
      <section class="ltx_bibliography"> References            (dropped)
      <math alttext="x^2">               formulas -> their LaTeX source

  PDFs lose all of that (columns, headers, broken words), which is why we prefer HTML.
  A small share of papers have no HTML (conversion failed); they stay abstract-only.

Why "section-aware" chunking:
  Fixed-size chunks (every 500 tokens) cut through the middle of arguments and mix
  e.g. the end of Related Work with the start of Method. Here a chunk never crosses a
  section boundary, is built from whole paragraphs, and carries a header like
      "Paper title > 4 Experiments > 4.2 Ablations"
  so the embedding knows WHERE in the paper the text sits ("contextual chunk headers").

Politeness: this module fetches ONE page. Waiting 15 s between pages (arXiv's
robots.txt Crawl-delay) is the caller's job, see the worker in Step 4b.
"""
import re
import time

import requests

from .config import CONTACT_EMAIL, log

HTML_URL = "https://arxiv.org/html/{arxiv_id}v{version}"
USER_AGENT = f"arxiv-research-assistant/0.3 (mailto:{CONTACT_EMAIL})"

# ~350-400 tokens: room for a full argument, small enough to be specific
TARGET_CHARS = 1600
# a smaller tail gets merged into the previous chunk of the same section
MIN_CHARS = 300
MAX_CHUNKS = 80            # cap per paper so a 60-page appendix can't flood the index
# appendices are mostly proofs and extra tables: keep some, not all
MAX_APPENDIX_CHUNKS = 15
# big tables get truncated; the caption + first rows carry the result
MAX_TABLE_CHARS = 2000

# Sections that add noise, not knowledge.
SKIP_SECTION = re.compile(
    r"acknowledg|author contribution|checklist|reproducibility statement|ethics statement|"
    r"ethical consideration|broader impact|societal impact|impact statement|funding|"
    r"competing interest|references|bibliography|prompt template|use of large language model|"
    r"ai use statement|ai assistance|llm usage|declaration of",
    re.IGNORECASE)

SECTION_CLASSES = ("ltx_section", "ltx_subsection",
                   "ltx_subsubsection", "ltx_appendix", "ltx_paragraph")
DROP_CLASSES = ("ltx_bibliography", "ltx_note", "ltx_authors", "ltx_abstract", "ltx_tabular",
                "ltx_TOC", "ltx_page_footer", "ltx_dates", "ltx_classification", "ltx_keywords")


# ------------------------------------------------------------------ fetch
class NoHTML(Exception):
    """The paper has no usable HTML version (conversion failed, or withdrawn)."""


class RateLimited(Exception):
    """arXiv answered 429/503 on every try. Callers should stop and come back later."""


def fetch_html(arxiv_id: str, version: int = 1, max_retries: int = 3) -> str:
    url = HTML_URL.format(arxiv_id=arxiv_id, version=version)
    for attempt in range(max_retries):
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=30)
        if resp.status_code == 200:
            # never let requests guess (it falls back to Latin-1)
            resp.encoding = "utf-8"
            if "ltx_document" not in resp.text and "ltx_page_main" not in resp.text:
                raise NoHTML(f"{arxiv_id}: page is not LaTeXML output")
            return resp.text
        if resp.status_code == 404:
            raise NoHTML(f"{arxiv_id}: no HTML version (404)")
        if resp.status_code in (429, 503):
            # arXiv is telling us to slow down. Respect Retry-After, else back off hard.
            wait = int(resp.headers.get("Retry-After", 60 * (attempt + 1)))
            log("arxiv_html_backoff", arxiv_id=arxiv_id,
                status=resp.status_code, wait_s=wait)
            if attempt + 1 < max_retries:
                time.sleep(min(wait, 120))
            continue
        resp.raise_for_status()
    raise RateLimited(
        f"{arxiv_id}: arXiv kept refusing after {max_retries} tries")


# ------------------------------------------------------------------ parse
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _heading(section) -> str:
    h = section.find(["h2", "h3", "h4", "h5", "h6"], recursive=False)
    return _clean(h.get_text(" ")) if h else ""


def parse_html(html: str) -> dict:
    """Return {"title": str, "blocks": [{"path": [..headings..], "text": str, "kind": "para"|"caption"}]}."""
    from bs4 import BeautifulSoup        # imported here so the query Lambda never pays for it
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.find(class_="ltx_title_document")
    title = _clean(title_el.get_text(" ")) if title_el else ""

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    # Formulas: keep the LaTeX source (e.g. "\\mathcal{L}_{tube}") instead of MathML noise.
    for m in soup.find_all("math"):
        alt = m.get("alttext", "")
        m.replace_with(f" {alt} " if alt and len(alt) < 120 else " [math] ")
    # Tables: results live in the cells, so flatten each table into "a | b | c" rows and
    # attach them to the caption. (Must run before ltx_tabular is dropped below.)
    for fig in soup.find_all("figure", class_="ltx_table"):
        rows = []
        for tr in fig.find_all("tr"):
            cells = [_clean(td.get_text(" "))
                     for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        cap = fig.find("figcaption")
        if rows and cap is not None:
            body = "\n".join(rows)
            if len(body) > MAX_TABLE_CHARS:
                body = body[:MAX_TABLE_CHARS].rsplit(
                    "\n", 1)[0] + "\n[table truncated]"
            cap.append(soup.new_string("\n" + body))
    for cls in DROP_CLASSES:
        for tag in soup.find_all(class_=cls):
            tag.decompose()

    blocks = []
    for el in soup.find_all(lambda t: (t.name == "div" and "ltx_para" in (t.get("class") or []))
                            or (t.name == "figcaption")):
        if el.find_parent("figcaption") or (el.name == "div" and el.find_parent("figure")):
            continue                     # text inside figures is covered by the caption
        if el.name == "div" and el.find_parent("div", class_="ltx_para"):
            continue                     # nested para: the outer one already has this text
        path = [_heading(s) for s in reversed(el.find_parents("section"))
                if any(c in (s.get("class") or []) for c in SECTION_CLASSES)]
        path = [p for p in path if p]
        if not path or any(SKIP_SECTION.search(p) for p in path):
            continue
        if el.name == "figcaption" and "\n" in el.get_text():
            # table: keep row breaks so "model | score" lines stay readable
            lines = [_clean(l) for l in el.get_text(" ").split("\n")]
            text = "\n".join(l for l in lines if l)
            kind = "table"
        else:
            text = _clean(el.get_text(" "))
            kind = "caption" if el.name == "figcaption" else "para"
        appendix = any("ltx_appendix" in (s.get("class") or [])
                       for s in el.find_parents("section"))
        if len(text) >= 40:
            blocks.append({"path": path, "text": text,
                          "kind": kind, "appendix": appendix})
    return {"title": title, "blocks": blocks}


# ------------------------------------------------------------------ chunk
def _split_long(text: str, limit: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
    parts, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > limit:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def chunk_blocks(blocks: list[dict], target: int = TARGET_CHARS, min_chars: int = MIN_CHARS) -> list[dict]:
    """Pack whole paragraphs into chunks that never cross a (sub)section boundary."""
    chunks: list[dict] = []
    cur_path, cur_parts = None, []

    def appendix_seen(path) -> bool:
        """True once the appendix budget is spent (the current chunk counts as one)."""
        n = sum(1 for c in chunks if c.get("appendix")) + \
            (1 if cur_parts and cur_appendix[0] else 0)
        return n >= MAX_APPENDIX_CHUNKS
    cur_appendix = [False]

    def flush():
        if not cur_parts:
            return
        text = "\n\n".join(cur_parts)
        if chunks and len(text) < min_chars and chunks[-1]["path"] == cur_path:
            # tiny tail: merge into the previous chunk
            chunks[-1]["text"] += "\n\n" + text
        else:
            chunks.append({"path": cur_path, "text": text,
                          "appendix": cur_appendix[0]})

    for b in blocks:
        # section + subsection; deeper levels stay inside their parent
        path = b["path"][:2]
        if b.get("appendix") and appendix_seen(path):
            continue
        pieces = _split_long(b["text"], target) if len(
            b["text"]) > target else [b["text"]]
        for piece in pieces:
            size = sum(len(p) for p in cur_parts)
            if path != cur_path or (cur_parts and size + len(piece) > target):
                flush()
                cur_path, cur_parts = path, []
                cur_appendix[0] = bool(b.get("appendix"))
            cur_parts.append(piece)
    flush()
    return _merge_small(chunks, target, min_chars)[:MAX_CHUNKS]


def _merge_small(chunks: list[dict], target: int, min_chars: int) -> list[dict]:
    """Fold tiny chunks into a neighbour under the same top-level section.

    Papers often have one-paragraph subsections ("5.3 Rank-order stability", 237 chars).
    Alone they make weak vectors; joined with a sibling they keep their context. The
    merged chunk lists both headings, e.g. "5 Results > 5.2 ...; 5.3 ...".
    """
    out: list[dict] = []
    for c in chunks:
        prev = out[-1] if out else None
        small = len(c["text"]) < min_chars or (
            prev and len(prev["text"]) < min_chars)
        if (prev and small and prev["path"][:1] == c["path"][:1]
                and len(prev["text"]) + len(c["text"]) <= target * 1.25):
            if c["path"] != prev["path"] and len(c["path"]) > 1:
                prev["path"] = prev["path"] + [c["path"][-1]
                                               ] if len(prev["path"]) > 1 else prev["path"] + c["path"][1:]
                prev["path"] = prev["path"][:1] + ["; ".join(prev["path"][1:])]
            prev["text"] += "\n\n" + c["text"]
        else:
            out.append(dict(c))
    return out


def chunk_header(title: str, path: list[str]) -> str:
    return " > ".join([title] + path)


def paper_chunks(arxiv_id: str, html: str, title: str | None = None) -> list[dict]:
    """HTML -> list of chunk dicts ready to embed: {key, arxiv_id, chunk, section, text, embed_text}."""
    parsed = parse_html(html)
    title = title or parsed["title"]
    out = []
    for i, c in enumerate(chunk_blocks(parsed["blocks"])):
        section = " > ".join(c["path"])
        out.append({
            "key": f"{arxiv_id}#{i:03d}",
            "arxiv_id": arxiv_id,
            "chunk": i,
            "section": section,
            "text": c["text"],
            # What gets embedded: the header gives the vector context the raw paragraph lacks.
            "embed_text": f"{chunk_header(title, c['path'])}\n\n{c['text']}",
        })
    return out
