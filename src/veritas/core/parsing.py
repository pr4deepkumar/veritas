"""PDF text extraction and chunking for the ingestion pipelines."""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Preferred split points, best first: paragraph, line, sentence, clause, word.
_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")
_SENTENCE_ENDS = ("\n", ". ", "? ", "! ")


@dataclass(frozen=True)
class Chunk:
    document: str
    page_num: int
    chunk_index: int  # position within the page
    text: str


def parse_pdf_to_page_json(source: Path | bytes, document: str | None = None) -> list[dict]:
    """Extract per-page text as ``{"document", "page_num", "contents"}`` records."""
    from pypdf import PdfReader

    if isinstance(source, Path):
        reader = PdfReader(str(source))
        name = document or source.stem
    else:
        if not document:
            raise ValueError("document name is required when parsing PDF bytes")
        reader = PdfReader(io.BytesIO(source))
        name = document

    # Many "encrypted" PDFs only restrict editing and open with an empty user password.
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError(f"{name}: PDF is password-protected")

    pages: list[dict] = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # pypdf raises a variety of errors on malformed content streams
            logger.warning("Could not extract text from %s page %d", name, page_num, exc_info=True)
            text = ""
        pages.append({"document": name, "page_num": page_num, "contents": text})
    return pages


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t\f\v\r]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters, cutting at the best
    natural boundary in the second half of each window and repeating roughly ``overlap``
    characters between consecutive chunks so a sentence cut at a boundary stays retrievable."""
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("max_chars must be positive and overlap in [0, max_chars)")
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in _BREAKS:
                cut = window.rfind(sep, max_chars // 2)
                if cut != -1:
                    end = start + cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = _next_start(text, start, end, overlap)
    return chunks


def _next_start(text: str, start: int, end: int, overlap: int) -> int:
    """Where the chunk after ``text[start:end]`` begins: about ``overlap`` characters back,
    moved to the start of that sentence so a chunk doesn't open mid-sentence ("which remains
    $0.40 per share."). If the sentence began more than twice ``overlap`` back, the start moves
    forward to the next word instead, so chunks never start mid-word."""
    target = max(end - overlap, start + 1)
    floor = max(end - 2 * overlap, start + 1)
    sentence = max(
        (found + len(sep) for sep in _SENTENCE_ENDS if (found := text.rfind(sep, floor, target)) != -1),
        default=-1,
    )
    if sentence != -1:
        return sentence
    space = text.find(" ", target, end)
    return space + 1 if space != -1 else target


def chunk_pages(pages: Iterable[dict], max_chars: int, overlap: int) -> list[Chunk]:
    """Chunk page records produced by :func:`parse_pdf_to_page_json`."""
    chunks: list[Chunk] = []
    for page in pages:
        document = str(page.get("document") or "unknown")
        page_num = int(page.get("page_num") or 0)
        for i, text in enumerate(chunk_text(str(page.get("contents") or ""), max_chars, overlap)):
            chunks.append(Chunk(document=document, page_num=page_num, chunk_index=i, text=text))
    return chunks
