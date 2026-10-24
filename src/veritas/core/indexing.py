"""Store-agnostic indexing: chunk IDs, the catalog flag, and turning pages into vectors.

Both vector stores (the local hybrid index and Pinecone) implement :class:`IndexTarget`, so
the same code indexes documents into either.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import Any, Protocol

from veritas.core.parsing import Chunk, chunk_pages
from veritas.core.settings import settings

logger = logging.getLogger(__name__)

# Metadata flag set on the first chunk of every document, so the catalog can list one record
# per document instead of scanning every vector.
DOC_HEAD_FIELD = "doc_head"


def document_key(document: str) -> str:
    """Stable ASCII prefix for a document's vector IDs. Hashing keeps IDs short and
    unambiguous for any file name (Pinecone's ``list(prefix=...)`` needs ASCII)."""
    return hashlib.sha1(document.encode("utf-8")).hexdigest()[:16]


def chunk_id(document: str, page_num: int, chunk_index: int) -> str:
    return f"{document_key(document)}#p{page_num}#c{chunk_index}"


class IndexTarget(Protocol):
    def embed(self, texts: Sequence[str], input_type: str = "passage") -> Sequence[Sequence[float]]: ...

    def replace_document(
        self, document: str, vectors: list[dict[str, Any]], legacy_ids: Sequence[str] = ()
    ) -> int: ...


def build_vectors(store: IndexTarget, chunks: Sequence[Chunk]) -> list[dict[str, Any]]:
    if not chunks:
        return []
    embeddings = store.embed([chunk.text for chunk in chunks], input_type="passage")
    vectors = []
    for position, (chunk, values) in enumerate(zip(chunks, embeddings, strict=True)):
        metadata: dict[str, Any] = {
            "document": chunk.document,
            "page_num": chunk.page_num,
            "chunk_index": chunk.chunk_index,
            "content": chunk.text,
        }
        if position == 0:
            metadata[DOC_HEAD_FIELD] = True
        vectors.append(
            {
                "id": chunk_id(chunk.document, chunk.page_num, chunk.chunk_index),
                "values": values,
                "metadata": metadata,
            }
        )
    return vectors


def index_document(
    store: IndexTarget, pages: list[dict], fallback_name: str, *, remove_legacy_ids: bool = False
) -> int:
    """Replace one document's vectors; returns the number of chunks indexed."""
    document = str(pages[0].get("document") or fallback_name) if pages else fallback_name
    pages = [{**page, "document": document} for page in pages]
    vectors = build_vectors(store, chunk_pages(pages, settings.chunk_size, settings.chunk_overlap))
    # The original Pinecone indexer stored one vector per page as "<document>-<page>"; removing
    # those keeps a re-indexed document from being retrieved twice.
    legacy_ids = (
        [f"{document}-{int(page.get('page_num') or 0)}" for page in pages] if remove_legacy_ids else []
    )
    store.replace_document(document, vectors, legacy_ids=legacy_ids)
    if not vectors:
        logger.warning("%s has no extractable text (scanned PDF?); nothing was indexed", document)
    return len(vectors)
