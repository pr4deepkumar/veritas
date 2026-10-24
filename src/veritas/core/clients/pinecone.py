"""Pinecone vector store: index bootstrap, embeddings, retrieval and the document catalog."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

from pinecone import Pinecone, ServerlessSpec
from pinecone.errors import PineconeError

from veritas.core.errors import ConfigurationError
from veritas.core.indexing import DOC_HEAD_FIELD, document_key
from veritas.core.settings import settings

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 96  # multilingual-e5-large accepts at most 96 inputs per request
UPSERT_BATCH_SIZE = 100  # keeps each request under Pinecone's per-request size cap
DELETE_BATCH_SIZE = 1000
MAX_QUERY_TOP_K = 1000  # largest top_k Pinecone allows when metadata is included


class PineconeStore:
    def __init__(self, client: Pinecone | None = None) -> None:
        if client is None:
            if not settings.pinecone_api_key:
                raise ConfigurationError("PINECONE_API_KEY is required")
            client = Pinecone(api_key=settings.pinecone_api_key)
        self.pc = client
        if not self.pc.has_index(settings.pinecone_index):
            logger.info("Creating Pinecone index %s", settings.pinecone_index)
            self.pc.create_index(
                name=settings.pinecone_index,
                dimension=settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        self.index = self.pc.Index(name=settings.pinecone_index)

    # -- embeddings -------------------------------------------------------------------------

    def embed(self, texts: Sequence[str], input_type: str = "passage") -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            response = self.pc.inference.embed(
                model=settings.embedding_model,
                inputs=list(texts[start : start + EMBED_BATCH_SIZE]),
                parameters={"input_type": input_type, "truncate": "END"},
            )
            vectors.extend(list(item.values) for item in response)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], input_type="query")[0]

    # -- reads ------------------------------------------------------------------------------

    def search(self, question: str, top_k: int = 5, document: str | None = None) -> list[Any]:
        return self.query(self.embed_query(question), top_k=top_k, document=document)

    def query(self, vector: list[float], top_k: int = 5, document: str | None = None) -> list[Any]:
        """Nearest chunks, optionally restricted to one document by a metadata filter (so a
        selected document still gets ``top_k`` results instead of a post-filtered subset)."""
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter={"document": {"$eq": document}} if document else None,
        )
        return list(response.matches)

    def list_documents(self) -> list[str]:
        names: set[str] = set()
        try:
            token: str | None = None
            while True:
                page = self.index.fetch_by_metadata(
                    filter={DOC_HEAD_FIELD: {"$eq": True}}, limit=100, pagination_token=token
                )
                for vector in page.vectors.values():
                    name = (vector.metadata or {}).get("document")
                    if name:
                        names.add(str(name))
                token = page.pagination.next if page.pagination else None
                if not token:
                    break
        except PineconeError:
            logger.warning("Catalog lookup via fetch_by_metadata failed; sampling instead", exc_info=True)
        return sorted(names) if names else self._sample_document_names()

    def _sample_document_names(self) -> list[str]:
        """Fallback for indexes built before chunks carried ``doc_head``. Best effort: it sees
        at most MAX_QUERY_TOP_K vectors. Re-index to get the complete catalog."""
        stats = self.index.describe_index_stats()
        total = int(stats.total_vector_count or 0)
        if total == 0:
            return []
        dim = settings.embedding_dimension
        # Cosine similarity is undefined for an all-zero vector; probe with a uniform unit vector.
        probe = [1.0 / math.sqrt(dim)] * dim
        response = self.index.query(vector=probe, top_k=min(MAX_QUERY_TOP_K, total), include_metadata=True)
        return sorted(
            {
                str(m.metadata["document"])
                for m in response.matches
                if m.metadata and m.metadata.get("document")
            }
        )

    # -- writes -----------------------------------------------------------------------------

    def upsert(self, vectors: list[dict[str, Any]]) -> None:
        if vectors:
            self.index.upsert(vectors=vectors, batch_size=UPSERT_BATCH_SIZE, show_progress=False)

    def replace_document(
        self, document: str, vectors: list[dict[str, Any]], legacy_ids: Sequence[str] = ()
    ) -> int:
        """Upsert a document's chunks, then delete its vectors that are not part of this
        version (pages that shrank or vanished) plus any ``legacy_ids`` from the old
        page-per-vector layout. Returns the number of IDs submitted for deletion."""
        self.upsert(vectors)
        keep = {v["id"] for v in vectors}
        stale = [
            item.id
            for page in self.index.list(prefix=f"{document_key(document)}#")
            for item in page.vectors
            if item.id and item.id not in keep
        ]
        stale.extend(legacy_ids)
        for start in range(0, len(stale), DELETE_BATCH_SIZE):
            self.index.delete(ids=stale[start : start + DELETE_BATCH_SIZE])
        return len(stale)
