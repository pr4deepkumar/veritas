"""Self-hosted hybrid retrieval: exact cosine vector search plus BM25 keyword search, fused
with Reciprocal Rank Fusion and persisted to a single NumPy file.

Why hybrid: embeddings match meaning ("revenue rose" ~ "sales grew") but small models blur
exact tokens such as names, codes and figures, which BM25 matches precisely. Reciprocal Rank
Fusion combines the two rankings without having to calibrate their very different scores.

Brute-force search over normalized float32 vectors is exact and takes a few milliseconds
for tens of thousands of chunks; an approximate index (HNSW, IVF) would only pay off well
beyond that.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from veritas.core.errors import ConfigurationError
from veritas.local.bm25 import BM25Index

logger = logging.getLogger(__name__)

INDEX_FILE = "veritas_index.npz"
INDEX_FORMAT = 1
RRF_K = 60  # the constant from the original Reciprocal Rank Fusion paper (Cormack et al., 2009)
CANDIDATES_PER_RESULT = 4  # each ranking contributes this many candidates per requested result


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str], input_type: str = "passage") -> np.ndarray: ...


@dataclass(frozen=True)
class Match:
    id: str
    score: float  # fused rank score, scaled so 1.0 means ranked first by both searches
    metadata: dict[str, Any]


@dataclass
class _Snapshot:
    """Immutable view of the index. Writers build a new one; readers keep whichever they took."""

    vectors: np.ndarray  # (chunks, dimension) float32, rows L2-normalized
    records: list[dict[str, Any]]  # per chunk: id, document, page_num, chunk_index, content, ...
    bm25: BM25Index
    sources: dict[str, str] = field(default_factory=dict)  # document -> hash of the file indexed


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Fuse rankings: each item scores sum(1 / (k + rank)) over the rankings that contain it."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def _normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms == 0, 1.0, norms)


def _as_bytes(value: Any) -> np.ndarray:
    return np.frombuffer(json.dumps(value, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)


def _from_bytes(array: np.ndarray) -> Any:
    return json.loads(array.tobytes().decode("utf-8"))


def _snapshot(vectors: np.ndarray, records: list[dict[str, Any]], sources: dict[str, str]) -> _Snapshot:
    return _Snapshot(vectors, records, BM25Index([record["content"] for record in records]), sources)


class LocalVectorStore:
    def __init__(self, directory: Path | str, embedder: Embedder) -> None:
        self.directory = Path(directory)
        self.embedder = embedder
        self._lock = threading.RLock()
        self._loaded_mtime: int | None = None
        self._snapshot = self._read()

    @property
    def path(self) -> Path:
        return self.directory / INDEX_FILE

    # -- persistence ----------------------------------------------------------------------

    def _read(self) -> _Snapshot:
        if not self.path.exists():
            return _snapshot(np.zeros((0, 0), dtype=np.float32), [], {})
        with np.load(self.path, allow_pickle=False) as data:
            meta = _from_bytes(data["meta"])
            records = _from_bytes(data["records"])
            vectors = data["vectors"].astype(np.float32)
        if meta.get("embedding_model") != self.embedder.model_name:
            raise ConfigurationError(
                f"The index in {self.directory} was built with {meta.get('embedding_model')}, not "
                f"{self.embedder.model_name}. Rebuild it: python -m pipelines.ingest_local --force"
            )
        self._loaded_mtime = self.path.stat().st_mtime_ns
        return _snapshot(vectors, records, meta.get("sources", {}))

    def save(self) -> None:
        """Write the index atomically: a temporary file, then a rename over the old one."""
        with self._lock:
            snap = self._snapshot
            self.directory.mkdir(parents=True, exist_ok=True)
            meta = {
                "format": INDEX_FORMAT,
                "embedding_model": self.embedder.model_name,
                "sources": snap.sources,
            }
            temporary = self.directory / f".{INDEX_FILE}.tmp"
            with temporary.open("wb") as handle:
                np.savez(handle, vectors=snap.vectors, records=_as_bytes(snap.records), meta=_as_bytes(meta))
            os.replace(temporary, self.path)
            self._loaded_mtime = self.path.stat().st_mtime_ns

    def refresh(self) -> None:
        """Reload if another process (e.g. the ingestion CLI) rewrote the index file."""
        try:
            mtime = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if mtime != self._loaded_mtime:
            with self._lock:
                self._snapshot = self._read()
                logger.info("Reloaded the local index: %d chunks", len(self._snapshot.records))

    # -- reads ----------------------------------------------------------------------------

    def warm_up(self) -> None:
        warm = getattr(self.embedder, "warm_up", None)
        if warm:
            warm()

    def embed(self, texts: Sequence[str], input_type: str = "passage") -> np.ndarray:
        return self.embedder.embed(texts, input_type=input_type)

    def list_documents(self) -> list[str]:
        self.refresh()
        return sorted({record["document"] for record in self._snapshot.records})

    def chunk_count(self) -> int:
        return len(self._snapshot.records)

    def source_hash(self, document: str) -> str | None:
        return self._snapshot.sources.get(document)

    @property
    def sources(self) -> dict[str, str]:
        return dict(self._snapshot.sources)

    def search(self, question: str, top_k: int = 5, document: str | None = None) -> list[Match]:
        self.refresh()
        snap = self._snapshot  # one consistent view, even if a write lands meanwhile
        if not snap.records:
            return []
        if document is None:
            candidates = np.arange(len(snap.records))
        else:
            candidates = np.array(
                [i for i, r in enumerate(snap.records) if r["document"] == document], dtype=int
            )
            if candidates.size == 0:
                return []

        pool = max(top_k * CANDIDATES_PER_RESULT, 20)
        query = np.asarray(self.embedder.embed([question], input_type="query"), dtype=np.float32)[0]
        similarity = snap.vectors[candidates] @ query  # cosine: rows and query are unit length
        dense_ranking = candidates[np.argsort(-similarity, kind="stable")[:pool]]

        keyword = snap.bm25.scores(question)[candidates]
        order = np.argsort(-keyword, kind="stable")[:pool]
        keyword_ranking = candidates[order[keyword[order] > 0]]

        fused = reciprocal_rank_fusion([dense_ranking.tolist(), keyword_ranking.tolist()])[:top_k]
        best_possible = 2.0 / (RRF_K + 1)
        return [
            Match(
                id=snap.records[i]["id"],
                score=round(score / best_possible, 4),
                metadata=dict(snap.records[i]),
            )
            for i, score in fused
        ]

    # -- writes ---------------------------------------------------------------------------

    def replace_document(
        self, document: str, vectors: list[dict[str, Any]], legacy_ids: Sequence[str] = ()
    ) -> int:
        """Swap in a document's chunks (``legacy_ids`` is accepted for interface parity with
        Pinecone and ignored). Returns how many old chunks were removed."""
        with self._lock:
            snap = self._snapshot
            keep = [i for i, record in enumerate(snap.records) if record["document"] != document]
            records = [snap.records[i] for i in keep] + [{"id": v["id"], **v["metadata"]} for v in vectors]
            matrix = snap.vectors[keep]
            if vectors:
                new_rows = _normalize(np.asarray([v["values"] for v in vectors], dtype=np.float32))
                if matrix.shape[0] and matrix.shape[1] != new_rows.shape[1]:
                    raise ConfigurationError(
                        "Embedding size changed; rebuild with: python -m pipelines.ingest_local --force"
                    )
                matrix = np.vstack([matrix, new_rows]) if matrix.shape[0] else new_rows
            self._snapshot = _snapshot(matrix.astype(np.float32), records, snap.sources)
            return len(snap.records) - len(keep)

    def delete_document(self, document: str) -> int:
        with self._lock:
            removed = self.replace_document(document, [])
            self.forget_source(document)
            return removed

    def set_source_hash(self, document: str, digest: str) -> None:
        with self._lock:
            self._snapshot.sources[document] = digest

    def forget_source(self, document: str) -> None:
        with self._lock:
            self._snapshot.sources.pop(document, None)
