"""Index local files (PDF, Markdown, plain text) into the local hybrid store.

The index mirrors a documents folder: new or edited files are (re)indexed, unchanged files
are skipped by comparing SHA-256 content hashes, and documents whose file was deleted are
removed.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from veritas.core.indexing import index_document
from veritas.core.parsing import parse_pdf_to_page_json
from veritas.core.settings import settings
from veritas.local.embeddings import LocalEmbedder
from veritas.local.store import LocalVectorStore

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".pdf", ".md", ".markdown", ".txt")


@dataclass
class SyncReport:
    indexed: dict[str, int] = field(default_factory=dict)  # document -> chunks
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # file -> error


def open_default_store() -> LocalVectorStore:
    return LocalVectorStore(settings.local_index_dir, LocalEmbedder(settings.local_embedding_model))


def document_name(path: Path, root: Path) -> str:
    """``root/reports/q2.pdf`` -> ``reports/q2``; files outside ``root`` use their stem."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(path.name)
    return relative.with_suffix("").as_posix()


def discover(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and not path.name.startswith(".")
    )


def load_pages(path: Path, document: str) -> list[dict]:
    if path.suffix.lower() == ".pdf":
        return parse_pdf_to_page_json(path, document=document)
    return [
        {"document": document, "page_num": 1, "contents": path.read_text(encoding="utf-8", errors="replace")}
    ]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_file(store: LocalVectorStore, path: Path, root: Path, *, force: bool = False) -> int | None:
    """Index one file; returns its chunk count, or None if it is unchanged since last time."""
    document = document_name(path, root)
    digest = file_digest(path)
    if not force and store.source_hash(document) == digest:
        return None
    chunks = index_document(store, load_pages(path, document), document)
    store.set_source_hash(document, digest)
    logger.info("Indexed %s: %d chunks", document, chunks)
    return chunks


def sync_folder(store: LocalVectorStore, root: Path, *, force: bool = False) -> SyncReport:
    """Make the index match ``root``, then save it."""
    report = SyncReport()
    seen = set()
    try:
        for path in discover(root):
            document = document_name(path, root)
            seen.add(document)
            try:
                chunks = index_file(store, path, root, force=force)
            except Exception as exc:  # one unreadable file shouldn't stop the rest
                logger.exception("Failed to index %s", path)
                report.failed[str(path)] = str(exc)
                continue
            if chunks is None:
                report.unchanged.append(document)
            else:
                report.indexed[document] = chunks
        for document in sorted(set(store.sources) - seen):
            store.delete_document(document)
            report.removed.append(document)
    finally:
        store.save()  # keep whatever finished, even if something went wrong later
    return report
