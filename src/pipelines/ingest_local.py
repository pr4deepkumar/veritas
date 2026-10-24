"""Index the local documents folder (PDF, Markdown, text) into the local hybrid store.

Usage: ``python -m pipelines.ingest_local [--dir PATH] [--force]``

Runs entirely on this machine: no API keys, no cloud services. A running API server picks up
the updated index on its next search.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from veritas.core.errors import PipelineError
from veritas.core.logging_config import setup_logging
from veritas.core.settings import settings
from veritas.local.ingest import open_default_store, sync_folder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index local documents into the local hybrid store.")
    parser.add_argument("--dir", type=Path, default=settings.documents_dir, help="documents folder to index")
    parser.add_argument("--force", action="store_true", help="re-index every file, even unchanged ones")
    args = parser.parse_args(argv)
    setup_logging(settings.log_level)

    store = open_default_store()
    report = sync_folder(store, args.dir, force=args.force)
    print(
        f"Indexed {len(report.indexed)} document(s) ({sum(report.indexed.values())} chunks), "
        f"{len(report.unchanged)} unchanged, {len(report.removed)} removed; "
        f"the index now holds {store.chunk_count()} chunks from {len(store.list_documents())} document(s)."
    )
    if report.failed:
        raise PipelineError(f"{len(report.failed)} file(s) failed: {', '.join(report.failed)}")


if __name__ == "__main__":
    main()
