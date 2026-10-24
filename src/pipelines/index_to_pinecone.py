"""Chunk, embed and upsert parsed documents into Pinecone, re-indexing only what changed.

A manifest (``<output prefix>_index_manifest.json``) records the ETag of each parsed JSON
file when it was last indexed, so unchanged documents are not re-embedded on every run and
a failed run resumes where it stopped.

Usage: ``python -m pipelines.index_to_pinecone [--force]``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import PurePosixPath
from typing import Any

from pipelines.s3_utils import (
    get_s3_client,
    list_objects,
    read_bytes,
    relative_stem,
    require_bucket,
    write_json,
)
from veritas.core.clients.pinecone import PineconeStore
from veritas.core.errors import PipelineError
from veritas.core.indexing import index_document
from veritas.core.logging_config import setup_logging
from veritas.core.settings import settings

logger = logging.getLogger(__name__)

MANIFEST_NAME = "_index_manifest.json"  # files under the output prefix starting with "_" are internal


def manifest_key() -> str:
    return f"{settings.s3_output_prefix}{MANIFEST_NAME}"


def is_internal(key: str) -> bool:
    return PurePosixPath(key).name.startswith("_")


def load_manifest(s3: Any, bucket: str) -> dict[str, str]:
    try:
        return json.loads(read_bytes(s3, bucket, manifest_key()))
    except s3.exceptions.NoSuchKey:
        return {}


def run(*, force: bool = False, s3_client: Any = None, store: PineconeStore | None = None) -> list[str]:
    """Index parsed documents that are new or changed since the last run; returns their keys."""
    bucket = require_bucket()
    s3 = s3_client or get_s3_client()
    documents = [
        obj
        for obj in list_objects(s3, bucket, settings.s3_output_prefix, ".json")
        if not is_internal(obj.key)
    ]
    known = {obj.key for obj in documents}
    manifest = {} if force else {k: v for k, v in load_manifest(s3, bucket).items() if k in known}
    pending = [obj for obj in documents if manifest.get(obj.key) != obj.etag]
    if not pending:
        logger.info("Index is up to date (%d documents)", len(documents))
        return []

    store = store or PineconeStore()
    indexed: list[str] = []
    failed: list[str] = []
    for obj in pending:
        try:
            pages = json.loads(read_bytes(s3, bucket, obj.key))
            count = index_document(
                store, pages, relative_stem(obj.key, settings.s3_output_prefix), remove_legacy_ids=True
            )
        except Exception:
            logger.exception("Failed to index s3://%s/%s", bucket, obj.key)
            failed.append(obj.key)
            continue
        manifest[obj.key] = obj.etag
        write_json(s3, bucket, manifest_key(), manifest)  # persist progress after every document
        indexed.append(obj.key)
        logger.info("Indexed %s: %d chunks", obj.key, count)

    logger.info(
        "Indexing: %d indexed, %d unchanged, %d failed",
        len(indexed),
        len(documents) - len(pending),
        len(failed),
    )
    if failed:
        raise PipelineError(f"{len(failed)} document(s) failed to index: {', '.join(failed[:10])}")
    return indexed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index parsed documents into Pinecone.")
    parser.add_argument("--force", action="store_true", help="re-index every document, ignoring the manifest")
    args = parser.parse_args(argv)
    setup_logging(settings.log_level)
    run(force=args.force)


if __name__ == "__main__":
    main()
