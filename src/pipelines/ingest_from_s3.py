"""Parse PDFs from S3 into per-page JSON, skipping PDFs whose parsed copy is up to date.

Usage: ``python -m pipelines.ingest_from_s3 [--force]``
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from pipelines.s3_utils import (
    get_s3_client,
    list_objects,
    read_bytes,
    relative_stem,
    require_bucket,
    write_json,
)
from veritas.core.errors import PipelineError
from veritas.core.logging_config import setup_logging
from veritas.core.parsing import parse_pdf_to_page_json
from veritas.core.settings import settings

logger = logging.getLogger(__name__)


def output_key(pdf_key: str) -> str:
    return f"{settings.s3_output_prefix}{relative_stem(pdf_key, settings.s3_input_prefix)}.json"


def run(*, force: bool = False, s3_client: Any = None) -> list[str]:
    """Parse new or changed PDFs and return the JSON keys written. A PDF that fails to parse
    doesn't stop the others; the run raises at the end so the scheduler can retry it."""
    bucket = require_bucket()
    s3 = s3_client or get_s3_client()
    pdfs = list_objects(s3, bucket, settings.s3_input_prefix, ".pdf")
    parsed = {obj.key: obj for obj in list_objects(s3, bucket, settings.s3_output_prefix, ".json")}

    written: list[str] = []
    failed: list[str] = []
    for pdf in pdfs:
        target = output_key(pdf.key)
        existing = parsed.get(target)
        if not force and existing is not None and existing.last_modified >= pdf.last_modified:
            continue
        try:
            document = relative_stem(pdf.key, settings.s3_input_prefix)
            pages = parse_pdf_to_page_json(read_bytes(s3, bucket, pdf.key), document=document)
            write_json(s3, bucket, target, pages)
        except Exception:
            logger.exception("Failed to parse s3://%s/%s", bucket, pdf.key)
            failed.append(pdf.key)
            continue
        written.append(target)
        logger.info("Parsed %s (%d pages) -> %s", pdf.key, len(pages), target)

    up_to_date = len(pdfs) - len(written) - len(failed)
    logger.info("Ingestion: %d parsed, %d up to date, %d failed", len(written), up_to_date, len(failed))
    if failed:
        raise PipelineError(f"{len(failed)} PDF(s) failed to parse: {', '.join(failed[:10])}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parse PDFs from S3 into per-page JSON.")
    parser.add_argument("--force", action="store_true", help="re-parse every PDF, even unchanged ones")
    args = parser.parse_args(argv)
    setup_logging(settings.log_level)
    run(force=args.force)


if __name__ == "__main__":
    main()
