"""Process-wide logging setup shared by the API and the ingestion pipelines."""

from __future__ import annotations

import logging
import sys

_NOISY_LOGGERS = ("botocore", "boto3", "urllib3", "pinecone", "httpx", "httpcore", "anthropic")


def setup_logging(level: str = "INFO") -> None:
    """Send log records to stdout. Leaves logging alone if the host (Airflow, pytest,
    ``uvicorn --log-config``) has already configured the root logger."""
    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
