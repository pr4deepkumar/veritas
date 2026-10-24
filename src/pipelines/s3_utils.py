"""S3 helpers shared by the ingestion and indexing pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from veritas.core.errors import ConfigurationError
from veritas.core.settings import settings


@dataclass(frozen=True)
class S3Object:
    key: str
    etag: str
    last_modified: datetime


def get_s3_client() -> Any:
    # Created on demand rather than at import, so importing these modules (e.g. while
    # Airflow parses the DAG) needs no AWS configuration.
    import boto3

    return boto3.client("s3", region_name=settings.aws_region)


def require_bucket() -> str:
    if not settings.s3_bucket:
        raise ConfigurationError("S3_BUCKET is required")
    return settings.s3_bucket


def list_objects(s3: Any, bucket: str, prefix: str, suffix: str) -> list[S3Object]:
    """Every object under ``prefix`` whose key ends with ``suffix`` (any case), across all
    pages of results; a single list call returns at most 1,000 keys."""
    objects: list[S3Object] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            if item["Key"].lower().endswith(suffix):
                objects.append(S3Object(item["Key"], item["ETag"].strip('"'), item["LastModified"]))
    return objects


def relative_stem(key: str, prefix: str) -> str:
    """``input_pdfs/2026/report.pdf`` -> ``2026/report``. Keeping sub-folders stops files with
    the same name in different folders from overwriting each other."""
    relative = key[len(prefix) :] if key.startswith(prefix) else key
    return str(PurePosixPath(relative).with_suffix(""))


def read_bytes(s3: Any, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def write_json(s3: Any, bucket: str, key: str, data: Any) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )
