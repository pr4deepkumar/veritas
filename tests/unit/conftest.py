"""Shared fixtures and fakes. Nothing in the unit tests touches the network."""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from veritas.core.settings import settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Pin settings so a developer's .env (real keys, other models) never leaks into tests,
    and local files go to a temporary folder instead of the repository's data/."""
    overrides = {
        "llm_provider": "local",
        "vector_store": "local",
        "local_index_dir": tmp_path / "index",
        "documents_dir": tmp_path / "documents",
        "anthropic_api_key": "",
        "pinecone_api_key": "",
        "chat_model": "claude-sonnet-5-5",
        "verifier_model": "",
        "answer_effort": "medium",
        "verifier_effort": "medium",
        "refusal_fallback": True,
        "embedding_dimension": 1024,
        "chunk_size": 1500,
        "chunk_overlap": 200,
        "s3_bucket": "test-bucket",
        "s3_input_prefix": "input_pdfs/",
        "s3_output_prefix": "parsed_json/",
        "web_dir": None,
    }
    for name, value in overrides.items():
        monkeypatch.setattr(settings, name, value)


@pytest.fixture
def make_pdf() -> Callable[..., bytes]:
    """Build a PDF with one page per argument (an empty string makes a blank page)."""

    def build(*pages: str) -> bytes:
        pdf = FPDF()
        pdf.set_font("Helvetica", size=12)
        for text in pages:
            pdf.add_page()
            if text:
                pdf.multi_cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        return bytes(pdf.output())

    return build


class FakeS3:
    """The slice of the boto3 S3 client the pipelines use, with paginated listings."""

    class exceptions:  # mirrors boto3's client.exceptions namespace
        class NoSuchKey(Exception):
            pass

    def __init__(self, page_size: int = 2) -> None:
        self.objects: dict[str, tuple[bytes, datetime]] = {}
        self.page_size = page_size
        self._clock = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def upload(self, key: str, body: bytes) -> None:
        self._clock += timedelta(seconds=1)  # every write is strictly newer than the last
        self.objects[key] = (body, self._clock)

    def body(self, key: str) -> bytes:
        return self.objects[key][0]

    def get_paginator(self, operation: str) -> FakeS3:
        assert operation == "list_objects_v2"
        return self

    def paginate(self, *, Bucket: str, Prefix: str):
        contents = [
            {"Key": key, "ETag": f'"{hashlib.md5(body).hexdigest()}"', "LastModified": modified}
            for key, (body, modified) in sorted(self.objects.items())
            if key.startswith(Prefix)
        ]
        if not contents:
            yield {"KeyCount": 0}
        for start in range(0, len(contents), self.page_size):
            yield {"Contents": contents[start : start + self.page_size]}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> None:
        self.upload(Key, Body)


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3()


class FakeEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words get similar embeddings.
    ``overrides`` pins the vector for specific texts, to control the dense ranking."""

    model_name = "fake-embedder"
    dimension = 64

    def __init__(self, overrides: dict[str, Sequence[float]] | None = None) -> None:
        self.overrides = overrides or {}
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: Sequence[str], input_type: str = "passage") -> np.ndarray:
        self.calls.append((list(texts), input_type))
        rows = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in zip(rows, texts, strict=True):
            if text in self.overrides:
                row[: len(self.overrides[text])] = self.overrides[text]
                continue
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                row[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.where(norms == 0, 1.0, norms)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
