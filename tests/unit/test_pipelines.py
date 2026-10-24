from __future__ import annotations

import json

import pytest

from pipelines import index_to_pinecone, ingest_from_s3
from pipelines.s3_utils import list_objects, relative_stem
from veritas.core.errors import ConfigurationError, PipelineError
from veritas.core.indexing import DOC_HEAD_FIELD, chunk_id
from veritas.core.settings import settings

MANIFEST = "parsed_json/_index_manifest.json"


class FakeStore:
    def __init__(self, fail_on=()) -> None:
        self.fail_on = set(fail_on)
        self.replaced: dict[str, tuple[list[dict], list[str]]] = {}

    def embed(self, texts, input_type="passage"):
        assert input_type == "passage"
        return [[float(len(text)), 1.0] for text in texts]

    def replace_document(self, document, vectors, legacy_ids=()):
        if document in self.fail_on:
            raise RuntimeError("upsert failed")
        self.replaced[document] = (vectors, list(legacy_ids))
        return 0


def put_parsed(s3, key: str, document: str, *pages: str) -> None:
    rows = [{"document": document, "page_num": n, "contents": text} for n, text in enumerate(pages, start=1)]
    s3.upload(key, json.dumps(rows).encode("utf-8"))


def test_listing_follows_pagination(s3):
    for n in range(5):
        s3.upload(f"input_pdfs/doc{n}.PDF", b"%PDF")
    s3.upload("input_pdfs/notes.txt", b"x")
    keys = [obj.key for obj in list_objects(s3, "test-bucket", "input_pdfs/", ".pdf")]
    assert keys == [f"input_pdfs/doc{n}.PDF" for n in range(5)]


def test_relative_stem_keeps_subfolders():
    assert relative_stem("input_pdfs/2026/q2.report.pdf", "input_pdfs/") == "2026/q2.report"
    assert relative_stem("elsewhere/x.pdf", "input_pdfs/") == "elsewhere/x"


def test_ingest_parses_new_pdfs_and_skips_up_to_date_ones(s3, make_pdf):
    s3.upload("input_pdfs/report.pdf", make_pdf("Revenue grew twelve percent."))
    s3.upload("input_pdfs/2026/report.pdf", make_pdf("A different report."))

    written = ingest_from_s3.run(s3_client=s3)

    assert sorted(written) == ["parsed_json/2026/report.json", "parsed_json/report.json"]
    pages = json.loads(s3.body("parsed_json/2026/report.json"))
    assert pages[0]["document"] == "2026/report"
    assert "different report" in pages[0]["contents"]

    assert ingest_from_s3.run(s3_client=s3) == []  # nothing changed

    s3.upload("input_pdfs/report.pdf", make_pdf("Revenue grew fifteen percent."))  # re-uploaded
    assert ingest_from_s3.run(s3_client=s3) == ["parsed_json/report.json"]
    assert len(ingest_from_s3.run(s3_client=s3, force=True)) == 2


def test_ingest_keeps_going_past_a_bad_pdf_then_fails(s3, make_pdf):
    s3.upload("input_pdfs/bad.pdf", b"this is not a pdf")
    s3.upload("input_pdfs/good.pdf", make_pdf("Fine."))

    with pytest.raises(PipelineError, match="bad.pdf"):
        ingest_from_s3.run(s3_client=s3)
    assert "parsed_json/good.json" in s3.objects


def test_pipelines_require_a_bucket(s3, monkeypatch):
    monkeypatch.setattr(settings, "s3_bucket", "")
    with pytest.raises(ConfigurationError):
        ingest_from_s3.run(s3_client=s3)
    with pytest.raises(ConfigurationError):
        index_to_pinecone.run(s3_client=s3, store=FakeStore())


def test_index_writes_chunk_vectors_with_catalog_flag_and_legacy_cleanup(s3):
    store = FakeStore()
    put_parsed(
        s3, "parsed_json/report.json", "report", "Revenue grew twelve percent.", "", "Income was ten million."
    )

    assert index_to_pinecone.run(s3_client=s3, store=store) == ["parsed_json/report.json"]

    vectors, legacy_ids = store.replaced["report"]
    assert [v["metadata"]["page_num"] for v in vectors] == [1, 3]
    assert vectors[0]["id"] == chunk_id("report", 1, 0)
    assert vectors[0]["metadata"]["content"] == "Revenue grew twelve percent."
    assert vectors[0]["metadata"][DOC_HEAD_FIELD] is True
    assert DOC_HEAD_FIELD not in vectors[1]["metadata"]
    assert legacy_ids == ["report-1", "report-2", "report-3"]


def test_long_pages_are_split_into_several_chunks(s3, monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 200)
    monkeypatch.setattr(settings, "chunk_overlap", 20)
    store = FakeStore()
    put_parsed(s3, "parsed_json/long.json", "long", "sentence number one. " * 40)

    index_to_pinecone.run(s3_client=s3, store=store)

    vectors, _ = store.replaced["long"]
    assert len(vectors) > 1
    assert [v["metadata"]["chunk_index"] for v in vectors] == list(range(len(vectors)))
    assert all(len(v["metadata"]["content"]) <= 200 for v in vectors)


def test_index_only_reprocesses_changed_documents(s3):
    store = FakeStore()
    put_parsed(s3, "parsed_json/a.json", "a", "alpha")
    put_parsed(s3, "parsed_json/b.json", "b", "beta")

    assert sorted(index_to_pinecone.run(s3_client=s3, store=store)) == [
        "parsed_json/a.json",
        "parsed_json/b.json",
    ]
    assert index_to_pinecone.run(s3_client=s3, store=store) == []

    put_parsed(s3, "parsed_json/a.json", "a", "alpha")  # re-parsed, identical text: same ETag
    put_parsed(s3, "parsed_json/b.json", "b", "beta, revised")
    assert index_to_pinecone.run(s3_client=s3, store=store) == ["parsed_json/b.json"]

    assert len(index_to_pinecone.run(s3_client=s3, store=store, force=True)) == 2
    assert set(json.loads(s3.body(MANIFEST))) == {"parsed_json/a.json", "parsed_json/b.json"}


def test_failed_documents_are_retried_on_the_next_run(s3):
    store = FakeStore(fail_on={"b"})
    put_parsed(s3, "parsed_json/a.json", "a", "alpha")
    put_parsed(s3, "parsed_json/b.json", "b", "beta")

    with pytest.raises(PipelineError, match="b.json"):
        index_to_pinecone.run(s3_client=s3, store=store)
    assert set(json.loads(s3.body(MANIFEST))) == {"parsed_json/a.json"}

    store.fail_on.clear()
    assert index_to_pinecone.run(s3_client=s3, store=store) == ["parsed_json/b.json"]


def test_nothing_to_do_never_connects_to_pinecone(s3, monkeypatch):
    def fail():
        raise AssertionError("Pinecone should not be contacted")

    monkeypatch.setattr(index_to_pinecone, "PineconeStore", fail)
    assert index_to_pinecone.run(s3_client=s3) == []
