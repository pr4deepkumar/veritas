from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from pinecone.errors import PineconeError

from veritas.core.clients.pinecone import EMBED_BATCH_SIZE, UPSERT_BATCH_SIZE, PineconeStore
from veritas.core.errors import ConfigurationError
from veritas.core.indexing import DOC_HEAD_FIELD, chunk_id, document_key
from veritas.core.settings import settings


def scored(id_: str, document: str | None) -> SimpleNamespace:
    return SimpleNamespace(id=id_, score=0.5, metadata={"document": document} if document else None)


class FakeIndex:
    def __init__(self, *, ids=(), matches=(), catalog_pages=(), catalog_error=None, total=0) -> None:
        self.ids = list(ids)
        self.matches = list(matches)
        self.catalog_pages = list(catalog_pages)
        self.catalog_error = catalog_error
        self.total = total
        self.queries: list[dict] = []
        self.upserts: list[dict] = []
        self.deleted: list[list[str]] = []
        self.catalog_calls: list[dict] = []

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(matches=self.matches)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def list(self, prefix: str):
        matching = [i for i in self.ids if i.startswith(prefix)]
        for start in range(0, len(matching), 2):  # two IDs per page
            yield SimpleNamespace(vectors=[SimpleNamespace(id=i) for i in matching[start : start + 2]])

    def delete(self, ids):
        self.deleted.append(list(ids))

    def fetch_by_metadata(self, **kwargs):
        self.catalog_calls.append(kwargs)
        if self.catalog_error:
            raise self.catalog_error
        return self.catalog_pages[len(self.catalog_calls) - 1]

    def describe_index_stats(self):
        return SimpleNamespace(total_vector_count=self.total)


class FakePinecone:
    def __init__(self, index: FakeIndex, exists: bool = True) -> None:
        self.index = index
        self.exists = exists
        self.created: list[dict] = []
        self.embed_calls: list[dict] = []
        self.inference = SimpleNamespace(embed=self._embed)

    def has_index(self, name: str) -> bool:
        return self.exists

    def create_index(self, **kwargs) -> None:
        self.created.append(kwargs)

    def Index(self, name: str) -> FakeIndex:
        return self.index

    def _embed(self, model, inputs, parameters):
        self.embed_calls.append({"model": model, "inputs": list(inputs), "parameters": parameters})
        return [SimpleNamespace(values=[float(len(text))]) for text in inputs]


def catalog_page(names, next_token=None):
    vectors = {f"id-{n}": SimpleNamespace(metadata={"document": n}) for n in names}
    return SimpleNamespace(
        vectors=vectors, pagination=SimpleNamespace(next=next_token) if next_token else None
    )


def make_store(index: FakeIndex | None = None, exists: bool = True) -> tuple[PineconeStore, FakePinecone]:
    client = FakePinecone(index or FakeIndex(), exists)
    return PineconeStore(client=client), client


def test_requires_an_api_key():
    with pytest.raises(ConfigurationError):
        PineconeStore()


def test_creates_the_index_only_when_missing():
    _, client = make_store(exists=False)
    assert client.created[0]["name"] == settings.pinecone_index
    assert client.created[0]["dimension"] == settings.embedding_dimension
    _, client = make_store(exists=True)
    assert client.created == []


def test_embeddings_are_batched_and_typed():
    store, client = make_store()
    vectors = store.embed([f"text {i}" for i in range(EMBED_BATCH_SIZE * 2 + 5)], input_type="passage")
    assert len(vectors) == EMBED_BATCH_SIZE * 2 + 5
    assert [len(call["inputs"]) for call in client.embed_calls] == [EMBED_BATCH_SIZE, EMBED_BATCH_SIZE, 5]
    assert client.embed_calls[0]["parameters"] == {"input_type": "passage", "truncate": "END"}

    store.embed_query("question")
    assert client.embed_calls[-1]["parameters"]["input_type"] == "query"


def test_query_filters_by_document_server_side():
    index = FakeIndex(matches=[scored("a", "Q2")])
    store, _ = make_store(index)

    store.query([0.1], top_k=4, document="Q2")
    store.query([0.1], top_k=4)

    assert index.queries[0]["filter"] == {"document": {"$eq": "Q2"}}
    assert index.queries[0]["top_k"] == 4
    assert index.queries[1]["filter"] is None


def test_catalog_pages_through_document_head_vectors():
    index = FakeIndex(catalog_pages=[catalog_page(["b", "a"], next_token="t1"), catalog_page(["c", "a"])])
    store, _ = make_store(index)

    assert store.list_documents() == ["a", "b", "c"]
    assert index.catalog_calls[0]["filter"] == {DOC_HEAD_FIELD: {"$eq": True}}
    assert index.catalog_calls[1]["pagination_token"] == "t1"


def test_catalog_falls_back_to_sampling_with_a_nonzero_probe():
    index = FakeIndex(
        catalog_error=PineconeError("unsupported"),
        matches=[scored("1", "b"), scored("2", "a"), scored("3", None)],
        total=5000,
    )
    store, _ = make_store(index)

    assert store.list_documents() == ["a", "b"]
    probe = index.queries[0]
    assert probe["top_k"] == 1000
    assert math.isclose(sum(v * v for v in probe["vector"]), 1.0)


def test_catalog_of_an_empty_index():
    store, _ = make_store(FakeIndex(catalog_pages=[catalog_page([])], total=0))
    assert store.list_documents() == []


def test_replace_document_upserts_in_batches_and_removes_stale_vectors():
    keep = chunk_id("report", 1, 0)
    stale = [chunk_id("report", 9, 0), chunk_id("report", 9, 1)]
    other_document = chunk_id("other", 1, 0)
    index = FakeIndex(ids=[keep, *stale, other_document])
    store, _ = make_store(index)

    deleted = store.replace_document(
        "report", [{"id": keep, "values": [0.1], "metadata": {}}], legacy_ids=["report-1"]
    )

    assert index.upserts[0]["batch_size"] == UPSERT_BATCH_SIZE
    assert index.deleted == [[*stale, "report-1"]]
    assert deleted == 3


def test_replace_document_without_stale_vectors_deletes_nothing():
    keep = chunk_id("report", 1, 0)
    store, _ = make_store(FakeIndex(ids=[keep]))
    assert store.replace_document("report", [{"id": keep, "values": [0.1], "metadata": {}}]) == 0
    assert store.index.deleted == []


def test_vector_ids_are_ascii_stable_and_scoped_to_the_document():
    key = document_key("Q2 Rapport financier – été")
    assert key.isascii() and len(key) == 16
    assert key == document_key("Q2 Rapport financier – été")
    assert chunk_id("Q2 Rapport financier – été", 3, 1) == f"{key}#p3#c1"
    assert document_key("report") != document_key("report#2")


def test_search_embeds_the_question_as_a_query_then_queries():
    index = FakeIndex(matches=[scored("a", "Q2")])
    store, client = make_store(index)
    assert [m.id for m in store.search("revenue?", top_k=3, document="Q2")] == ["a"]
    assert client.embed_calls[-1]["parameters"]["input_type"] == "query"
    assert index.queries[-1]["filter"] == {"document": {"$eq": "Q2"}}
