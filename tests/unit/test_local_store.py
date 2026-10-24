from __future__ import annotations

import os

import pytest
from conftest import FakeEmbedder

from veritas.core.errors import ConfigurationError
from veritas.core.indexing import build_vectors, chunk_id
from veritas.core.parsing import Chunk
from veritas.local.store import LocalVectorStore, reciprocal_rank_fusion


def add(store: LocalVectorStore, document: str, *texts: str) -> None:
    chunks = [Chunk(document, page_num=1, chunk_index=i, text=text) for i, text in enumerate(texts)]
    store.replace_document(document, build_vectors(store, chunks))


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)
    assert [item for item, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_search_finds_the_relevant_chunk(tmp_path, embedder):
    store = LocalVectorStore(tmp_path, embedder)
    add(store, "report", "Revenue grew 9 percent to 95 million.", "Headcount reached 2150 employees.")
    add(store, "policy", "Parental leave lasts 16 weeks.")

    matches = store.search("How long is parental leave?", top_k=2)

    assert matches[0].metadata["document"] == "policy"
    assert matches[0].metadata["content"] == "Parental leave lasts 16 weeks."
    assert matches[0].score == 1.0  # ranked first by both searches
    assert len(matches) == 2
    assert embedder.calls[-1] == (["How long is parental leave?"], "query")


def test_keyword_matches_rescue_what_embeddings_rank_low(tmp_path):
    # Dense search alone prefers B; only A contains the exact code, so BM25 ranks it first,
    # and fusing the two rankings puts A on top.
    texts = {
        "A": "The XJ-9 sensor ships in March.",
        "B": "Our newest sensor ships next spring.",
        "C": "Unrelated catering menu.",
    }
    query = "When does the XJ-9 ship?"
    embedder = FakeEmbedder(
        overrides={texts["A"]: [0.6, 0.8], texts["B"]: [1.0, 0.0], texts["C"]: [0.0, 1.0], query: [1.0, 0.1]}
    )
    store = LocalVectorStore(tmp_path, embedder)
    for name, text in texts.items():
        add(store, name, text)

    assert [m.metadata["document"] for m in store.search(query, top_k=3)][:2] == ["A", "B"]


def test_search_can_be_restricted_to_one_document(tmp_path, embedder):
    store = LocalVectorStore(tmp_path, embedder)
    add(store, "report", "Revenue grew 9 percent.")
    add(store, "policy", "Revenue sharing is not offered.")

    matches = store.search("revenue", top_k=5, document="policy")

    assert {m.metadata["document"] for m in matches} == {"policy"}
    assert store.search("revenue", document="missing") == []


def test_replacing_a_document_drops_its_old_chunks_only(tmp_path, embedder):
    store = LocalVectorStore(tmp_path, embedder)
    add(store, "report", "old text one", "old text two")
    add(store, "policy", "policy text")

    add(store, "report", "new text")

    assert store.chunk_count() == 2
    assert store.list_documents() == ["policy", "report"]
    assert {m.id for m in store.search("text", top_k=5)} == {
        chunk_id("report", 1, 0),
        chunk_id("policy", 1, 0),
    }
    assert store.delete_document("report") == 1
    assert store.list_documents() == ["policy"]


def test_the_index_survives_a_restart(tmp_path, embedder):
    store = LocalVectorStore(tmp_path, embedder)
    add(store, "report", "Revenue grew 9 percent.")
    store.set_source_hash("report", "abc123")
    store.save()

    reopened = LocalVectorStore(tmp_path, embedder)

    assert reopened.list_documents() == ["report"]
    assert reopened.source_hash("report") == "abc123"
    assert reopened.search("revenue")[0].metadata["content"] == "Revenue grew 9 percent."
    assert not list(tmp_path.glob(".*.tmp"))  # the atomic write cleaned up after itself


def test_a_different_embedding_model_is_refused(tmp_path, embedder):
    store = LocalVectorStore(tmp_path, embedder)
    add(store, "report", "text")
    store.save()

    other = FakeEmbedder()
    other.model_name = "another-model"
    with pytest.raises(ConfigurationError, match="--force"):
        LocalVectorStore(tmp_path, other)


def test_changes_saved_by_another_process_are_picked_up(tmp_path, embedder):
    server = LocalVectorStore(tmp_path, embedder)
    cli = LocalVectorStore(tmp_path, embedder)
    add(cli, "report", "Revenue grew 9 percent.")
    cli.save()
    stat = (tmp_path / "veritas_index.npz").stat()
    os.utime(tmp_path / "veritas_index.npz", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert server.list_documents() == ["report"]


def test_an_empty_index_returns_nothing(tmp_path, embedder):
    store = LocalVectorStore(tmp_path, embedder)
    assert store.search("anything") == []
    assert store.list_documents() == []
