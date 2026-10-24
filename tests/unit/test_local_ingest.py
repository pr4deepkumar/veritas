from __future__ import annotations

from veritas.local.ingest import discover, document_name, sync_folder
from veritas.local.store import LocalVectorStore


def test_document_names_keep_subfolders(tmp_path):
    assert document_name(tmp_path / "reports" / "q2.pdf", tmp_path) == "reports/q2"
    assert document_name(tmp_path.parent / "outside.md", tmp_path) == "outside"


def test_discovery_skips_hidden_and_unsupported_files(tmp_path):
    for name in ["a.md", "b.txt", "c.PDF", ".hidden.md", "notes.docx"]:
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in discover(tmp_path)] == ["a.md", "b.txt", "c.PDF"]
    assert discover(tmp_path / "missing") == []


def test_sync_indexes_skips_unchanged_reindexes_edits_and_removes_deleted(tmp_path, embedder, make_pdf):
    docs = tmp_path / "docs"
    (docs / "reports").mkdir(parents=True)
    (docs / "policy.md").write_text("# Leave\n\nParental leave lasts 16 weeks.")
    (docs / "notes.txt").write_text("The cafeteria opens at eight.")
    (docs / "reports" / "q2.pdf").write_bytes(make_pdf("Revenue grew nine percent.", "Margin improved."))
    store = LocalVectorStore(tmp_path / "index", embedder)

    first = sync_folder(store, docs)

    assert set(first.indexed) == {"policy", "notes", "reports/q2"}
    assert first.indexed["reports/q2"] == 2  # one chunk per page
    assert (tmp_path / "index" / "veritas_index.npz").exists()

    second = sync_folder(store, docs)
    assert second.indexed == {}
    assert sorted(second.unchanged) == ["notes", "policy", "reports/q2"]

    (docs / "notes.txt").write_text("The cafeteria now opens at seven.")
    (docs / "policy.md").unlink()
    third = sync_folder(store, docs)

    assert list(third.indexed) == ["notes"]
    assert third.removed == ["policy"]
    assert store.list_documents() == ["notes", "reports/q2"]
    assert "seven" in store.search("cafeteria")[0].metadata["content"]


def test_an_unreadable_file_doesnt_stop_the_others(tmp_path, embedder):
    (tmp_path / "broken.pdf").write_bytes(b"not really a pdf")
    (tmp_path / "fine.md").write_text("Some useful text here.")
    store = LocalVectorStore(tmp_path / "index", embedder)

    report = sync_folder(store, tmp_path)

    assert list(report.indexed) == ["fine"]
    assert list(report.failed) == [str(tmp_path / "broken.pdf")]
    assert store.list_documents() == ["fine"]


def test_force_reindexes_everything(tmp_path, embedder):
    (tmp_path / "a.md").write_text("alpha text")
    store = LocalVectorStore(tmp_path / "index", embedder)
    sync_folder(store, tmp_path)
    assert list(sync_folder(store, tmp_path, force=True).indexed) == ["a"]
