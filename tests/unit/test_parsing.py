from __future__ import annotations

from itertools import pairwise

import pytest

from veritas.core.parsing import Chunk, chunk_pages, chunk_text, normalize_whitespace, parse_pdf_to_page_json


def test_parse_pdf_bytes_extracts_every_page(make_pdf):
    data = make_pdf("Revenue grew twelve percent.", "", "Operating income was ten million.")

    pages = parse_pdf_to_page_json(data, document="2026/report")

    assert [p["page_num"] for p in pages] == [1, 2, 3]
    assert {p["document"] for p in pages} == {"2026/report"}
    assert "Revenue grew twelve percent." in pages[0]["contents"]
    assert pages[1]["contents"].strip() == ""
    assert "ten million" in pages[2]["contents"]


def test_parse_pdf_path_names_the_document_after_the_file(make_pdf, tmp_path):
    path = tmp_path / "annual.pdf"
    path.write_bytes(make_pdf("Hello"))
    assert parse_pdf_to_page_json(path)[0]["document"] == "annual"


def test_parsing_bytes_requires_a_document_name(make_pdf):
    with pytest.raises(ValueError):
        parse_pdf_to_page_json(make_pdf("Hello"))


def test_normalize_whitespace():
    assert normalize_whitespace("a \t b\r\n\n\n\nc  \n d ") == "a b\n\nc\nd"


def test_short_text_is_a_single_chunk():
    assert chunk_text("  Hello   world  ", max_chars=100, overlap=10) == ["Hello world"]


def test_blank_text_has_no_chunks():
    assert chunk_text(" \n\t ", max_chars=100, overlap=10) == []


def test_long_text_chunks_fit_cover_every_word_and_overlap():
    words = [f"word{i}" for i in range(600)]
    chunks = chunk_text(" ".join(words), max_chars=500, overlap=80)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert set(" ".join(chunks).split()) == set(words)
    for previous, following in pairwise(chunks):
        first_word = following.split()[0]
        assert first_word in words  # never starts mid-word
        assert first_word in previous.split()  # repeats the boundary region


def test_the_overlap_starts_at_a_sentence_start():
    sentences = [f"Sentence number {i} says something about topic {i}." for i in range(40)]
    chunks = chunk_text(" ".join(sentences), max_chars=400, overlap=60)
    assert len(chunks) > 1
    for following in chunks[1:]:
        assert following.startswith("Sentence number"), following[:40]


def test_sentence_boundaries_are_preferred():
    chunks = chunk_text("This sentence has a handful of words in it. " * 30, max_chars=200, overlap=0)
    assert len(chunks) > 1
    assert all(chunk.endswith(".") for chunk in chunks)


def test_paragraph_boundaries_are_preferred():
    paragraph = " ".join(["alpha"] * 30)
    chunks = chunk_text(f"{paragraph}\n\n{paragraph}\n\n{paragraph}", max_chars=400, overlap=0)
    assert chunks == [f"{paragraph}\n\n{paragraph}", paragraph]


def test_unbroken_text_is_still_split():
    chunks = chunk_text("x" * 1000, max_chars=300, overlap=50)
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert len("".join(chunks)) >= 1000


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_parameters(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=max_chars, overlap=overlap)


def test_chunk_pages_numbers_chunks_within_each_page():
    pages = [
        {"document": "d", "page_num": 1, "contents": "lorem " * 120},
        {"document": "d", "page_num": 2, "contents": ""},
        {"document": "d", "page_num": 3, "contents": "short"},
    ]

    chunks = chunk_pages(pages, max_chars=300, overlap=50)

    page_one = [c for c in chunks if c.page_num == 1]
    assert len(page_one) > 1
    assert [c.chunk_index for c in page_one] == list(range(len(page_one)))
    assert {c.page_num for c in chunks} == {1, 3}
    assert chunks[-1] == Chunk(document="d", page_num=3, chunk_index=0, text="short")
