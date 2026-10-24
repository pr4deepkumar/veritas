from __future__ import annotations

import math

import pytest

from veritas.local.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Q2 revenue of $95 million, up 9%!") == ["q2", "revenue", "95", "million", "up", "9"]


def test_chunks_without_query_terms_score_zero():
    index = BM25Index(["gross margin improved", "parental leave policy"])
    assert index.scores("parental leave").tolist()[0] == 0.0
    assert index.scores("unrelated words").tolist() == [0.0, 0.0]


def test_rare_terms_weigh_more_than_common_ones():
    index = BM25Index(["revenue grew", "revenue fell", "revenue flat", "margin grew"])
    scores = index.scores("margin revenue")
    assert scores[3] > scores[0]  # "margin" is in one chunk, "revenue" in three


def test_shorter_chunks_score_higher_for_the_same_match():
    index = BM25Index(["dividend", "dividend " + "filler " * 30])
    scores = index.scores("dividend")
    assert scores[0] > scores[1] > 0


def test_repeated_terms_help_with_diminishing_returns():
    index = BM25Index(["cash", "cash cash", "cash cash cash cash", "other text here"])
    one, two, four = index.scores("cash")[:3]
    assert one < two < four
    assert (four - two) < 2 * (two - one)


def test_matches_the_okapi_formula():
    texts = ["alpha beta", "alpha", "gamma"]
    index = BM25Index(texts, k1=1.5, b=0.75)
    n, df, tf, length, avgdl = 3, 2, 1, 2, (2 + 1 + 1) / 3
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * length / avgdl))
    assert index.scores("alpha")[0] == pytest.approx(expected, rel=1e-5)


def test_empty_index():
    assert BM25Index([]).scores("anything").size == 0
