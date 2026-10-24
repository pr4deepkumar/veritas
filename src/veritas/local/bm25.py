"""Okapi BM25 keyword search, implemented from scratch with an in-memory inverted index.

For a query q and chunk d:

    score(q, d) = sum over terms t in q of
                  idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

where tf is the term's count in the chunk, |d| the chunk's length in terms, avgdl the mean
length, N the number of chunks and df the number of chunks containing t. This idf variant
(the one Lucene uses) never goes negative, so a very common term can't penalize a match.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"[^\W_]+")
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its of on or our so such than that the their them then there these they this to
    was we were what when where which while who why will with would you your""".split()  # noqa: SIM905
)


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


class BM25Index:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.size = len(texts)
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(self.size, dtype=np.float32)
        for doc_id, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths[doc_id] = sum(counts.values())
            for term, tf in counts.items():
                postings.setdefault(term, []).append((doc_id, tf))
        # Inverted index: term -> (chunk ids, term frequencies) as arrays for vectorized scoring.
        self._postings = {
            term: (np.array([d for d, _ in hits]), np.array([tf for _, tf in hits], dtype=np.float32))
            for term, hits in postings.items()
        }
        self._idf = {
            term: math.log(1 + (self.size - len(hits) + 0.5) / (len(hits) + 0.5))
            for term, hits in postings.items()
        }
        avgdl = float(lengths.mean()) if self.size else 0.0
        # The length-normalization part of the denominator depends only on the chunk: precompute it.
        self._length_norm = (
            k1 * (1 - b + b * lengths / avgdl) if avgdl > 0 else np.full(self.size, k1, dtype=np.float32)
        )

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every indexed chunk for ``query`` (0 where no query term occurs)."""
        scores = np.zeros(self.size, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self._postings:
                continue
            doc_ids, tf = self._postings[term]
            scores[doc_ids] += self._idf[term] * tf * (self.k1 + 1) / (tf + self._length_norm[doc_ids])
        return scores
