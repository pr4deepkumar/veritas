"""Web evidence from the DuckDuckGo Instant Answer API.

This API returns knowledge-graph answers (mostly Wikipedia abstracts and related topics),
not a ranked list of web pages, so many specific research questions yield no results.
"""

from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import unquote, urlparse

import requests

from veritas.core.models import SourceItem


def _leaf_topics(topics: list[dict]) -> Iterator[dict]:
    """RelatedTopics mixes result entries with named groups that nest more entries."""
    for topic in topics:
        if "Topics" in topic:
            yield from _leaf_topics(topic.get("Topics") or [])
        else:
            yield topic


def _title_from(url: str | None, text: str) -> str:
    # Topic URLs end in the entity name (".../Mercury_(planet)"); the text starts with it too.
    if url:
        slug = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]).replace("_", " ").strip()
        if slug:
            return slug
    return text if len(text) <= 80 else text[:80].rsplit(" ", 1)[0] + "…"


def parse_instant_answer(payload: dict, max_results: int) -> list[SourceItem]:
    items: list[SourceItem] = []
    abstract = (payload.get("AbstractText") or "").strip()
    if abstract:
        items.append(
            SourceItem(
                title=payload.get("Heading") or "DuckDuckGo",
                url=payload.get("AbstractURL") or None,
                snippet=abstract,
                kind="web",
            )
        )
    for topic in [
        *_leaf_topics(payload.get("Results") or []),
        *_leaf_topics(payload.get("RelatedTopics") or []),
    ]:
        if len(items) >= max_results:
            break
        text = (topic.get("Text") or "").strip()
        if not text:
            continue
        url = topic.get("FirstURL") or None
        items.append(SourceItem(title=_title_from(url, text), url=url, snippet=text, kind="web"))
    return items[:max_results]


class DuckDuckGoClient:
    API = "https://api.duckduckgo.com/"

    def search(self, query: str, max_results: int = 5) -> list[SourceItem]:
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        resp = requests.get(self.API, params=params, timeout=(5, 15))
        resp.raise_for_status()
        return parse_instant_answer(resp.json(), max_results)
