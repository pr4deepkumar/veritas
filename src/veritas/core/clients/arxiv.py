"""arXiv search via the public Atom API."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import requests

from veritas.core.models import SourceItem

_NS = {"atom": "http://www.w3.org/2005/Atom"}

# arXiv ORs space-separated terms, so filler words in a natural-language question match
# almost every paper and dilute the ranking. Search on the content words only.
_STOPWORDS = frozenset(
    """
    a about after all also an and any are as at be been being between both but by can
    compare compared could did do does doing during each for from had has have how i if
    in into is it its may might more most of on or other our over should so some such
    than that the their them then there these they this those through to under using
    versus vs was we were what when where which while who whom whose why will with would
    you your
    """.split()  # noqa: SIM905
)
MAX_TERMS = 12


def build_search_query(question: str) -> str:
    """Turn a question into an arXiv ``search_query`` of content terms (empty if none)."""
    terms: list[str] = []
    for word in re.findall(r"[^\W_]+", question.lower()):
        if len(word) > 1 and word not in _STOPWORDS and word not in terms:
            terms.append(word)
    return " OR ".join(f"all:{term}" for term in terms[:MAX_TERMS])


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_feed(xml_text: str) -> list[SourceItem]:
    root = ET.fromstring(xml_text)
    items: list[SourceItem] = []
    for entry in root.findall("atom:entry", _NS):
        title = _clean(entry.findtext("atom:title", namespaces=_NS))
        summary = _clean(entry.findtext("atom:summary", namespaces=_NS))
        if not title or not summary:
            continue
        url = (
            next(
                (
                    link.get("href")
                    for link in entry.findall("atom:link", _NS)
                    if link.get("rel") == "alternate" and link.get("href")
                ),
                None,
            )
            or _clean(entry.findtext("atom:id", namespaces=_NS))
            or None
        )
        items.append(SourceItem(title=title, url=url, snippet=summary, kind="arxiv"))
    return items


class ArxivClient:
    API = "https://export.arxiv.org/api/query"

    def search(self, query: str, max_results: int = 5) -> list[SourceItem]:
        search_query = build_search_query(query)
        if not search_query:
            return []
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        resp = requests.get(self.API, params=params, timeout=(5, 15))
        resp.raise_for_status()
        return parse_feed(resp.text)[:max_results]
