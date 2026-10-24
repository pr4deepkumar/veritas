from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from veritas.agents.arxiv_agent import ArxivAgent
from veritas.agents.web_search_agent import WebSearchAgent
from veritas.core.clients.arxiv import ArxivClient, build_search_query, parse_feed
from veritas.core.clients.duckduckgo import parse_instant_answer

# Shapes taken from real Instant Answer responses: an article (abstract, no related topics)
# and a disambiguation page (related topics, some nested in named groups).
ARTICLE = {
    "Heading": "Transformer (machine learning model)",
    "AbstractText": "A transformer is a deep learning architecture based on multi-head attention.",
    "AbstractURL": "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
    "RelatedTopics": [],
    "Results": [],
}
DISAMBIGUATION = {
    "Heading": "Mercury",
    "AbstractText": "",
    "RelatedTopics": [
        {"FirstURL": "https://duckduckgo.com/Mercury_(planet)", "Text": "Mercury (planet) The first planet."},
        {
            "Name": "Computing",
            "Topics": [
                {
                    "FirstURL": "https://duckduckgo.com/Mercury_(programming_language)",
                    "Text": "Mercury (programming language) A functional logic language.",
                }
            ],
        },
        {"FirstURL": "", "Text": ""},
    ],
}

ARXIV_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2201.00978v1</id>
    <title>PyramidTNT: Improved
      Transformer-in-Transformer Baselines</title>
    <link href="https://arxiv.org/abs/2201.00978v1" rel="alternate" type="text/html"/>
    <link href="https://arxiv.org/pdf/2201.00978v1" rel="related" type="application/pdf"/>
    <summary>  Transformer networks have achieved
  great progress.  </summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/9999.00001v1</id>
    <title>An entry without an abstract</title>
    <summary></summary>
  </entry>
</feed>"""


def test_article_abstract_is_the_first_web_result():
    (item,) = parse_instant_answer(ARTICLE, max_results=3)
    assert item.title == "Transformer (machine learning model)"
    assert item.url == ARTICLE["AbstractURL"]
    assert item.snippet.startswith("A transformer")
    assert item.kind == "web"


def test_related_topic_groups_are_flattened_and_titled_from_urls():
    items = parse_instant_answer(DISAMBIGUATION, max_results=5)
    assert [i.title for i in items] == ["Mercury (planet)", "Mercury (programming language)"]


def test_max_results_is_respected():
    assert len(parse_instant_answer(DISAMBIGUATION, max_results=1)) == 1


def test_title_falls_back_to_the_text_without_a_url():
    payload = {"RelatedTopics": [{"FirstURL": "", "Text": "word " * 40}]}
    (item,) = parse_instant_answer(payload, max_results=1)
    assert len(item.title) <= 81
    assert item.title.endswith("…")
    assert item.url is None


def test_empty_payload_has_no_results():
    assert parse_instant_answer({}, max_results=3) == []


def test_arxiv_feed_is_cleaned_and_links_to_the_abstract_page():
    (item,) = parse_feed(ARXIV_FEED)
    assert item.title == "PyramidTNT: Improved Transformer-in-Transformer Baselines"
    assert item.snippet == "Transformer networks have achieved great progress."
    assert item.url == "https://arxiv.org/abs/2201.00978v1"
    assert item.kind == "arxiv"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What is the efficacy of Mamba compared to Transformers?",
            "all:efficacy OR all:mamba OR all:transformers",
        ),
        ("GPT-4 vs. GPT-4: (risks)?", "all:gpt OR all:risks"),
        ("What is it?", ""),
    ],
)
def test_arxiv_query_keeps_content_words_only(question, expected):
    assert build_search_query(question) == expected


def test_arxiv_search_sends_https_request(monkeypatch):
    captured = {}

    def fake_get(url, params, timeout):
        captured.update(url=url, params=params)
        return SimpleNamespace(text=ARXIV_FEED, raise_for_status=lambda: None)

    monkeypatch.setattr("veritas.core.clients.arxiv.requests.get", fake_get)

    items = ArxivClient().search("mamba transformers", max_results=2)

    assert captured["url"].startswith("https://")
    assert captured["params"]["search_query"] == "all:mamba OR all:transformers"
    assert captured["params"]["max_results"] == 2
    assert len(items) == 1


def test_arxiv_search_without_content_words_makes_no_request(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("no request expected")

    monkeypatch.setattr("veritas.core.clients.arxiv.requests.get", fail)
    assert ArxivClient().search("What is it?") == []


class BrokenClient:
    def search(self, query, max_results):
        raise requests.ConnectionError("network down")


def test_agents_never_fail_the_request_when_a_source_is_down():
    assert WebSearchAgent(BrokenClient()).search("q") == []
    assert ArxivAgent(BrokenClient()).search("q") == []
