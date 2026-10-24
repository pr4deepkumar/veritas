"""Numbered evidence passages shared by answer synthesis and grounding verification.

The answer generator and the verifier must see the same passages under the same numbers,
otherwise a citation such as ``[3]`` would point at different text in each step. Both build
their prompts from one :class:`EvidenceSet` via :func:`format_evidence`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple
from xml.sax.saxutils import quoteattr

from veritas.core.models import DocumentHit, SourceItem

# Passage text is untrusted; keep it from closing the wrapper tags it is quoted in.
_WRAPPER_CLOSE_TAG = re.compile(r"</\s*(passage|evidence)\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class Evidence:
    index: int  # 1-based citation number
    kind: str  # "document", "web" or "arxiv"
    title: str
    text: str
    url: str | None = None
    page_num: int | None = None


class EvidenceSet(NamedTuple):
    hits: list[DocumentHit]  # copies with citation_index set
    sources: list[SourceItem]  # copies with citation_index set
    passages: list[Evidence]


def build_evidence(hits: Sequence[DocumentHit], sources: Sequence[SourceItem] = ()) -> EvidenceSet:
    """Number retrieved chunks first ([1]..[n]), then external sources ([n+1]..)."""
    numbered_hits: list[DocumentHit] = []
    numbered_sources: list[SourceItem] = []
    passages: list[Evidence] = []
    for hit in hits:
        index = len(passages) + 1
        numbered_hits.append(hit.model_copy(update={"citation_index": index}))
        passages.append(
            Evidence(
                index=index,
                kind="document",
                title=hit.document,
                text=hit.content or hit.preview,
                page_num=hit.page_num,
            )
        )
    for source in sources:
        index = len(passages) + 1
        numbered_sources.append(source.model_copy(update={"citation_index": index}))
        passages.append(
            Evidence(index=index, kind=source.kind, title=source.title, text=source.snippet, url=source.url)
        )
    return EvidenceSet(numbered_hits, numbered_sources, passages)


def format_evidence(passages: Sequence[Evidence]) -> str:
    """Render passages as tagged blocks for a prompt."""
    if not passages:
        return "<evidence>\n(no passages were retrieved)\n</evidence>"
    blocks = []
    for passage in passages:
        attrs = [
            f"index={quoteattr(str(passage.index))}",
            f"source={quoteattr(passage.kind)}",
            f"title={quoteattr(passage.title)}",
        ]
        if passage.page_num is not None:
            attrs.append(f"page={quoteattr(str(passage.page_num))}")
        if passage.url:
            attrs.append(f"url={quoteattr(passage.url)}")
        body = _WRAPPER_CLOSE_TAG.sub(r"</ \1>", passage.text.strip()) or "(empty)"
        blocks.append(f"<passage {' '.join(attrs)}>\n{body}\n</passage>")
    return "<evidence>\n" + "\n".join(blocks) + "\n</evidence>"
