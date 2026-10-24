"""Pydantic models for the public API and for data passed between agents."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# nli / llm-as-judge use Claude; local-nli runs an open-source NLI model on the local CPU.
VerificationMode = Literal["nli", "llm-as-judge", "local-nli"]
Verdict = Literal["supported", "partially_supported", "unsupported"]
# verified: claims were judged and `score` is set. no_context: nothing was retrieved to check
# the answer against. failed: the verifier errored or declined.
GroundingStatus = Literal["verified", "no_context", "failed"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=2000)
    selected_document: str | None = Field(default=None, max_length=512)
    top_k: int = Field(default=5, ge=1, le=20)
    include_web: bool = True
    include_arxiv: bool = True
    verify_grounding: bool = True
    # None: the server's default (Claude NLI when an API key is configured, local NLI otherwise).
    verification_mode: VerificationMode | None = None

    @field_validator("selected_document")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        return value or None


class DocumentHit(BaseModel):
    """A chunk retrieved from the vector index."""

    id: str
    score: float
    document: str
    page_num: int | None = None
    preview: str
    content: str | None = None
    citation_index: int | None = None  # the [n] the answer uses to cite this chunk


class SourceItem(BaseModel):
    """A web or arXiv search result."""

    title: str
    url: str | None = None
    snippet: str
    kind: str
    citation_index: int | None = None


class ClaimVerdict(BaseModel):
    claim_text: str
    citation_indices: list[int] = Field(default_factory=list)
    verdict: Verdict
    reason: str | None = None


class GroundingReport(BaseModel):
    status: GroundingStatus = "verified"
    score: float | None = None  # mean claim weight in [0, 1]; set only when status is "verified"
    claims: list[ClaimVerdict] = Field(default_factory=list)
    unsupported_sentences: list[str] = Field(default_factory=list)
    mode: VerificationMode | None = None
    detail: str | None = None  # why the answer could not be verified


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    retrieval: list[DocumentHit]
    grounding: GroundingReport | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class SessionRecord(BaseModel):
    document: str = Field(max_length=512)
    question: str = Field(max_length=4000)
    answer: str = Field(max_length=100_000)
    grounding_score: float | None = Field(default=None, ge=0.0, le=1.0)
    flagged_claims: list[ClaimVerdict] = Field(default_factory=list, max_length=200)
    created_at: datetime = Field(default_factory=_utcnow)


class ExportPayload(BaseModel):
    records: list[SessionRecord] = Field(max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)
