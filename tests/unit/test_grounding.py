from __future__ import annotations

from unittest.mock import MagicMock

import anthropic
import httpx2
import pytest

from veritas.core.clients.claude import ClaudeClient
from veritas.core.errors import LLMOutputError, LLMRefusalError
from veritas.core.evidence import Evidence
from veritas.core.grounding import (
    JUDGE_SYSTEM_PROMPT,
    NLI_SYSTEM_PROMPT,
    GroundingVerifier,
    _ClaimJudgment,
    _GroundingJudgment,
    score_claims,
)
from veritas.core.settings import settings

EVIDENCE = [
    Evidence(index=1, kind="document", title="doc1", text="chunk1 text", page_num=1),
    Evidence(index=2, kind="web", title="Result", text="chunk2 text", url="https://example.com"),
]


def judgment(*claims: tuple[str, str]) -> _GroundingJudgment:
    return _GroundingJudgment(
        claims=[
            _ClaimJudgment(claim_text=text, citation_indices=[1], verdict=verdict, reason="r")
            for text, verdict in claims
        ]
    )


@pytest.fixture
def client() -> MagicMock:
    return MagicMock(spec=ClaudeClient)


@pytest.fixture
def verifier(client: MagicMock) -> GroundingVerifier:
    return GroundingVerifier(client)


def test_answer_without_evidence_is_unverified_not_fully_grounded(verifier, client):
    report = verifier.verify("The answer text.", evidence=[])

    assert report.status == "no_context"
    assert report.score is None
    assert report.claims == []
    assert report.detail
    client.generate_structured.assert_not_called()


def test_score_is_mean_of_verdict_weights(verifier, client):
    # (1.0 + 0.5 + 0.0) / 3 = 0.5
    client.generate_structured.return_value = judgment(
        ("Claim 1 is true.", "supported"),
        ("Claim 2 is mostly true.", "partially_supported"),
        ("Claim 3 is false.", "unsupported"),
    )

    report = verifier.verify("Claim 1 is true. Claim 2 is mostly true. Claim 3 is false.", EVIDENCE)

    assert report.status == "verified"
    assert report.mode == "nli"
    assert report.score == pytest.approx(0.5)
    assert [c.verdict for c in report.claims] == ["supported", "partially_supported", "unsupported"]
    assert report.unsupported_sentences == ["Claim 2 is mostly true.", "Claim 3 is false."]


def test_answer_with_no_factual_claims_scores_one():
    report = score_claims([])
    assert report.status == "verified"
    assert report.score == 1.0


@pytest.mark.parametrize(
    ("mode", "prompt"), [("nli", NLI_SYSTEM_PROMPT), ("llm-as-judge", JUDGE_SYSTEM_PROMPT)]
)
def test_mode_selects_prompt_and_sends_numbered_evidence(verifier, client, mode, prompt):
    client.generate_structured.return_value = judgment()

    report = verifier.verify("Revenue grew 12% [1].", EVIDENCE, mode=mode)

    kwargs = client.generate_structured.call_args.kwargs
    assert kwargs["system"] == prompt
    assert kwargs["output_model"] is _GroundingJudgment
    assert '<passage index="1"' in kwargs["prompt"]
    assert '<passage index="2"' in kwargs["prompt"]
    assert "<answer>\nRevenue grew 12% [1].\n</answer>" in kwargs["prompt"]
    assert report.mode == mode


def test_verifier_model_and_effort_come_from_settings(verifier, client, monkeypatch):
    monkeypatch.setattr(settings, "verifier_model", "claude-opus-5-5")
    monkeypatch.setattr(settings, "verifier_effort", "high")
    client.generate_structured.return_value = judgment()

    verifier.verify("answer", EVIDENCE)

    kwargs = client.generate_structured.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["effort"] == "high"


def test_verifier_model_defaults_to_chat_model(verifier, client):
    client.generate_structured.return_value = judgment()
    verifier.verify("answer", EVIDENCE)
    assert client.generate_structured.call_args.kwargs["model"] == settings.chat_model


@pytest.mark.parametrize(
    "error",
    [
        LLMRefusalError("cyber"),
        LLMOutputError("cut off"),
        anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")),
    ],
)
def test_model_failures_leave_the_answer_unverified(verifier, client, error):
    client.generate_structured.side_effect = error

    report = verifier.verify("answer", EVIDENCE)

    assert report.status == "failed"
    assert report.score is None
    assert report.claims == []
    assert report.detail


def test_programming_errors_are_not_swallowed(verifier, client):
    client.generate_structured.side_effect = TypeError("bug")
    with pytest.raises(TypeError):
        verifier.verify("answer", EVIDENCE)
