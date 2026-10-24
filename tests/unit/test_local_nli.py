"""Local NLI verification logic, with a rule-based stand-in for the model (no download)."""

from __future__ import annotations

import sys

import pytest

from veritas.core.errors import ConfigurationError
from veritas.core.evidence import Evidence
from veritas.core.grounding import GroundingVerifier
from veritas.local.nli import HuggingFaceNLI, LocalNLIVerifier, _context, premise_windows, split_claims

EVIDENCE = [
    Evidence(
        index=1,
        kind="document",
        title="Q2",
        text="Revenue grew 12% year over year. Gross margin improved to 61%.",
    ),
    Evidence(index=2, kind="document", title="HR", text="Headcount ended the year at 2,150 employees."),
]


def fake_nli(contradiction_words: tuple[str, ...] = (), max_premise_chars: int | None = None):
    """Entailment when the hypothesis appears in the premise, contradiction when it contains
    one of ``contradiction_words``, neutral otherwise. With ``max_premise_chars``, longer
    premises come back neutral, like a real NLI model losing track of a long passage."""
    seen: list[tuple[str, str]] = []

    def classify(pairs):
        seen.extend(pairs)
        results = []
        for premise, hypothesis in pairs:
            too_long = max_premise_chars is not None and len(premise) > max_premise_chars
            if not too_long and hypothesis.removeprefix("It ").lower() in premise.lower():
                label = "entailment"
            elif any(word in hypothesis for word in contradiction_words):
                label = "contradiction"
            else:
                label = "neutral"
            results.append(
                {name: 0.9 if name == label else 0.05 for name in ("entailment", "neutral", "contradiction")}
            )
        return results

    classify.seen = seen
    return classify


def judge(answer: str, classifier=None):
    classifier = classifier or fake_nli()
    return LocalNLIVerifier(classifier).judge(answer, EVIDENCE), classifier


def spans(answer: str) -> list[tuple[str, list[int]]]:
    return [(claim.text, claim.citations) for claim in split_claims(answer)]


# -- claim splitting --------------------------------------------------------------------------


def test_one_claim_per_cited_sentence():
    assert spans("Revenue grew 12% year-over-year [1]. Costs fell 3.5% [2][3].") == [
        ("Revenue grew 12% year-over-year", [1]),
        ("Costs fell 3.5%", [2, 3]),
    ]


def test_a_citation_mid_sentence_closes_a_clause():
    assert spans("The product has a 24-hour battery [1] and will retail at $199 [2].") == [
        ("The product has a 24-hour battery", [1]),
        ("will retail at $199", [2]),
    ]


def test_uncited_sentences_are_claims_without_citations():
    assert spans("Lumirax was approved on March 14, 2025.") == [
        ("Lumirax was approved on March 14, 2025", [])
    ]


def test_sentences_about_missing_evidence_are_not_claims():
    assert spans("The headset costs $349 [1]. The sources do not state a release date.") == [
        ("The headset costs $349", [1])
    ]


@pytest.mark.parametrize(
    "sentence",
    [
        "The release doesn't say how quickly the buyback will be completed.",
        "The label does not say whether it is waterproof.",
        "The benchmark did not report results for other languages.",
        "The documents don\N{RIGHT SINGLE QUOTATION MARK}t mention the price.",
        "The launch date isn't specified.",
        "There is no information about pricing.",
    ],
)
def test_statements_of_what_the_sources_omit_are_not_claims(sentence):
    assert spans(f"The headset costs $349 [1]. {sentence}") == [("The headset costs $349", [1])]


@pytest.mark.parametrize(
    "answer",
    [
        "The warranty does not cover water damage.",  # a claim about the warranty, not the sources
        "The company did not report a loss [1].",  # citing a passage makes it a claim
    ],
)
def test_negative_claims_are_still_checked(answer):
    assert len(spans(answer)) == 1


def test_a_skipped_sentence_keeps_the_citation_it_starts_with():
    assert spans("Revenue grew 12% last year. [1] The report doesn't say why.") == [
        ("Revenue grew 12% last year", [1])
    ]


def test_a_citation_after_the_period_belongs_to_the_previous_sentence():
    assert spans("Revenue grew 12% last year. [1] Costs fell sharply [2].") == [
        ("Revenue grew 12% last year", [1]),
        ("Costs fell sharply", [2]),
    ]


def test_bullets_and_decimals():
    assert spans("- Margin was 61.5% in Q2 [1]\n- Cash ended at $310 million [2]") == [
        ("Margin was 61.5% in Q2", [1]),
        ("Cash ended at $310 million", [2]),
    ]


# -- verdicts -------------------------------------------------------------------------------


def test_an_entailed_claim_is_supported():
    (verdict,), _ = judge("Gross margin improved to 61% [1].")
    assert verdict.verdict == "supported"
    assert "[1]" in verdict.reason


def test_a_contradicted_claim_is_unsupported():
    (verdict,), _ = judge("Gross margin fell sharply [1].", fake_nli(contradiction_words=("fell",)))
    assert verdict.verdict == "unsupported"
    assert verdict.reason.startswith("Contradicted")


def test_a_neutral_claim_about_the_same_facts_is_partially_supported():
    (verdict,), _ = judge("Gross margin improved to 61% in every region [1].")
    assert verdict.verdict == "partially_supported"


def test_a_neutral_claim_on_another_topic_is_unsupported():
    (verdict,), _ = judge("The CEO is Jane Doe [1].")
    assert verdict.verdict == "unsupported"


def test_citing_a_missing_passage_is_unsupported_and_the_right_passage_is_named():
    (verdict,), _ = judge("Headcount ended the year at 2,150 employees [7].")
    assert verdict.verdict == "unsupported"
    assert verdict.reason.startswith("Cites passage [7], which does not exist.")
    assert "Passage [2] does support it" in verdict.reason


def test_a_claim_is_judged_only_against_the_passages_it_cites():
    (verdict,), classifier = judge("Headcount ended the year at 2,150 employees [1].")
    assert verdict.verdict == "unsupported"  # true, but passage [1] doesn't say it
    assert classifier.seen[0][0] == f"Q2: {EVIDENCE[0].text}"  # judged against the cited passage first
    assert verdict.reason.startswith("Cited passage [1] doesn't support this claim")
    assert "Passage [2] does support it" in verdict.reason  # ...then explained as a wrong citation


def test_a_fabricated_fact_is_not_explained_as_a_wrong_citation():
    (verdict,), _ = judge("The CEO is Jane Doe [1].")
    assert verdict.verdict == "unsupported"
    assert "does support it" not in verdict.reason


def test_long_passages_are_also_checked_sentence_by_sentence():
    passage = (
        "Northwind reported revenue of $95 million. Gross margin improved to 61%. "
        "Operating income was $14 million. Headcount reached 2,150. The dividend stayed at $0.40."
    )
    evidence = [Evidence(index=1, kind="document", title="Q2", text=passage)]
    verifier = LocalNLIVerifier(fake_nli(max_premise_chars=80))

    (verdict,) = verifier.judge("Gross margin improved to 61% [1].", evidence)

    assert verdict.verdict == "supported"  # the whole passage reads as neutral; a window entails it


def test_premise_windows_pair_up_consecutive_sentences():
    assert premise_windows("One two three. Four five six.") == []  # short passages need none
    assert premise_windows("A b c. D e f. G h i.") == ["A b c. D e f.", "D e f. G h i."]


def test_an_uncited_claim_is_checked_against_every_passage():
    (verdict,), classifier = judge("Headcount ended the year at 2,150 employees.")
    assert verdict.verdict == "supported"
    assert "[2]" in verdict.reason
    assert len(classifier.seen) == 3  # all passages together, then each one alone


def test_a_clause_without_its_subject_gets_a_placeholder_one():
    _, classifier = judge("The report covers revenue [1] and shows gross margin improved to 61% [1].")
    assert ("It shows gross margin improved to 61%") in [hypothesis for _, hypothesis in classifier.seen]


# -- integration with GroundingVerifier ------------------------------------------------------


def test_local_mode_needs_no_claude_client():
    verifier = GroundingVerifier(client=None, local=LocalNLIVerifier(fake_nli()))

    report = verifier.verify(
        "Gross margin improved to 61% [1]. The CEO is Jane Doe [1].", EVIDENCE, mode="local-nli"
    )

    assert report.status == "verified"
    assert report.mode == "local-nli"
    assert report.score == pytest.approx(0.5)
    assert report.unsupported_sentences == ["The CEO is Jane Doe"]


def test_claude_modes_without_a_key_fail_cleanly():
    report = GroundingVerifier(client=None).verify("Gross margin improved to 61% [1].", EVIDENCE, mode="nli")
    assert report.status == "failed"
    assert "ANTHROPIC_API_KEY" in report.detail


def test_missing_local_dependencies_fail_cleanly(monkeypatch):
    def unavailable(model_name, device):
        raise ConfigurationError("needs the [local] extra")

    monkeypatch.setattr("veritas.core.grounding.load_local_verifier", unavailable)
    report = GroundingVerifier(client=None).verify(
        "Gross margin improved to 61% [1].", EVIDENCE, mode="local-nli"
    )
    assert report.status == "failed"
    assert "[local] extra" in report.detail


def test_without_transformers_installed_the_error_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # makes `import transformers` fail
    with pytest.raises(ConfigurationError, match=r"\.\[local\]"):
        HuggingFaceNLI("any/model")


def test_every_premise_is_headed_by_its_source_title():
    # "Hardware revenue declined 2%" doesn't say whose; the title does ("Northwind Q2 report").
    passage = Evidence(
        index=1,
        kind="document",
        title="reports/northwind_q2-report",
        text="Revenue grew 9%. Hardware revenue declined 2%. Gross margin improved to 61%.",
    )
    classifier = fake_nli()
    LocalNLIVerifier(classifier).judge("Northwind's hardware revenue declined 2% [1].", [passage])
    premises = [premise for premise, _ in classifier.seen]
    assert len(premises) == 3  # the whole passage and its two windows
    assert all(premise.startswith("northwind q2 report: ") for premise in premises)


@pytest.mark.parametrize(
    ("title", "context"),
    [
        ("reports/Q2_2026-results", "Q2 2026 results"),
        ("Attention Is All You Need", "Attention Is All You Need"),
    ],
)
def test_titles_become_plain_words(title, context):
    assert _context(title) == context
