"""Tests for the scoring logic of tests/evaluation/run_eval.py and its dataset (no model calls)."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from types import SimpleNamespace

import pytest
import run_eval

EXPECTED = [
    {"claim_text": "The new product features an upgraded battery life of 24 hours", "verdict": "supported"},
    {"claim_text": "will retail at a starting price of $199", "verdict": "unsupported"},
]
DATASET = json.loads(run_eval.DEFAULT_DATASET.read_text(encoding="utf-8"))
DATASETS = {
    "development": DATASET,
    "independent": json.loads(run_eval.INDEPENDENT_DATASET.read_text(encoding="utf-8")),
    "held-out": json.loads(run_eval.HELD_OUT_DATASET.read_text(encoding="utf-8")),
}
each_dataset = pytest.mark.parametrize("dataset", DATASETS.values(), ids=DATASETS.keys())


def report(*claims: tuple[str, str], status: str = "verified", detail: str | None = None):
    return SimpleNamespace(
        status=status,
        detail=detail,
        claims=[SimpleNamespace(claim_text=text, verdict=verdict, reason="r") for text, verdict in claims],
    )


def test_correct_verdicts_are_true_positives_and_negatives():
    counts, _ = run_eval.score_case(
        "case",
        report(
            ("The new product features an upgraded battery life of 24 hours", "supported"),
            ("It will retail at a starting price of $199", "unsupported"),
        ),
        EXPECTED,
    )
    assert counts == Counter(TN=1, TP=1)


def test_a_missed_issue_is_a_false_negative():
    counts, details = run_eval.score_case(
        "case",
        report(("The new product features an upgraded battery life of 24 hours", "supported")),
        EXPECTED,
    )
    assert counts == Counter(TN=1, FN=1)
    assert details[-1]["status"] == "FN (missed)"


def test_flagging_an_unlabeled_claim_is_a_false_positive():
    counts, details = run_eval.score_case(
        "case", report(("Water resistance is 50 meters", "unsupported")), EXPECTED[:1]
    )
    assert counts == Counter(FP=1, TN=1)
    assert [d["status"] for d in details] == ["TN (missed)", "FP (unmatched)"]


def test_a_claim_split_in_two_is_scored_once_and_caught_by_either_piece():
    labeled = [
        {
            "claim_text": "The device has a two-year limited warranty that covers manufacturing defects and accidental damage",
            "verdict": "partially_supported",
        }
    ]
    counts, _ = run_eval.score_case(
        "case",
        report(
            ("The device has a two-year limited warranty that covers manufacturing defects", "supported"),
            ("The warranty covers accidental damage", "unsupported"),
        ),
        labeled,
    )
    assert counts == Counter(TP=1)


def test_a_failed_verification_is_counted_as_an_error_not_a_miss():
    counts, details = run_eval.score_case("case", report(status="failed", detail="timeout"), EXPECTED)
    assert counts == Counter(errors=1)
    assert details[0]["reason"] == "timeout"


def test_metrics():
    metrics = run_eval.compute_metrics(Counter(TP=3, FP=1, FN=1, TN=5))
    assert metrics == pytest.approx({"precision": 0.75, "recall": 0.75, "f1": 0.75, "accuracy": 0.8})


def test_metrics_without_data_do_not_divide_by_zero():
    assert run_eval.compute_metrics(Counter()) == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
    }


@each_dataset
def test_dataset_is_well_formed(dataset):
    assert len({case["id"] for case in dataset}) == len(dataset)
    for case in dataset:
        passages = run_eval.case_evidence(case)  # parses retrieval and sources
        assert passages, case["id"]
        for claim in case["expected_claims"]:
            assert claim["verdict"] in {"supported", "partially_supported", "unsupported"}
            assert claim["claim_text"] in case["answer"], f"{case['id']}: label is not verbatim answer text"


@each_dataset
def test_perfect_predictions_score_perfectly_on_the_whole_dataset(dataset):
    """Guards the matcher: each labeled claim must match itself, not a neighbour."""
    total: Counter = Counter()
    for case in dataset:
        counts, _ = run_eval.score_case(
            case["id"],
            report(*[(c["claim_text"], c["verdict"]) for c in case["expected_claims"]]),
            case["expected_claims"],
        )
        total.update(counts)
        assert counts["FP"] == counts["FN"] == 0, case["id"]
    assert run_eval.compute_metrics(total)["f1"] == 1.0


def test_results_table_records_how_the_numbers_were_produced():
    result = {"precision": 0.9, "recall": 0.8, "f1": 0.847, "TP": 8, "FP": 1, "FN": 2, "TN": 9, "errors": 0}
    table = run_eval.results_markdown({"local-nli": result}, {"local-nli": "org/model"}, DATASET[:2])
    assert "Dataset: 2 cases, 2 labeled claims (1 grounding issues)" in table
    assert "| local-nli | `org/model` | 90.0% | 80.0% | 84.7% | 8 / 1 / 2 / 9 | 0 |" in table


@pytest.mark.parametrize(("first", "second"), list(itertools.combinations(DATASETS, 2)))
def test_the_sets_do_not_share_cases(first, second):
    for field in ("id", "question", "answer"):
        assert not {c[field] for c in DATASETS[first]} & {c[field] for c in DATASETS[second]}, field
