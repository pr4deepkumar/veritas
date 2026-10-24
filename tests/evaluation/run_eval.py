"""Benchmark the grounding verifier against labeled cases (``eval_dataset.json`` by default).

A grounding issue (``unsupported`` or ``partially_supported``) is the positive class, so
precision is "of the claims flagged, how many were real issues" and recall is "of the real
issues, how many were flagged". Every labeled claim is scored exactly once: the verifier's
claims are matched to the labeled claim they overlap most, and a labeled claim counts as
flagged when any claim matched to it is flagged, so splitting one claim into two pieces is not
penalized twice. Claims the dataset doesn't label count as false positives if flagged. Cases
the verifier could not evaluate are reported as errors, not as misses.

Usage:
    python tests/evaluation/run_eval.py                       # free, offline: local NLI model
    python tests/evaluation/run_eval.py --mode nli llm-as-judge  # Claude modes: needs ANTHROPIC_API_KEY
    python tests/evaluation/run_eval.py --limit 5             # quick smoke test on the first cases
    python tests/evaluation/run_eval.py --dataset tests/evaluation/eval_dataset_heldout.json

The local-nli mode needs ``pip install -e ".[local]"`` and downloads the model on first use.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(__file__).with_name("eval_dataset.json")  # development set
# Written by independent agents that never saw this code. The first set was held out for the
# first version of the verifier and has since been used as development data. The second was
# written after the verifier was frozen; nothing is tuned on it.
INDEPENDENT_DATASET = Path(__file__).with_name("eval_dataset_independent.json")
HELD_OUT_DATASET = Path(__file__).with_name("eval_dataset_heldout.json")
MODES = ("local-nli", "nli", "llm-as-judge")
CLAUDE_MODES = {"nli", "llm-as-judge"}
ISSUE_VERDICTS = {"unsupported", "partially_supported"}
MATCH_THRESHOLD = 0.5
_WORD = re.compile(r"[^\W_]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from in is it its of on or that the to was
    were with""".split()  # noqa: SIM905
)


def _content_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS}


def match_score(predicted: str, labeled: str) -> float:
    """Similarity of a predicted claim to a labeled one. Containment and word overlap count,
    so a fragment of a labeled claim ("covers accidental damage") still matches it."""
    pred, target = predicted.lower(), labeled.lower()
    score = difflib.SequenceMatcher(None, pred, target).ratio()
    if target in pred or pred in target:
        score = max(score, 0.8)
    words = _content_words(pred)
    if len(words) >= 3:
        score = max(score, len(words & _content_words(target)) / len(words))
    return score


def best_match_index(predicted: str, expected_claims: list[dict]) -> int | None:
    scores = [match_score(predicted, claim["claim_text"]) for claim in expected_claims]
    if not scores:
        return None
    best = max(range(len(scores)), key=scores.__getitem__)
    return best if scores[best] > MATCH_THRESHOLD else None


def _status(actual_issue: bool, predicted_issue: bool) -> str:
    return {(True, True): "TP", (False, True): "FP", (True, False): "FN", (False, False): "TN"}[
        (actual_issue, predicted_issue)
    ]


def score_case(case_id: str, report: Any, expected_claims: list[dict]) -> tuple[Counter, list[dict]]:
    """Confusion counts and per-claim details for one case's GroundingReport."""
    counts: Counter = Counter()
    details: list[dict] = []
    if report.status != "verified":
        counts["errors"] += 1
        details.append(
            {
                "case_id": case_id,
                "claim": "-",
                "expected": "-",
                "predicted": "-",
                "status": "ERROR",
                "reason": report.detail or report.status,
            }
        )
        return counts, details

    matched: list[list[Any]] = [[] for _ in expected_claims]
    unmatched: list[Any] = []
    for pred in report.claims:
        index = best_match_index(pred.claim_text, expected_claims)
        if index is None:
            unmatched.append(pred)
        else:
            matched[index].append(pred)

    for expected, preds in zip(expected_claims, matched, strict=True):
        status = _status(
            expected["verdict"] in ISSUE_VERDICTS, any(p.verdict in ISSUE_VERDICTS for p in preds)
        )
        counts[status] += 1
        details.append(
            {
                "case_id": case_id,
                "claim": expected["claim_text"],
                "expected": expected["verdict"],
                "predicted": ", ".join(p.verdict for p in preds) or "N/A",
                "status": status if preds else f"{status} (missed)",
                "reason": " | ".join(p.reason or "" for p in preds)
                or "The verifier did not extract this claim.",
            }
        )

    for pred in unmatched:  # a claim the dataset doesn't label
        status = _status(False, pred.verdict in ISSUE_VERDICTS)
        counts[status] += 1
        details.append(
            {
                "case_id": case_id,
                "claim": pred.claim_text,
                "expected": "N/A",
                "predicted": pred.verdict,
                "status": f"{status} (unmatched)",
                "reason": pred.reason,
            }
        )
    return counts, details


def compute_metrics(counts: Counter) -> dict[str, float]:
    tp, fp, fn, tn = (counts[k] for k in ("TP", "FP", "FN", "TN"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / total if total else 0.0,
    }


def case_evidence(case: dict) -> list[Any]:
    """Numbered passages for a case: retrieved chunks first, then any web/arXiv sources."""
    from veritas.core.evidence import build_evidence
    from veritas.core.models import DocumentHit, SourceItem

    hits = [DocumentHit(**hit) for hit in case["retrieval"]]
    sources = [SourceItem(**source) for source in case.get("sources", [])]
    return build_evidence(hits, sources).passages


def run_evaluation(dataset: list[dict], mode: str, verifier: Any, verbose: bool = True) -> dict[str, Any]:
    print(f"\n=== Mode: {mode} ===")
    counts: Counter = Counter()
    details: list[dict] = []
    for case in dataset:
        report = verifier.verify(case["answer"], case_evidence(case), mode=mode)
        case_counts, case_details = score_case(case["id"], report, case["expected_claims"])
        counts.update(case_counts)
        details.extend(case_details)

    for row in details:
        if not verbose and row["status"] in ("TP", "TN"):
            continue
        print(
            f"[{row['case_id']}] {row['status']:<16} expected={row['expected']:<20} predicted={row['predicted']}"
        )
        print(f"    claim:  {row['claim'][:110]}")
        print(f"    reason: {row['reason']}")

    result = {**compute_metrics(counts), **{k: counts[k] for k in ("TP", "FP", "FN", "TN", "errors")}}
    print(
        f"\nTP={result['TP']} FP={result['FP']} FN={result['FN']} TN={result['TN']} errors={result['errors']}  "
        f"precision={result['precision']:.1%} recall={result['recall']:.1%} f1={result['f1']:.1%}"
    )
    return result


def results_markdown(results: dict[str, dict[str, Any]], models: dict[str, str], dataset: list[dict]) -> str:
    claims = [claim for case in dataset for claim in case["expected_claims"]]
    issues = sum(claim["verdict"] in ISSUE_VERDICTS for claim in claims)
    lines = [
        f"Dataset: {len(dataset)} cases, {len(claims)} labeled claims ({issues} grounding issues). "
        f"Run on {date.today().isoformat()}.",
        "",
        "| Mode | Model | Precision | Recall | F1 | TP / FP / FN / TN | Errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, r in results.items():
        lines.append(
            f"| {mode} | `{models[mode]}` | {r['precision']:.1%} | {r['recall']:.1%} | {r['f1']:.1%} "
            f"| {r['TP']} / {r['FP']} / {r['FN']} / {r['TN']} | {r['errors']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", nargs="+", choices=MODES, default=["local-nli"])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, help="only evaluate the first N cases")
    parser.add_argument("--quiet", action="store_true", help="only print the cases the verifier got wrong")
    args = parser.parse_args(argv)

    if str(ROOT / "src") not in sys.path:  # allow running from a checkout without installing
        sys.path.insert(0, str(ROOT / "src"))
    from veritas.core.clients.claude import ClaudeClient
    from veritas.core.grounding import GroundingVerifier
    from veritas.core.settings import settings

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))[: args.limit]
    modes = list(dict.fromkeys(args.mode))
    client = ClaudeClient() if CLAUDE_MODES & set(modes) else None  # local-nli needs no key
    verifier = GroundingVerifier(client)
    models = {
        mode: settings.local_nli_model
        if mode == "local-nli"
        else f"{settings.verifier_model_name} (effort: {settings.verifier_effort})"
        for mode in modes
    }
    results = {mode: run_evaluation(dataset, mode, verifier, verbose=not args.quiet) for mode in modes}
    print("\n" + results_markdown(results, models, dataset))


if __name__ == "__main__":
    main()
