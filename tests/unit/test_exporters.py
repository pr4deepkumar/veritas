from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from veritas.core.exporters import CSV_FIELDS, to_csv, to_pdf_bytes
from veritas.core.models import ClaimVerdict, SessionRecord

RECORD = SessionRecord(
    document="Q2_Report",
    question="How did revenue change?",
    answer="Revenue grew 12% — driven by “strong” demand… [1]",
    grounding_score=0.75,
    flagged_claims=[
        ClaimVerdict(
            claim_text="driven by strong demand",
            citation_indices=[1],
            verdict="partially_supported",
            reason="The passage mentions growth, not its cause.",
        )
    ],
    created_at=datetime(2026, 11, 16, 12, 30, tzinfo=timezone.utc),
)


def rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def test_csv_includes_grounding_results():
    (row,) = rows(to_csv([RECORD]))
    assert list(row) == CSV_FIELDS
    assert row["answer"] == RECORD.answer
    assert row["created_at"] == "2026-11-16T12:30:00+00:00"
    assert row["grounding_score"] == "0.75"
    assert row["flagged_claims"] == "[partially_supported] driven by strong demand"


def test_csv_leaves_score_blank_when_unverified():
    record = SessionRecord(document="d", question="q", answer="a")
    assert rows(to_csv([record]))[0]["grounding_score"] == ""


def test_csv_neutralizes_spreadsheet_formulas():
    record = SessionRecord(
        document='=HYPERLINK("http://evil.example")', question="+1", answer="- a bullet list"
    )
    (row,) = rows(to_csv([record]))
    assert row["document"] == '\'=HYPERLINK("http://evil.example")'
    assert row["question"] == "'+1"
    assert row["answer"] == "'- a bullet list"


def test_pdf_renders_several_records_with_non_latin_text():
    long_word = "https://example.com/" + "a" * 250
    records = [
        RECORD,
        RECORD,
        SessionRecord(
            document="d", question="Emoji 🚀 and 中文?", answer=f"See {long_word} for ≥ 5 → items."
        ),
    ]
    data = to_pdf_bytes(records)
    assert data.startswith(b"%PDF")
    assert len(data) > 1000


def test_pdf_with_no_records():
    assert to_pdf_bytes([]).startswith(b"%PDF")
