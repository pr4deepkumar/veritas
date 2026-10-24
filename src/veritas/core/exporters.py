"""CSV and PDF exports of a research session, including grounding results."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from veritas.core.models import SessionRecord

CSV_FIELDS = ["document", "question", "answer", "created_at", "grounding_score", "flagged_claims"]

# Spreadsheet apps evaluate cells starting with these characters as formulas. Answers often
# start with "-" (bullet lists), and retrieved text could carry a deliberate payload.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# The built-in PDF fonts only cover Latin-1; map the punctuation models commonly emit.
_PDF_REPLACEMENTS = str.maketrans(
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{SINGLE LOW-9 QUOTATION MARK}": "'",
        "\N{LEFT DOUBLE QUOTATION MARK}": '"',
        "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
        "\N{DOUBLE LOW-9 QUOTATION MARK}": '"',
        "\N{EN DASH}": "-",
        "\N{EM DASH}": "-",
        "\N{MINUS SIGN}": "-",
        "\N{HORIZONTAL ELLIPSIS}": "...",
        "\N{BULLET}": "*",
        "\N{NO-BREAK SPACE}": " ",
        "\N{LESS-THAN OR EQUAL TO}": "<=",
        "\N{GREATER-THAN OR EQUAL TO}": ">=",
        "\N{RIGHTWARDS ARROW}": "->",
        "\N{LEFTWARDS ARROW}": "<-",
        "\N{MULTIPLICATION SIGN}": "x",
    }
)


def _csv_cell(value: str) -> str:
    return f"'{value}" if value.startswith(_FORMULA_PREFIXES) else value


def _format_score(score: float | None) -> str:
    return "" if score is None else f"{score:.2f}"


def to_csv(records: Sequence[SessionRecord]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for record in records:
        flagged = " | ".join(f"[{c.verdict}] {c.claim_text}" for c in record.flagged_claims)
        writer.writerow(
            {
                "document": _csv_cell(record.document),
                "question": _csv_cell(record.question),
                "answer": _csv_cell(record.answer),
                "created_at": record.created_at.isoformat(),
                "grounding_score": _format_score(record.grounding_score),
                "flagged_claims": _csv_cell(flagged),
            }
        )
    return buffer.getvalue()


def _pdf_text(text: str) -> str:
    return text.translate(_PDF_REPLACEMENTS).encode("latin-1", "replace").decode("latin-1")


def _paragraph(pdf: FPDF, text: str, height: float) -> None:
    # Return to the left margin after each block; multi_cell's default leaves the cursor at
    # the right margin, so the next full-width cell would have no room.
    pdf.multi_cell(0, height, _pdf_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def to_pdf_bytes(records: Sequence[SessionRecord]) -> bytes:
    pdf = FPDF()
    pdf.set_title("VERITAS Research Session Report")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    _paragraph(pdf, "Research Session Report", 10)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    _paragraph(pdf, f"{len(records)} quer{'y' if len(records) == 1 else 'ies'}", 6)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for idx, record in enumerate(records, start=1):
        pdf.set_font("Helvetica", "B", 12)
        _paragraph(pdf, f"{idx}. {record.question}", 7)

        meta = [f"Document: {record.document}", record.created_at.isoformat(sep=" ", timespec="minutes")]
        if record.grounding_score is not None:
            meta.append(f"Grounding score: {round(record.grounding_score * 100)}%")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        _paragraph(pdf, "  |  ".join(meta), 5)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

        pdf.set_font("Helvetica", "", 10)
        _paragraph(pdf, record.answer, 5.5)

        if record.flagged_claims:
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 10)
            _paragraph(pdf, "Flagged claims", 6)
            pdf.set_font("Helvetica", "", 9)
            for claim in record.flagged_claims:
                label = claim.verdict.replace("_", " ").upper()
                line = f"[{label}] {claim.claim_text}"
                if claim.reason:
                    line += f" - {claim.reason}"
                _paragraph(pdf, line, 5)
        pdf.ln(5)

    return bytes(pdf.output())
