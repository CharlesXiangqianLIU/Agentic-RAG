"""PDF -> ParsedDocument with table extraction.

Uses pdfplumber for both prose text and structured tables. For each page
we first locate tables via ``Page.find_tables()``, then extract prose
from the page with the table regions cropped out (``outside_bbox``).
That keeps the same content from appearing twice in the index — once as
a linearised text run, once as ``ParsedTable`` rows. Tables themselves
go through ``extract_tables`` on the original (uncropped) page.

If ``pdfplumber`` is unavailable, the parser falls back to ``pypdf``
text-only extraction with a logged warning. Per-page extraction
failures are logged and the page is skipped.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from ingestion.parser import (
    ParsedDocument,
    ParsedParagraph,
    ParsedRow,
    ParsedTable,
    _looks_like_header,
)

log = logging.getLogger(__name__)


def parse_pdf(path: Path | str) -> ParsedDocument:
    """Parse a PDF using pdfplumber; fall back to pypdf if unavailable."""
    p = Path(path)
    try:
        import pdfplumber  # noqa: F401 — kept for the optional fallback below
    except ImportError:
        log.warning(
            "pdfplumber not installed; falling back to pypdf text-only extraction "
            "for %s. Install pdfplumber to enable table extraction.",
            p.name,
        )
        return _parse_pdf_text_only(p)

    return _parse_pdf_with_tables(p)


# ---------------------------------------------------------------------------
# pdfplumber backend (preferred): text + structured tables
# ---------------------------------------------------------------------------


def _parse_pdf_with_tables(path: Path) -> ParsedDocument:
    import pdfplumber

    doc = ParsedDocument(filename=path.name)
    current_section = "Introduction"

    with pdfplumber.open(str(path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            # Find table regions first so prose extraction can exclude them.
            try:
                table_objs = page.find_tables()
            except Exception as exc:
                log.warning("pdfplumber failed to locate tables on page %s of %s: %s", page_index, path.name, exc)
                table_objs = []
            table_bboxes = [getattr(t, "bbox", None) for t in table_objs]
            table_bboxes = [bb for bb in table_bboxes if bb]

            # ── Prose paragraphs (with tables cropped out) ─────────────────
            page_text = _prose_text_outside_tables(page, table_bboxes, page_index, path.name)

            for block in re.split(r"\n\s*\n", page_text):
                body = block.strip()
                if not body:
                    continue
                is_header = _looks_like_header(body)
                if is_header:
                    current_section = body
                doc.paragraphs.append(
                    ParsedParagraph(text=body, page_number=page_index, is_header=is_header)
                )

            # ── Structured tables (from the original, uncropped page) ──────
            try:
                tables = page.extract_tables() or []
            except Exception as exc:
                log.warning("pdfplumber failed to extract tables on page %s of %s: %s", page_index, path.name, exc)
                tables = []

            for raw_table in tables:
                rows = _table_rows_from_raw(raw_table, page_index)
                if rows:
                    doc.tables.append(
                        ParsedTable(rows=rows, section=current_section, page_number=page_index)
                    )

    return doc


def _prose_text_outside_tables(
    page, table_bboxes: list, page_index: int, filename: str
) -> str:
    """Extract page text after cropping out every table bbox.

    No bboxes → behave like the previous version and call ``extract_text``
    on the raw page. Errors from ``outside_bbox`` (e.g. invalid bbox) fall
    back to the uncropped extraction so we don't lose the whole page.
    """
    try:
        if not table_bboxes:
            return page.extract_text() or ""
        cropped = page
        for bbox in table_bboxes:
            cropped = cropped.outside_bbox(bbox)
        return cropped.extract_text() or ""
    except Exception as exc:
        log.warning(
            "pdfplumber failed to extract prose on page %s of %s: %s",
            page_index, filename, exc,
        )
        return ""


def _table_rows_from_raw(raw_table: list[list[str | None]], page_number: int) -> list[ParsedRow]:
    """Convert a pdfplumber raw table to ``ParsedRow``s.

    pdfplumber returns each cell as ``str | None``. We coerce None to "" and
    drop rows that are entirely empty after stripping.
    """
    rows: list[ParsedRow] = []
    for raw_row in raw_table:
        cells = [("" if cell is None else str(cell)).strip() for cell in raw_row]
        if any(cells):
            rows.append(ParsedRow(cells=cells, page_number=page_number))
    return rows


# ---------------------------------------------------------------------------
# pypdf fallback (text only)
# ---------------------------------------------------------------------------


def _parse_pdf_text_only(path: Path) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    doc = ParsedDocument(filename=path.name)

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            log.warning("pypdf failed to extract page %s of %s: %s", page_index, path.name, exc)
            continue

        for block in re.split(r"\n\s*\n", page_text):
            body = block.strip()
            if not body:
                continue
            is_header = _looks_like_header(body)
            doc.paragraphs.append(
                ParsedParagraph(text=body, page_number=page_index, is_header=is_header)
            )

    return doc
