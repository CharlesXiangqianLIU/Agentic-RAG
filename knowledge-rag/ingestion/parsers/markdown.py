"""Markdown -> ParsedDocument.

Uses markdown-it-py to produce a flat token stream, then folds tokens back
into paragraphs, headings (with their level recorded as the page-number is
unavailable in Markdown), and tables. Page numbers are not meaningful for
Markdown, so we synthesize them every ~5 000 characters of body text so
that very long documents still get distinguishable attribution.
"""
from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt

from ingestion.parser import (
    ParsedDocument,
    ParsedParagraph,
    ParsedRow,
    ParsedTable,
)

# Soft page boundary in characters; mirrors how a printed Markdown render
# breaks across pages and prevents one giant "page 1" for very long files.
_CHARS_PER_PSEUDO_PAGE = 5000


def parse_markdown(path: Path | str) -> ParsedDocument:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(text)

    doc = ParsedDocument(filename=p.name)
    current_section = "Introduction"
    chars_emitted = 0

    def _page_for_chars(n: int) -> int:
        return 1 + (n // _CHARS_PER_PSEUDO_PAGE)

    # ── walk tokens ─────────────────────────────────────────────────────
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # Headings: heading_open / inline / heading_close
        if tok.type == "heading_open":
            inline = tokens[i + 1]
            heading_text = (inline.content or "").strip()
            if heading_text:
                current_section = heading_text
                doc.paragraphs.append(
                    ParsedParagraph(
                        text=heading_text,
                        page_number=_page_for_chars(chars_emitted),
                        is_header=True,
                    )
                )
                chars_emitted += len(heading_text)
            # skip past heading_close
            while i < len(tokens) and tokens[i].type != "heading_close":
                i += 1
            i += 1
            continue

        # Paragraphs
        if tok.type == "paragraph_open":
            inline = tokens[i + 1]
            body = (inline.content or "").strip()
            if body:
                doc.paragraphs.append(
                    ParsedParagraph(
                        text=body,
                        page_number=_page_for_chars(chars_emitted),
                        is_header=False,
                    )
                )
                chars_emitted += len(body)
            while i < len(tokens) and tokens[i].type != "paragraph_close":
                i += 1
            i += 1
            continue

        # Tables
        if tok.type == "table_open":
            rows: list[ParsedRow] = []
            table_page = _page_for_chars(chars_emitted)
            # Walk rows until table_close
            j = i + 1
            while j < len(tokens) and tokens[j].type != "table_close":
                if tokens[j].type == "tr_open":
                    cells: list[str] = []
                    k = j + 1
                    while k < len(tokens) and tokens[k].type != "tr_close":
                        if tokens[k].type in ("th_open", "td_open"):
                            inline = tokens[k + 1]
                            cells.append((inline.content or "").strip())
                        k += 1
                    if any(c for c in cells):
                        rows.append(ParsedRow(cells=cells, page_number=table_page))
                    j = k
                j += 1
            if rows:
                doc.tables.append(
                    ParsedTable(
                        rows=rows,
                        section=current_section,
                        page_number=table_page,
                    )
                )
                chars_emitted += sum(len(c) for r in rows for c in r.cells)
            i = j + 1
            continue

        # List items / blockquotes / fences: flatten any inline content into a paragraph
        if tok.type == "inline" and tok.content.strip():
            body = tok.content.strip()
            doc.paragraphs.append(
                ParsedParagraph(
                    text=body,
                    page_number=_page_for_chars(chars_emitted),
                    is_header=False,
                )
            )
            chars_emitted += len(body)
        elif tok.type == "fence" and (tok.content or "").strip():
            body = tok.content.strip()
            doc.paragraphs.append(
                ParsedParagraph(
                    text=body,
                    page_number=_page_for_chars(chars_emitted),
                    is_header=False,
                )
            )
            chars_emitted += len(body)

        i += 1

    return doc
