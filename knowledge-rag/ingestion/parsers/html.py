"""HTML -> ParsedDocument.

Uses BeautifulSoup to walk the document body in document order. Strips
common chrome tags (``script``, ``style``, ``nav``, ``header``,
``footer``, ``aside``, ``noscript``) before extraction. ``h1``-``h6``
elements become heading paragraphs and update the current section.
``<table>`` elements are decomposed into ``ParsedTable`` with one row per
``<tr>``. Page numbers are not meaningful for HTML (single-page documents)
so the whole document lives on page 1.
"""
from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from ingestion.parser import (
    ParsedDocument,
    ParsedParagraph,
    ParsedRow,
    ParsedTable,
)

_CHROME_TAGS = ("script", "style", "nav", "header", "footer", "aside", "noscript")
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_BODY_TAGS = _HEADING_TAGS + ("p", "li", "blockquote", "pre", "table")


def parse_html(path: Path | str) -> ParsedDocument:
    p = Path(path)
    html = p.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # Drop chrome before walking
    for tag_name in _CHROME_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    doc = ParsedDocument(filename=p.name)
    current_section = "Introduction"

    root = soup.body or soup

    for elem in root.find_all(_BODY_TAGS):
        name = elem.name.lower()

        if name == "table":
            rows: list[ParsedRow] = []
            for tr in elem.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(ParsedRow(cells=cells, page_number=1))
            if rows:
                doc.tables.append(
                    ParsedTable(rows=rows, section=current_section, page_number=1)
                )
            continue

        text = elem.get_text(separator=" ", strip=True)
        if not text:
            continue

        if name in _HEADING_TAGS:
            current_section = text
            doc.paragraphs.append(
                ParsedParagraph(text=text, page_number=1, is_header=True)
            )
        else:
            doc.paragraphs.append(
                ParsedParagraph(text=text, page_number=1, is_header=False)
            )

    return doc
