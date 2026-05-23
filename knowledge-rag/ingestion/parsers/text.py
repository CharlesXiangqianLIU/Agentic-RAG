"""Plain text -> ParsedDocument.

Paragraphs are separated by blank lines. Short ALL-CAPS lines are detected
as section headings via the same ``_looks_like_header`` heuristic used by
the .docx parser. Tables are not extracted (txt has no table syntax).
Pseudo page numbers increment every ~5 000 characters for long files.
"""
from __future__ import annotations

import re
from pathlib import Path

from ingestion.parser import (
    ParsedDocument,
    ParsedParagraph,
    _looks_like_header,
)

_CHARS_PER_PSEUDO_PAGE = 5000


def parse_text(path: Path | str) -> ParsedDocument:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")

    doc = ParsedDocument(filename=p.name)
    chars_emitted = 0
    for block in re.split(r"\n\s*\n", text):
        body = block.strip()
        if not body:
            continue
        is_header = _looks_like_header(body)
        page = 1 + (chars_emitted // _CHARS_PER_PSEUDO_PAGE)
        doc.paragraphs.append(
            ParsedParagraph(text=body, page_number=page, is_header=is_header)
        )
        chars_emitted += len(body)
    return doc
