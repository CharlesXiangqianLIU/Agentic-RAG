# tests/test_parser.py
"""parse_docx end-to-end test against the neutral sample.docx fixture."""
from ingestion.parser import parse_docx, ParsedDocument, ParsedParagraph, ParsedTable, ParsedRow


def test_parse_returns_parsed_document(tmp_docx):
    result = parse_docx(tmp_docx)
    assert isinstance(result, ParsedDocument)
    assert result.filename.endswith(".docx")


def test_parse_extracts_paragraphs(tmp_docx):
    result = parse_docx(tmp_docx)
    assert len(result.paragraphs) > 0
    assert result.paragraphs[0].page_number >= 1


def test_parse_extracts_tables_with_rows(tmp_docx):
    result = parse_docx(tmp_docx)
    assert len(result.tables) > 0
    table = result.tables[0]
    assert len(table.rows) > 0
    assert table.rows[0].page_number >= 1


def test_parse_detects_section_headers(tmp_docx):
    result = parse_docx(tmp_docx)
    headers = [p for p in result.paragraphs if p.is_header]
    assert len(headers) > 0


def test_parsed_row_has_text_method(tmp_docx):
    result = parse_docx(tmp_docx)
    row = result.tables[0].rows[0]
    text = row.to_text()
    assert isinstance(text, str)
    assert "|" in text


def test_paragraphs_have_is_header_bool(tmp_docx):
    result = parse_docx(tmp_docx)
    for p in result.paragraphs:
        assert isinstance(p.is_header, bool)
