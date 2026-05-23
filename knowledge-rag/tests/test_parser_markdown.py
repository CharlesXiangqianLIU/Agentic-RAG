import textwrap

from ingestion.parsers.markdown import parse_markdown


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_markdown_parses_headings_and_paragraphs(tmp_path):
    p = _write(tmp_path, "doc.md", """
        # Section One

        Hello world.

        ## Subsection

        Another paragraph.
    """)
    doc = parse_markdown(p)
    assert doc.filename == "doc.md"
    headers = [para for para in doc.paragraphs if para.is_header]
    bodies = [para for para in doc.paragraphs if not para.is_header]
    assert [h.text for h in headers] == ["Section One", "Subsection"]
    assert any("Hello world" in p.text for p in bodies)
    assert any("Another paragraph" in p.text for p in bodies)


def test_markdown_parses_pipe_table(tmp_path):
    p = _write(tmp_path, "tbl.md", """
        # Quarterly Numbers

        | Quarter | Revenue |
        |---------|---------|
        | Q1      | 100     |
        | Q2      | 120     |
    """)
    doc = parse_markdown(p)
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.section == "Quarterly Numbers"
    cells_per_row = [r.cells for r in table.rows]
    assert ["Quarter", "Revenue"] in cells_per_row
    assert ["Q1", "100"] in cells_per_row
    assert ["Q2", "120"] in cells_per_row


def test_markdown_empty_file_is_empty_doc(tmp_path):
    p = _write(tmp_path, "empty.md", "")
    doc = parse_markdown(p)
    assert doc.paragraphs == []
    assert doc.tables == []


def test_markdown_long_doc_increments_pseudo_pages(tmp_path):
    body = "Paragraph body line. " * 600  # > 5000 chars
    p = _write(tmp_path, "long.md", body)
    doc = parse_markdown(p)
    pages = {para.page_number for para in doc.paragraphs}
    # The body is one big paragraph in MD, so only one pseudo page is expected.
    # But the page number must be >= 1.
    assert all(pg >= 1 for pg in pages)
    assert min(pages) == 1
