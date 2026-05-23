import textwrap

from ingestion.parsers.text import parse_text


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_text_splits_on_blank_lines(tmp_path):
    p = _write(tmp_path, "doc.txt", """
        First paragraph here.

        Second paragraph here.

        Third one with more words.
    """)
    doc = parse_text(p)
    bodies = [para.text for para in doc.paragraphs]
    assert any("First paragraph" in t for t in bodies)
    assert any("Second paragraph" in t for t in bodies)
    assert any("Third one" in t for t in bodies)


def test_text_detects_all_caps_short_heading(tmp_path):
    p = _write(tmp_path, "doc.txt", """
        SECTION TITLE

        Body content follows here.
    """)
    doc = parse_text(p)
    headers = [para for para in doc.paragraphs if para.is_header]
    assert any(h.text == "SECTION TITLE" for h in headers)


def test_text_empty_file(tmp_path):
    p = _write(tmp_path, "empty.txt", "")
    doc = parse_text(p)
    assert doc.paragraphs == []
    assert doc.tables == []


def test_text_no_tables_ever(tmp_path):
    p = _write(tmp_path, "doc.txt", "| a | b |\n| c | d |\n")
    doc = parse_text(p)
    assert doc.tables == []  # plain text parser does not interpret pipe tables
