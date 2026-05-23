import pytest

from ingestion.parser import parse_document, SUPPORTED_EXTENSIONS


def test_parse_document_rejects_unknown_extension(tmp_path):
    p = tmp_path / "weird.zip"
    p.write_bytes(b"PK\x03\x04")
    with pytest.raises(ValueError, match="Unsupported document extension"):
        parse_document(p)


def test_supported_extensions_constant_includes_all_formats():
    for ext in (".docx", ".md", ".markdown", ".txt", ".pdf", ".html", ".htm"):
        assert ext in SUPPORTED_EXTENSIONS


def test_parse_document_dispatches_text(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("Hello world.\n\nSecond paragraph.", encoding="utf-8")
    doc = parse_document(p)
    bodies = [para.text for para in doc.paragraphs]
    assert any("Hello world" in t for t in bodies)


def test_parse_document_dispatches_markdown(tmp_path):
    p = tmp_path / "hello.md"
    p.write_text("# Title\n\nBody.", encoding="utf-8")
    doc = parse_document(p)
    headers = [para for para in doc.paragraphs if para.is_header]
    assert headers and headers[0].text == "Title"


def test_parse_document_dispatches_html(tmp_path):
    p = tmp_path / "hello.html"
    p.write_text("<html><body><h1>T</h1><p>B</p></body></html>", encoding="utf-8")
    doc = parse_document(p)
    headers = [para for para in doc.paragraphs if para.is_header]
    assert headers and headers[0].text == "T"
