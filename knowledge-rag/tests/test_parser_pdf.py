"""PDF parser tests.

pdfplumber and pypdf are both mocked so the test suite never opens a real
PDF binary fixture. The two branches are tested independently:

* ``_parse_pdf_with_tables`` — the preferred pdfplumber-backed path,
  including table extraction.
* ``_parse_pdf_text_only`` — the pypdf fallback used when pdfplumber is
  unavailable, exercising the same paragraph-splitting / heading logic.
"""
from unittest.mock import MagicMock, patch

from ingestion.parsers.pdf import (
    parse_pdf,
    _parse_pdf_text_only,
    _parse_pdf_with_tables,
)


# ---------------------------------------------------------------------------
# pypdf fallback path
# ---------------------------------------------------------------------------


def _mock_pypdf_reader(page_texts):
    reader = MagicMock()
    reader.pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        reader.pages.append(page)
    return reader


def test_text_only_one_paragraph_per_block_per_page(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    pages = [
        "Page one paragraph one.\n\nPage one paragraph two.",
        "Page two paragraph one.",
    ]
    with patch("pypdf.PdfReader", return_value=_mock_pypdf_reader(pages)):
        doc = _parse_pdf_text_only(p)

    assert doc.filename == "doc.pdf"
    pairs = {(para.page_number, para.text) for para in doc.paragraphs}
    assert (1, "Page one paragraph one.") in pairs
    assert (1, "Page one paragraph two.") in pairs
    assert (2, "Page two paragraph one.") in pairs


def test_text_only_detects_all_caps_heading(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    pages = ["INTRODUCTION\n\nA body paragraph follows.\n"]
    with patch("pypdf.PdfReader", return_value=_mock_pypdf_reader(pages)):
        doc = _parse_pdf_text_only(p)

    headers = [para for para in doc.paragraphs if para.is_header]
    assert any(h.text == "INTRODUCTION" for h in headers)


def test_text_only_skips_pages_that_fail_to_extract(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    reader = MagicMock()
    good = MagicMock()
    good.extract_text.return_value = "Good page content."
    bad = MagicMock()
    bad.extract_text.side_effect = RuntimeError("decryption failed")
    reader.pages = [good, bad]

    with patch("pypdf.PdfReader", return_value=reader):
        doc = _parse_pdf_text_only(p)

    assert any("Good page content" in para.text for para in doc.paragraphs)


def test_text_only_empty_pdf(tmp_path):
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"%PDF-stub")
    with patch("pypdf.PdfReader", return_value=_mock_pypdf_reader([])):
        doc = _parse_pdf_text_only(p)
    assert doc.paragraphs == []


# ---------------------------------------------------------------------------
# pdfplumber path with table extraction
# ---------------------------------------------------------------------------


def _mock_pdfplumber_open(pages_spec):
    """Build a context-manager mock whose ``pages`` attribute exposes the spec.

    ``pages_spec`` is a list of dicts per page; supported keys:
      * ``text`` — what ``extract_text`` returns (also used as the
        cropped-page text unless ``prose_text`` is set).
      * ``tables`` — what ``extract_tables`` returns.
      * ``table_bboxes`` — list of (x0, top, x1, bottom) tuples. When
        non-empty, ``find_tables`` returns matching Table-like objects
        and ``outside_bbox`` returns a cropped page whose ``extract_text``
        yields ``prose_text`` (defaulting to ``text``).
      * ``prose_text`` — what the cropped page's ``extract_text`` returns.
    """
    pdf_mock = MagicMock()
    pages = []
    for spec in pages_spec:
        page = MagicMock()
        page.extract_text.return_value = spec.get("text", "")
        page.extract_tables.return_value = spec.get("tables", [])

        bboxes = spec.get("table_bboxes", [])
        prose_text = spec.get("prose_text", spec.get("text", ""))

        # find_tables() returns objects with a .bbox attribute
        page.find_tables.return_value = [MagicMock(bbox=bb) for bb in bboxes]

        # outside_bbox() returns a cropped page whose extract_text yields prose_text.
        # We make outside_bbox chainable by returning the same cropped mock.
        cropped = MagicMock()
        cropped.extract_text.return_value = prose_text
        cropped.outside_bbox.return_value = cropped
        page.outside_bbox.return_value = cropped

        pages.append(page)
    pdf_mock.pages = pages

    cm = MagicMock()
    cm.__enter__.return_value = pdf_mock
    cm.__exit__.return_value = None
    return cm


def test_pdfplumber_extracts_paragraphs_per_page(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    spec = [
        {"text": "Page one paragraph one.\n\nPage one paragraph two."},
        {"text": "Page two paragraph one."},
    ]
    with patch("pdfplumber.open", return_value=_mock_pdfplumber_open(spec)):
        doc = _parse_pdf_with_tables(p)

    pairs = {(para.page_number, para.text) for para in doc.paragraphs}
    assert (1, "Page one paragraph one.") in pairs
    assert (2, "Page two paragraph one.") in pairs


def test_pdfplumber_extracts_tables(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    spec = [{
        "text": "QUARTERLY NUMBERS\n\nIntroductory text.",
        "tables": [
            [
                ["Quarter", "Revenue"],
                ["Q1", "100"],
                ["Q2", "120"],
            ]
        ],
    }]
    with patch("pdfplumber.open", return_value=_mock_pdfplumber_open(spec)):
        doc = _parse_pdf_with_tables(p)

    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.page_number == 1
    assert table.section == "QUARTERLY NUMBERS"  # heading detected from prose preceding it
    cells_per_row = [r.cells for r in table.rows]
    assert ["Quarter", "Revenue"] in cells_per_row
    assert ["Q1", "100"] in cells_per_row
    assert ["Q2", "120"] in cells_per_row


def test_pdfplumber_handles_none_cells(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    spec = [{
        "tables": [
            [
                ["A", None, "C"],
                [None, "B", None],
                [None, None, None],  # entirely blank — should be skipped
            ]
        ],
    }]
    with patch("pdfplumber.open", return_value=_mock_pdfplumber_open(spec)):
        doc = _parse_pdf_with_tables(p)

    assert len(doc.tables) == 1
    rows = [r.cells for r in doc.tables[0].rows]
    assert rows == [["A", "", "C"], ["", "B", ""]]


def test_pdfplumber_per_page_text_failure_does_not_break_tables(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    pdf_mock = MagicMock()
    page = MagicMock()
    page.extract_text.side_effect = RuntimeError("text extraction broken")
    page.extract_tables.return_value = [[["A", "B"], ["1", "2"]]]
    pdf_mock.pages = [page]
    cm = MagicMock()
    cm.__enter__.return_value = pdf_mock
    cm.__exit__.return_value = None

    with patch("pdfplumber.open", return_value=cm):
        doc = _parse_pdf_with_tables(p)

    # text failed → no paragraphs, but tables still came through
    assert doc.paragraphs == []
    assert len(doc.tables) == 1
    assert doc.tables[0].rows[0].cells == ["A", "B"]


def test_pdfplumber_crops_out_tables_from_prose(tmp_path):
    """When a page has a table bbox, prose comes from the cropped page only."""
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    spec = [{
        # The uncropped page text would include the linearised table.
        "text": "Intro paragraph.\n\nQuarter Revenue\nQ1 100\nQ2 120",
        # The cropped page (tables removed) only retains the prose.
        "prose_text": "Intro paragraph.",
        "tables": [[["Quarter", "Revenue"], ["Q1", "100"], ["Q2", "120"]]],
        "table_bboxes": [(0, 100, 500, 200)],
    }]
    with patch("pdfplumber.open", return_value=_mock_pdfplumber_open(spec)):
        doc = _parse_pdf_with_tables(p)

    para_texts = " ".join(para.text for para in doc.paragraphs)
    assert "Intro paragraph" in para_texts
    # The linearised table content must NOT show up in prose any more.
    assert "Quarter Revenue" not in para_texts
    assert "Q1 100" not in para_texts
    # The structured table is still there.
    assert len(doc.tables) == 1


def test_pdfplumber_no_tables_skips_cropping(tmp_path):
    """When find_tables returns nothing, prose extraction is unchanged."""
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    spec = [{"text": "Page with no tables.", "table_bboxes": []}]
    with patch("pdfplumber.open", return_value=_mock_pdfplumber_open(spec)):
        doc = _parse_pdf_with_tables(p)

    assert any("no tables" in para.text for para in doc.paragraphs)
    assert doc.tables == []


def test_pdfplumber_crop_failure_falls_back_to_empty_prose(tmp_path):
    """If outside_bbox() raises, log a warning and emit no prose (tables stay)."""
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    pdf_mock = MagicMock()
    page = MagicMock()
    page.find_tables.return_value = [MagicMock(bbox=(0, 0, 1, 1))]
    page.outside_bbox.side_effect = RuntimeError("crop failed")
    page.extract_tables.return_value = [[["a", "b"], ["1", "2"]]]
    pdf_mock.pages = [page]
    cm = MagicMock()
    cm.__enter__.return_value = pdf_mock
    cm.__exit__.return_value = None

    with patch("pdfplumber.open", return_value=cm):
        doc = _parse_pdf_with_tables(p)

    assert doc.paragraphs == []
    assert len(doc.tables) == 1


def test_pdfplumber_empty_pdf(tmp_path):
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"%PDF-stub")
    with patch("pdfplumber.open", return_value=_mock_pdfplumber_open([])):
        doc = _parse_pdf_with_tables(p)
    assert doc.paragraphs == []
    assert doc.tables == []


# ---------------------------------------------------------------------------
# Top-level dispatcher: pdfplumber preferred, pypdf fallback on ImportError
# ---------------------------------------------------------------------------


def test_parse_pdf_uses_pdfplumber_when_available(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    spec = [{"text": "via pdfplumber", "tables": [[["x", "y"], ["1", "2"]]]}]
    with patch("pdfplumber.open", return_value=_mock_pdfplumber_open(spec)):
        doc = parse_pdf(p)

    assert any("via pdfplumber" in para.text for para in doc.paragraphs)
    assert len(doc.tables) == 1


def test_parse_pdf_falls_back_to_pypdf_when_pdfplumber_missing(tmp_path, monkeypatch):
    """If `import pdfplumber` raises ImportError, parse_pdf should use pypdf."""
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-stub")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pdfplumber":
            raise ImportError("simulated absence of pdfplumber")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with patch("pypdf.PdfReader", return_value=_mock_pypdf_reader(["fallback content"])):
        doc = parse_pdf(p)

    assert any("fallback content" in para.text for para in doc.paragraphs)
    assert doc.tables == []  # pypdf path does not extract tables
