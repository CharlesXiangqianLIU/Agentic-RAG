from ingestion.parsers.html import parse_html


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_html_extracts_headings_and_paragraphs(tmp_path):
    html = """
    <html><body>
      <h1>Annual Report</h1>
      <p>Introductory paragraph.</p>
      <h2>Revenue</h2>
      <p>Revenue grew steadily.</p>
    </body></html>
    """
    p = _write(tmp_path, "doc.html", html)
    doc = parse_html(p)
    headers = [para for para in doc.paragraphs if para.is_header]
    bodies = [para for para in doc.paragraphs if not para.is_header]
    assert [h.text for h in headers] == ["Annual Report", "Revenue"]
    assert any("Introductory" in p.text for p in bodies)
    assert any("Revenue grew" in p.text for p in bodies)


def test_html_strips_chrome(tmp_path):
    html = """
    <html><body>
      <nav>Home About Contact</nav>
      <header>Header bar</header>
      <p>Real content.</p>
      <footer>© 2026</footer>
      <script>alert('xss');</script>
    </body></html>
    """
    p = _write(tmp_path, "doc.html", html)
    doc = parse_html(p)
    bodies_text = " ".join(p.text for p in doc.paragraphs)
    assert "Real content" in bodies_text
    assert "Home About Contact" not in bodies_text
    assert "Header bar" not in bodies_text
    assert "2026" not in bodies_text
    assert "alert" not in bodies_text


def test_html_extracts_table(tmp_path):
    html = """
    <html><body>
      <h2>Numbers</h2>
      <table>
        <tr><th>Quarter</th><th>Revenue</th></tr>
        <tr><td>Q1</td><td>100</td></tr>
        <tr><td>Q2</td><td>120</td></tr>
      </table>
    </body></html>
    """
    p = _write(tmp_path, "tbl.html", html)
    doc = parse_html(p)
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.section == "Numbers"
    rows = [r.cells for r in table.rows]
    assert ["Quarter", "Revenue"] in rows
    assert ["Q1", "100"] in rows
    assert ["Q2", "120"] in rows


def test_html_empty_body(tmp_path):
    p = _write(tmp_path, "doc.html", "<html><body></body></html>")
    doc = parse_html(p)
    assert doc.paragraphs == []
    assert doc.tables == []


def test_html_htm_extension_also_supported(tmp_path):
    from ingestion.parser import parse_document

    p = _write(tmp_path, "alt.htm", "<html><body><p>via parse_document</p></body></html>")
    doc = parse_document(p)
    assert any("via parse_document" in para.text for para in doc.paragraphs)
