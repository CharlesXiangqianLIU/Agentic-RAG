# tests/test_chunker.py
"""Chunker tests using a domain-neutral synthetic ParsedDocument.

The fixture mirrors create_fixture.py: an H1 + intro paragraph, a 6-column
table (header + 2 data rows), an H2 + closing paragraph. No domain
abbreviations or unit patterns are used, so the assertions hold with an
empty default domain pack.
"""
from ingestion.chunker import chunk_document, Chunk
from ingestion.parser import ParsedDocument, ParsedParagraph, ParsedTable, ParsedRow


def make_doc() -> ParsedDocument:
    doc = ParsedDocument(filename="quarterly_review_2026Q1.docx")
    doc.paragraphs = [
        ParsedParagraph("Quarterly Performance Review", 1, True),
        ParsedParagraph(
            "The team reviewed all open accounts and recorded the headline numbers below.",
            1,
            False,
        ),
        ParsedParagraph("Notes", 2, True),
        ParsedParagraph(
            "Account A-002 grew the fastest this quarter; A-001 stayed on plan.",
            2,
            False,
        ),
    ]
    doc.tables = [
        ParsedTable(
            section="Quarterly Performance Review",
            rows=[
                ParsedRow(["Account", "Region", "Quarter", "Status", "Owner", "Revenue"], 1),
                ParsedRow(["A-001", "North", "Q1 2026", "Active", "Alice", "120000"], 1),
                ParsedRow(["A-002", "South", "Q1 2026", "Active", "Bob", "150000"], 1),
            ],
            page_number=1,
        )
    ]
    return doc


def test_chunks_are_produced():
    chunks = chunk_document(make_doc())
    assert len(chunks) > 0


def test_chunk_has_required_fields():
    chunk = chunk_document(make_doc())[0]
    assert chunk.text
    assert chunk.page_number >= 1
    assert chunk.section
    assert chunk.source_file == "quarterly_review_2026Q1.docx"
    assert chunk.chunk_type in ("paragraph", "table_row")


def test_table_rows_are_individual_chunks():
    chunks = chunk_document(make_doc())
    table_chunks = [c for c in chunks if c.chunk_type == "table_row"]
    # 2 data rows (header row is not emitted as its own chunk)
    assert len(table_chunks) == 2


def test_table_row_chunk_contains_header_context():
    chunks = chunk_document(make_doc())
    table_chunks = [c for c in chunks if c.chunk_type == "table_row"]
    for chunk in table_chunks:
        assert "Account" in chunk.text  # header is prepended


def test_table_row_never_split():
    """Each data row must remain atomic — the row that mentions A-001 must
    also carry its Revenue cell '120000' (and not be torn across chunks)."""
    chunks = chunk_document(make_doc())
    for chunk in chunks:
        if "A-001" in chunk.text and "North" in chunk.text:
            assert "120000" in chunk.text


def test_chunk_tokens_within_bounds():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = chunk_document(make_doc())
    paragraph_chunks = [c for c in chunks if c.chunk_type == "paragraph"]
    for chunk in paragraph_chunks:
        tokens = len(enc.encode(chunk.text))
        assert tokens <= 600


def test_synonyms_in_table_chunk():
    chunks = chunk_document(make_doc())
    table_chunks = [c for c in chunks if c.chunk_type == "table_row"]
    # All table chunks expose a synonyms list (empty with an empty domain pack).
    for chunk in table_chunks:
        assert isinstance(chunk.synonyms, list)


def test_table_row_has_structured_fields_in_metadata():
    """Verify that table_row chunks have structured_fields dict in metadata."""
    chunks = chunk_document(make_doc())
    table_chunks = [c for c in chunks if c.chunk_type == "table_row"]
    assert len(table_chunks) == 2
    for chunk in table_chunks:
        assert "structured_fields" in chunk.metadata
        assert isinstance(chunk.metadata["structured_fields"], dict)


def test_structured_fields_map_headers_to_values():
    """Verify that structured_fields correctly maps header names to cell values.

    With an empty default domain pack the chunker does not rewrite values
    (no abbreviation expansion, no unit normalization), so values are
    expected verbatim.
    """
    chunks = chunk_document(make_doc())
    table_chunks = [c for c in chunks if c.chunk_type == "table_row"]
    assert len(table_chunks) == 2

    first = table_chunks[0].metadata["structured_fields"]
    assert first["Account"] == "A-001"
    assert first["Region"] == "North"
    assert first["Quarter"] == "Q1 2026"
    assert first["Status"] == "Active"
    assert first["Owner"] == "Alice"
    assert first["Revenue"] == "120000"

    second = table_chunks[1].metadata["structured_fields"]
    assert second["Account"] == "A-002"
    assert second["Region"] == "South"
    assert second["Revenue"] == "150000"
