"""End-to-end: chunk → index → hybrid search against a real Qdrant.

Skips automatically when Qdrant isn't reachable (see ``conftest.py``).
The LLM is NOT invoked — this test verifies the retrieval half of the
pipeline (the half most sensitive to deployment drift). LLM-level
behaviour is already covered by the unit-mocked answer/critic suites.
"""
from __future__ import annotations

import os

import pytest

# Mark every test in this module as e2e for pytest -k filtering.
pytestmark = pytest.mark.e2e


def _ingest_neutral_corpus(collection: str, qdrant_url: str) -> int:
    from ingestion.chunker import chunk_document
    from ingestion.parser import (
        ParsedDocument,
        ParsedParagraph,
        ParsedRow,
        ParsedTable,
    )
    import ingestion.indexer as indexer_mod
    from retrieval.embedder import embed_texts, embed_texts_sparse

    # Wire indexer to the e2e Qdrant + collection.
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["QDRANT_COLLECTION"] = collection
    import config
    config.QDRANT_URL = qdrant_url
    config.QDRANT_COLLECTION = collection
    indexer_mod.QDRANT_URL = qdrant_url
    indexer_mod.QDRANT_COLLECTION = collection
    indexer_mod._client = None
    indexer_mod._collection_ensured = False
    # Force the indexer to use the live embedder, not the module-level stubs
    # that the unit-test suite installs for fast, mocked runs.
    indexer_mod.embed_texts = embed_texts
    indexer_mod.embed_texts_sparse = embed_texts_sparse

    doc = ParsedDocument(filename="quarterly_review_2026Q1.docx")
    doc.paragraphs = [
        ParsedParagraph("Quarterly Performance Review", 1, True),
        ParsedParagraph(
            "Account A-001 in the North region closed Q1 2026 on plan with revenue 120000.",
            1,
            False,
        ),
        ParsedParagraph(
            "Account A-002 in the South region grew the fastest with revenue 150000.",
            2,
            False,
        ),
    ]
    doc.tables = [
        ParsedTable(
            section="Quarterly Performance Review",
            rows=[
                ParsedRow(["Account", "Region", "Quarter", "Revenue"], 1),
                ParsedRow(["A-001", "North", "Q1 2026", "120000"], 1),
                ParsedRow(["A-002", "South", "Q1 2026", "150000"], 1),
            ],
            page_number=1,
        )
    ]

    chunks = chunk_document(doc)
    indexer_mod.ensure_collection(vector_size=1024)
    indexer_mod.index_chunks(chunks)
    return len(chunks)


def test_e2e_roundtrip_finds_indexed_chunk(e2e_qdrant_url, e2e_collection_name):
    n_chunks = _ingest_neutral_corpus(e2e_collection_name, e2e_qdrant_url)
    assert n_chunks > 0

    # Force searcher to use the e2e collection too.
    import retrieval.searcher as searcher_mod
    searcher_mod.QDRANT_URL = e2e_qdrant_url
    searcher_mod.QDRANT_COLLECTION = e2e_collection_name
    searcher_mod._client = None

    from retrieval.searcher import hybrid_search

    results = hybrid_search("Which account had the highest revenue?", top_k=5, enable_rerank=False)
    assert results, "expected at least one hit from the e2e Qdrant"

    # The South / A-002 chunk should be among the top results.
    joined = " ".join(r.text for r in results[:3])
    assert "A-002" in joined or "South" in joined or "150000" in joined


def test_e2e_metadata_filter_restricts_results(e2e_qdrant_url, e2e_collection_name):
    _ingest_neutral_corpus(e2e_collection_name, e2e_qdrant_url)

    import retrieval.searcher as searcher_mod
    searcher_mod.QDRANT_URL = e2e_qdrant_url
    searcher_mod.QDRANT_COLLECTION = e2e_collection_name
    searcher_mod._client = None

    from retrieval.searcher import hybrid_search

    results = hybrid_search(
        "revenue summary", top_k=5, enable_rerank=False,
        filters={"source_file": "quarterly_review_2026Q1.docx"},
    )
    # We only indexed one source file, so every hit must come from it.
    assert results
    for r in results:
        assert r.source_file == "quarterly_review_2026Q1.docx"
