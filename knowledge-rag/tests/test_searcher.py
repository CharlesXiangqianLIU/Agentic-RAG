# tests/test_searcher.py
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from retrieval.searcher import hybrid_search, SearchResult, _expand_query, get_client
import retrieval.searcher as searcher_mod
from domain.loader import DomainPack


@pytest.fixture(autouse=True)
def _chemistry_domain_pack(monkeypatch):
    """Inject a small chemistry pack so abbreviation-expansion tests have data.

    Tests that need a different/empty pack can override via monkeypatch.setattr.
    """
    pack = DomainPack(
        abbreviations={
            "DCM": "Dichloromethane",
            "THF": "Tetrahydrofuran",
            "MeOH": "Methanol",
        },
    )
    monkeypatch.setattr(searcher_mod, "get_domain_pack", lambda: pack)
    yield pack


def make_mock_point(text="test", source_file="a.docx", page_number=1, section="S1", score=0.9):
    p = MagicMock()
    p.payload = {
        "text": text,
        "source_file": source_file,
        "page_number": page_number,
        "section": section,
    }
    p.score = score
    return p


@pytest.fixture(autouse=True)
def no_rerank(monkeypatch):
    """Disable BGE reranker in all searcher tests — model not available in CI."""
    monkeypatch.setattr("retrieval.reranker._get_reranker", lambda: None)
    monkeypatch.setattr("retrieval.reranker.rerank", lambda query, results, **kw: results)


def test_search_returns_list_of_search_results():
    with patch("retrieval.searcher.get_client") as mock_client, \
         patch("retrieval.searcher.embed_query", return_value=np.array([0.1] * 1024)), \
         patch("retrieval.searcher.embed_query_sparse", return_value={"indices": [1, 2], "values": [0.8, 0.6]}):
        mock_qdrant = MagicMock()
        mock_client.return_value = mock_qdrant
        mock_qdrant.query_points.return_value.points = [
            make_mock_point(text="Entry 3 | 87%", source_file="PRJ-031.docx",
                           page_number=12, section="Table 1", score=0.95)
        ]
        results = hybrid_search("What is the yield?")
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)


def test_search_result_attribution_format():
    result = SearchResult(
        text="87%",
        source_file="PRJ-031.docx",
        page_number=12,
        section="Table 1",
        score=0.95,
        payload={},
    )
    assert result.attribution == "[Source: PRJ-031.docx | Page 12 | Section: Table 1]"


def test_search_result_fields():
    with patch("retrieval.searcher.get_client") as mock_client, \
         patch("retrieval.searcher.embed_query", return_value=np.array([0.1] * 1024)), \
         patch("retrieval.searcher.embed_query_sparse", return_value={"indices": [1, 2], "values": [0.8, 0.6]}):
        mock_qdrant = MagicMock()
        mock_client.return_value = mock_qdrant
        mock_qdrant.query_points.return_value.points = [
            make_mock_point(text="some text", source_file="doc.docx",
                           page_number=5, section="Intro", score=0.8)
        ]
        results = hybrid_search("query")
        r = results[0]
        assert r.text == "some text"
        assert r.source_file == "doc.docx"
        assert r.page_number == 5
        assert r.section == "Intro"
        assert r.score == 0.8


def test_search_with_filters_passes_filter_to_qdrant():
    with patch("retrieval.searcher.get_client") as mock_client, \
         patch("retrieval.searcher.embed_query", return_value=np.array([0.1] * 1024)), \
         patch("retrieval.searcher.embed_query_sparse", return_value={"indices": [1, 2], "values": [0.8, 0.6]}):
        mock_qdrant = MagicMock()
        mock_client.return_value = mock_qdrant
        mock_qdrant.query_points.return_value.points = []
        hybrid_search("query", filters={"client": "Client_A"})
        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        assert call_kwargs.get("query_filter") is not None
        assert call_kwargs.get("timeout") is not None


def test_expand_query_expands_abbreviation():
    result = _expand_query("DCM solvent")
    assert "Dichloromethane" in result


def test_expand_query_skips_if_full_already_present():
    result = _expand_query("Dichloromethane DCM")
    # "Dichloromethane" should appear exactly once (not doubled)
    assert result.lower().count("dichloromethane") == 1


def test_expand_query_no_change_for_unknown_terms():
    query = "water pH"
    result = _expand_query(query)
    assert result == query


def test_hybrid_search_expands_query_before_embedding():
    captured = {}

    def fake_embed_query(q):
        captured["query"] = q
        return np.array([0.1] * 1024)

    with patch("retrieval.searcher.get_client") as mock_client, \
         patch("retrieval.searcher.embed_query", side_effect=fake_embed_query), \
         patch("retrieval.searcher.embed_query_sparse", return_value={"indices": [1, 2], "values": [0.8, 0.6]}):
        mock_qdrant = MagicMock()
        mock_client.return_value = mock_qdrant
        mock_qdrant.query_points.return_value.points = []
        hybrid_search("DCM")
        assert "Dichloromethane" in captured["query"]


def test_hybrid_search_calls_rerank_when_enabled():
    """When enable_rerank=True (default), rerank() is called with the query and results."""
    captured = {}

    def fake_rerank(query, results, **kw):
        captured["query"] = query
        captured["n_results"] = len(results)
        return results

    with patch("retrieval.searcher.get_client") as mock_client, \
         patch("retrieval.searcher.embed_query", return_value=np.array([0.1] * 1024)), \
         patch("retrieval.searcher.embed_query_sparse", return_value={"indices": [1, 2], "values": [0.8, 0.6]}), \
         patch("retrieval.reranker.rerank", side_effect=fake_rerank):
        mock_qdrant = MagicMock()
        mock_client.return_value = mock_qdrant
        mock_qdrant.query_points.return_value.points = [
            make_mock_point(text="yield 87%", score=0.9)
        ]
        hybrid_search("yield Entry 3")

    assert captured.get("query") == "yield Entry 3"
    assert captured.get("n_results") == 1


def test_hybrid_search_skips_rerank_when_disabled():
    """When enable_rerank=False, rerank() should not be called."""
    with patch("retrieval.searcher.get_client") as mock_client, \
         patch("retrieval.searcher.embed_query", return_value=np.array([0.1] * 1024)), \
         patch("retrieval.searcher.embed_query_sparse", return_value={"indices": [1, 2], "values": [0.8, 0.6]}), \
         patch("retrieval.reranker.rerank") as mock_rerank:
        mock_qdrant = MagicMock()
        mock_client.return_value = mock_qdrant
        mock_qdrant.query_points.return_value.points = [make_mock_point()]
        hybrid_search("query", enable_rerank=False)
    mock_rerank.assert_not_called()


def test_get_client_sets_qdrant_timeout(monkeypatch):
    import retrieval.searcher as searcher

    monkeypatch.setattr(searcher, "_client", None)
    with patch("retrieval.searcher.QdrantClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        result = get_client()
    assert result is mock_client
    assert "timeout" in mock_client_cls.call_args.kwargs
