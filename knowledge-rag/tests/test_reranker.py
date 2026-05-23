# tests/test_reranker.py
from unittest.mock import patch, MagicMock
from retrieval.reranker import rerank
from retrieval.searcher import SearchResult


def make_results():
    return [
        SearchResult("unrelated text about something else", "a.docx", 1, "S1", 0.6, {}),
        SearchResult("Entry 3 | Pd(OAc)2 | 80 °C | 87% yield", "b.docx", 5, "S2", 0.5, {}),
        SearchResult("general introduction paragraph", "c.docx", 2, "S3", 0.4, {}),
    ]


def test_rerank_returns_list():
    mock_reranker = MagicMock()
    mock_reranker.compute_score.return_value = [0.2, 0.9, 0.1]
    with patch("retrieval.reranker._get_reranker", return_value=mock_reranker):
        results = rerank("What is the yield of Entry 3?", make_results())
        assert isinstance(results, list)


def test_rerank_sorts_by_score_descending():
    mock_reranker = MagicMock()
    mock_reranker.compute_score.return_value = [0.2, 0.9, 0.1]
    with patch("retrieval.reranker._get_reranker", return_value=mock_reranker):
        results = rerank("What is the yield of Entry 3?", make_results())
        assert "87%" in results[0].text


def test_rerank_respects_top_k():
    mock_reranker = MagicMock()
    mock_reranker.compute_score.return_value = [0.3, 0.9, 0.5]
    with patch("retrieval.reranker._get_reranker", return_value=mock_reranker):
        results = rerank("query", make_results(), top_k=2)
        assert len(results) == 2


def test_rerank_empty_input():
    results = rerank("query", [])
    assert results == []


def test_rerank_passes_query_document_pairs():
    mock_reranker = MagicMock()
    mock_reranker.compute_score.return_value = [0.5, 0.8, 0.3]
    with patch("retrieval.reranker._get_reranker", return_value=mock_reranker):
        rerank("test query", make_results())
        call_args = mock_reranker.compute_score.call_args
        pairs = call_args[0][0]
        assert len(pairs) == 3
        assert all(p[0] == "test query" for p in pairs)


def test_rerank_filters_below_min_score():
    mock_reranker = MagicMock()
    # scores: result[0]=0.2, result[1]=0.9, result[2]=0.1
    mock_reranker.compute_score.return_value = [0.2, 0.9, 0.1]
    with patch("retrieval.reranker._get_reranker", return_value=mock_reranker):
        results = rerank("query", make_results(), min_score=0.5)
        # Only the 0.9-scored result (result[1]) should pass the threshold
        assert len(results) == 1
        assert "87%" in results[0].text


def test_rerank_min_score_zero_returns_all_top_k():
    mock_reranker = MagicMock()
    mock_reranker.compute_score.return_value = [0.2, 0.9, 0.1]
    with patch("retrieval.reranker._get_reranker", return_value=mock_reranker):
        results = rerank("query", make_results(), top_k=3, min_score=0.0)
        # All 3 results pass the 0.0 threshold, top_k=3 returns all
        assert len(results) == 3


def test_rerank_updates_result_scores():
    mock_reranker = MagicMock()
    mock_reranker.compute_score.return_value = [0.2, 0.9, 0.5]
    with patch("retrieval.reranker._get_reranker", return_value=mock_reranker):
        input_results = make_results()
        results = rerank("query", input_results, top_k=3, min_score=0.0)
        # After rerank, scores should be updated to reranker scores (sorted desc)
        assert results[0].score == 0.9
        assert results[1].score == 0.5
        assert results[2].score == 0.2
