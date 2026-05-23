import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Import module to ensure it's available for patching
import retrieval.embedder


def test_embed_texts_returns_numpy_array():
    mock_model = MagicMock()
    mock_model.encode.return_value = {"dense_vecs": np.array([[0.1] * 1024])}
    with patch("retrieval.embedder._get_model", return_value=mock_model):
        from retrieval.embedder import embed_texts
        result = embed_texts(["Entry 1 | Pd(OAc)2 | 80 °C | 87%"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1024)


def test_embed_batch_returns_correct_shape():
    mock_model = MagicMock()
    mock_model.encode.return_value = {"dense_vecs": np.array([[0.1] * 1024] * 3)}
    with patch("retrieval.embedder._get_model", return_value=mock_model):
        from retrieval.embedder import embed_texts
        result = embed_texts(["a", "b", "c"])
        assert result.shape[0] == 3


def test_embed_query_returns_1d_array():
    mock_model = MagicMock()
    mock_model.encode_queries.return_value = {"dense_vecs": np.array([[0.5] * 1024])}
    with patch("retrieval.embedder._get_model", return_value=mock_model):
        from retrieval.embedder import embed_query
        vec = embed_query("What is the yield?")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(vec) == 1024


def test_embed_query_uses_encode_queries():
    """embed_query() uses encode_queries() for asymmetric BGE-M3 query encoding."""
    mock_model = MagicMock()
    mock_model.encode_queries.return_value = {"dense_vecs": np.array([[0.1] * 1024])}
    with patch("retrieval.embedder._get_model", return_value=mock_model):
        from retrieval.embedder import embed_query
        embed_query("test query")
        mock_model.encode_queries.assert_called_once()


def test_embed_texts_sparse_returns_indices_and_values():
    mock_model = MagicMock()
    mock_model.encode.return_value = {"lexical_weights": [{"reaction": 0.8, "yield": 0.6}]}
    with patch("retrieval.embedder._get_model", return_value=mock_model):
        from retrieval.embedder import embed_texts_sparse
        result = embed_texts_sparse(["some text"])
        assert len(result) == 1
        assert "indices" in result[0]
        assert "values" in result[0]
        assert len(result[0]["indices"]) == 2
        assert all(isinstance(i, int) for i in result[0]["indices"])


def test_embed_query_sparse_returns_dict():
    mock_model = MagicMock()
    mock_model.encode_queries.return_value = {"lexical_weights": [{"catalyst": 0.9}]}
    with patch("retrieval.embedder._get_model", return_value=mock_model):
        from retrieval.embedder import embed_query_sparse
        result = embed_query_sparse("catalyst test")
        assert "indices" in result
        assert "values" in result


def test_embed_query_is_cached(monkeypatch):
    """embed_query should not call the model a second time for the same query."""
    from retrieval.embedder import _embed_query_cached
    _embed_query_cached.cache_clear()  # ensure clean state

    with patch("retrieval.embedder._get_model") as mock_model:
        mock_encode_queries = MagicMock(return_value={"dense_vecs": [[0.1] * 1024]})
        mock_model.return_value.encode_queries = mock_encode_queries

        from retrieval.embedder import embed_query
        # Call twice with same query
        embed_query("test query")
        embed_query("test query")

        # encode_queries should only be called once due to LRU cache
        assert mock_encode_queries.call_count == 1


def test_embed_query_sparse_is_cached(monkeypatch):
    """embed_query_sparse should not call the model a second time for the same query."""
    from retrieval.embedder import _embed_query_sparse_cached
    _embed_query_sparse_cached.cache_clear()  # ensure clean state

    with patch("retrieval.embedder._get_model") as mock_model:
        mock_encode_queries = MagicMock(return_value={"lexical_weights": [{"reaction": 0.8, "yield": 0.6}]})
        mock_model.return_value.encode_queries = mock_encode_queries

        from retrieval.embedder import embed_query_sparse
        # Call twice with same query
        result1 = embed_query_sparse("sparse test query")
        result2 = embed_query_sparse("sparse test query")

        # encode_queries should only be called once due to LRU cache
        assert mock_encode_queries.call_count == 1
        # Results should be equal dicts
        assert result1["indices"] == result2["indices"]
        assert result1["values"] == result2["values"]


def test_embed_texts_passes_batch_size_to_model():
    """embed_texts() should forward the batch_size argument to model.encode()."""
    mock_model = MagicMock()
    mock_model.encode.return_value = {"dense_vecs": np.array([[0.1] * 1024] * 2)}
    with patch("retrieval.embedder._get_model", return_value=mock_model):
        from retrieval.embedder import embed_texts
        embed_texts(["a", "b"], batch_size=16)
        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("batch_size") == 16


def test_embed_texts_sparse_passes_batch_size_to_model():
    """embed_texts_sparse() should forward the batch_size argument to model.encode()."""
    mock_model = MagicMock()
    mock_model.encode.return_value = {"lexical_weights": [{"reaction": 0.8}, {"yield": 0.6}]}
    with patch("retrieval.embedder._get_model", return_value=mock_model):
        from retrieval.embedder import embed_texts_sparse
        embed_texts_sparse(["a", "b"], batch_size=16)
        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("batch_size") == 16


def test_embed_query_returns_copy():
    """embed_query should return a copy so callers cannot mutate the cache."""
    from retrieval.embedder import _embed_query_cached
    _embed_query_cached.cache_clear()

    with patch("retrieval.embedder._get_model") as mock_model:
        mock_model.return_value.encode_queries.return_value = {"dense_vecs": [[0.1] * 1024]}

        from retrieval.embedder import embed_query
        vec1 = embed_query("mutation test")
        vec1[:] = 0.0  # mutate the returned array

        vec2 = embed_query("mutation test")
        # Cache should still hold original values, not the mutated ones
        assert not np.all(vec2 == 0.0)
