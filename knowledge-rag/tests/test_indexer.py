# tests/test_indexer.py
from unittest.mock import MagicMock, patch
from ingestion.indexer import build_payload, _chunk_point_id
from ingestion.chunker import Chunk


def make_chunk(**kwargs) -> Chunk:
    defaults = dict(
        text="Entry 1 | Pd(OAc)2 | 80 °C | 12 h | 87%",
        source_file="PRJ-2024-031.docx",
        page_number=12,
        section="Optimization Table",
        chunk_type="table_row",
        synonyms=["Pd(OAc)2", "Palladium(II) acetate"],
        metadata={"project_id": "PRJ-2024-031", "client": "Client_A"},
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


def test_build_payload_includes_required_fields():
    payload = build_payload(make_chunk())
    assert payload["text"] == "Entry 1 | Pd(OAc)2 | 80 °C | 12 h | 87%"
    assert payload["source_file"] == "PRJ-2024-031.docx"
    assert payload["page_number"] == 12
    assert payload["section"] == "Optimization Table"
    assert payload["chunk_type"] == "table_row"
    assert payload["synonyms"] == ["Pd(OAc)2", "Palladium(II) acetate"]


def test_build_payload_merges_metadata():
    payload = build_payload(make_chunk())
    assert payload["project_id"] == "PRJ-2024-031"
    assert payload["client"] == "Client_A"


def test_build_payload_no_metadata():
    chunk = make_chunk(metadata={})
    payload = build_payload(chunk)
    assert payload["source_file"] == "PRJ-2024-031.docx"
    assert "project_id" not in payload


def test_index_chunks_calls_upsert(mock_embedder):
    """index_chunks should call qdrant upsert once per batch."""
    from ingestion.indexer import index_chunks
    chunks = [make_chunk() for _ in range(3)]
    with patch("ingestion.indexer.get_client") as mock_client, \
         patch("ingestion.indexer.embed_texts", return_value=[[0.1]*1024]*3), \
         patch("ingestion.indexer.embed_texts_sparse", return_value=[{"indices": [1], "values": [0.5]}]*3):
        mock_qdrant = MagicMock()
        mock_client.return_value = mock_qdrant
        mock_qdrant.get_collections.return_value.collections = []
        index_chunks(chunks, batch_size=10)
        mock_qdrant.upsert.assert_called_once()


def test_chunk_point_id_is_deterministic():
    chunk = make_chunk()
    id1 = _chunk_point_id(chunk)
    id2 = _chunk_point_id(chunk)
    assert id1 == id2


def test_chunk_point_id_differs_for_different_chunks():
    chunk_a = make_chunk(text="Entry 1 | Pd(OAc)2 | 80 °C | 12 h | 87%")
    chunk_b = make_chunk(text="Entry 2 | Pd/C | 100 °C | 6 h | 72%")
    assert _chunk_point_id(chunk_a) != _chunk_point_id(chunk_b)


import pytest

def test_index_chunks_uses_configurable_batch_size():
    """index_chunks respects the batch_size parameter: 5 chunks at batch_size=2 → 3 upsert calls."""
    from ingestion.indexer import index_chunks
    chunks = [make_chunk(text=f"Entry {i} | reagent | 80 °C | 12 h | 87%") for i in range(5)]
    with patch("ingestion.indexer.get_client") as mock_client, \
         patch("ingestion.indexer.embed_texts", side_effect=lambda texts: [[0.1]*1024]*len(texts)), \
         patch("ingestion.indexer.embed_texts_sparse", side_effect=lambda texts: [{"indices": [1], "values": [0.5]}]*len(texts)):
        mock_qdrant = MagicMock()
        mock_client.return_value = mock_qdrant
        mock_qdrant.get_collections.return_value.collections = []
        index_chunks(chunks, batch_size=2)
        # 5 chunks / batch_size=2 → ceil(5/2) = 3 upsert calls
        assert mock_qdrant.upsert.call_count == 3


@pytest.fixture
def mock_embedder():
    with patch("ingestion.indexer.embed_texts", return_value=[[0.1]*1024]), \
         patch("ingestion.indexer.embed_texts_sparse", return_value=[{"indices": [1], "values": [0.5]}]):
        yield


# ---------------------------------------------------------------------------
# ensure_collection dim sanity check
# ---------------------------------------------------------------------------


def _patch_indexer_state():
    """Reset module-level singleton state so ensure_collection() actually runs."""
    import ingestion.indexer as idx
    idx._client = None
    idx._collection_ensured = False


def test_ensure_collection_matching_dim_does_not_raise():
    """Existing collection with the same dim should pass through silently."""
    _patch_indexer_state()
    mock_q = MagicMock()
    existing = MagicMock()
    existing.name = "knowledge_rag"
    mock_q.get_collections.return_value.collections = [existing]
    info = MagicMock()
    info.config.params.vectors = {"dense": MagicMock(size=1024)}
    mock_q.get_collection.return_value = info

    from ingestion.indexer import ensure_collection
    with patch("ingestion.indexer.get_client", return_value=mock_q), \
         patch("ingestion.indexer.QDRANT_COLLECTION", "knowledge_rag"):
        ensure_collection(vector_size=1024)  # must not raise
    mock_q.create_collection.assert_not_called()


def test_ensure_collection_dim_mismatch_raises():
    """If the on-disk collection has a different dim, raise a friendly error."""
    _patch_indexer_state()
    mock_q = MagicMock()
    existing = MagicMock()
    existing.name = "knowledge_rag"
    mock_q.get_collections.return_value.collections = [existing]
    info = MagicMock()
    # Existing collection has dim 768, but we'll ask for 1024.
    info.config.params.vectors = {"dense": MagicMock(size=768)}
    mock_q.get_collection.return_value = info

    from ingestion.indexer import ensure_collection
    with patch("ingestion.indexer.get_client", return_value=mock_q), \
         patch("ingestion.indexer.QDRANT_COLLECTION", "knowledge_rag"):
        with pytest.raises(RuntimeError, match="dim=768.*dim=1024"):
            ensure_collection(vector_size=1024)


def test_ensure_collection_creates_when_absent():
    """No existing collection -> create it with the requested dim."""
    _patch_indexer_state()
    mock_q = MagicMock()
    mock_q.get_collections.return_value.collections = []

    from ingestion.indexer import ensure_collection
    with patch("ingestion.indexer.get_client", return_value=mock_q), \
         patch("ingestion.indexer.QDRANT_COLLECTION", "knowledge_rag"):
        ensure_collection(vector_size=1024)
    mock_q.create_collection.assert_called_once()


def test_ensure_collection_swallows_introspection_failures():
    """If get_collection() raises, fall through silently rather than block ingestion."""
    _patch_indexer_state()
    mock_q = MagicMock()
    existing = MagicMock()
    existing.name = "knowledge_rag"
    mock_q.get_collections.return_value.collections = [existing]
    mock_q.get_collection.side_effect = RuntimeError("api gone")

    from ingestion.indexer import ensure_collection
    with patch("ingestion.indexer.get_client", return_value=mock_q), \
         patch("ingestion.indexer.QDRANT_COLLECTION", "knowledge_rag"):
        ensure_collection(vector_size=1024)  # must not raise
