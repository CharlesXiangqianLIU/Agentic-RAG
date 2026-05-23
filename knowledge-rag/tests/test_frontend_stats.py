def test_qdrant_collection_config_is_set():
    from config import QDRANT_COLLECTION
    assert isinstance(QDRANT_COLLECTION, str)
    assert len(QDRANT_COLLECTION) > 0
