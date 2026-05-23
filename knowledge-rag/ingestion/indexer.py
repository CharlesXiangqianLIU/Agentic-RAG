# ingestion/indexer.py
import threading
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams, SparseVector,
)
from ingestion.chunker import Chunk
from config import QDRANT_URL, QDRANT_COLLECTION, INGESTION_BATCH_SIZE

# Module-level references so tests can patch `ingestion.indexer.embed_texts` and
# `ingestion.indexer.embed_texts_sparse` without needing retrieval/embedder.py to
# exist at import time; resolved lazily on first call to index_chunks.
try:
    from retrieval.embedder import embed_texts, embed_texts_sparse
except ModuleNotFoundError:
    embed_texts = None  # type: ignore[assignment]
    embed_texts_sparse = None  # type: ignore[assignment]

_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")  # fixed namespace


def _chunk_point_id(chunk: Chunk) -> str:
    """Deterministic UUID5 from chunk identity fields."""
    key = f"{chunk.source_file}|{chunk.page_number}|{chunk.section}|{chunk.text[:120]}"
    return str(uuid.uuid5(_NAMESPACE, key))


_client: QdrantClient | None = None
_collection_ensured: bool = False
_ensure_lock = threading.Lock()


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def ensure_collection(vector_size: int = 1024) -> None:
    """Idempotently create the Qdrant collection.

    If a collection with the configured name already exists, verifies that
    its dense-vector dimension matches ``vector_size`` and raises a
    self-explanatory error otherwise. This catches the easy footgun of
    swapping ``EMBEDDING_MODEL`` (changing dim) without re-creating the
    collection.
    """
    global _collection_ensured
    with _ensure_lock:
        if _collection_ensured:
            return
        client = get_client()
        existing = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())},
            )
        else:
            _assert_collection_dim_matches(client, vector_size)
        _collection_ensured = True


def _assert_collection_dim_matches(client: QdrantClient, expected_dim: int) -> None:
    """Compare the existing collection's dense-vector dim against expected.

    Raises ``RuntimeError`` with a clear remediation hint when the
    embedding model and the on-disk collection disagree on dim.
    """
    try:
        info = client.get_collection(QDRANT_COLLECTION)
    except Exception:
        # If the introspection call fails (older qdrant, transient error),
        # skip the check rather than block ingestion.
        return

    actual_dim: int | None = None
    try:
        dense_cfg = info.config.params.vectors["dense"]
        actual_dim = getattr(dense_cfg, "size", None)
    except (AttributeError, KeyError, TypeError):
        actual_dim = None

    if actual_dim is None or actual_dim == expected_dim:
        return

    raise RuntimeError(
        f"Qdrant collection {QDRANT_COLLECTION!r} was created with dense-vector "
        f"dim={actual_dim} but the configured embedding model produces dim={expected_dim}. "
        "Did you change EMBEDDING_MODEL without re-creating the collection? "
        f"Delete the collection (e.g. via the Qdrant dashboard, or "
        f"`client.delete_collection({QDRANT_COLLECTION!r})`) and re-run ingestion."
    )


def build_payload(chunk: Chunk) -> dict:
    payload = {
        "text": chunk.text,
        "source_file": chunk.source_file,
        "page_number": chunk.page_number,
        "section": chunk.section,
        "chunk_type": chunk.chunk_type,
        "synonyms": chunk.synonyms,
    }
    payload.update(chunk.metadata)
    return payload


def index_chunks(chunks: list[Chunk], batch_size: int = INGESTION_BATCH_SIZE) -> None:
    # embed_texts and embed_texts_sparse are imported at module level (with
    # fallback) so tests can patch ingestion.indexer.embed_texts /
    # ingestion.indexer.embed_texts_sparse without retrieval/embedder.py existing.
    import ingestion.indexer as _self
    _embed = _self.embed_texts
    _embed_sparse = _self.embed_texts_sparse
    if _embed is None:
        from retrieval.embedder import embed_texts as _embed, embed_texts_sparse as _embed_sparse  # type: ignore[assignment]
    global _collection_ensured
    if not _collection_ensured:
        ensure_collection()
        _collection_ensured = True
    client = get_client()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        dense_vectors = _embed(texts)
        sparse_vectors = _embed_sparse(texts)

        points = [
            PointStruct(
                id=_chunk_point_id(batch[j]),
                vector={
                    "dense": list(dense_vectors[j]),
                    "sparse": SparseVector(
                        indices=sparse_vectors[j]["indices"],
                        values=sparse_vectors[j]["values"],
                    ),
                },
                payload=build_payload(batch[j]),
            )
            for j in range(len(batch))
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
