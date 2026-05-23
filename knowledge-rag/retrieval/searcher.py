# retrieval/searcher.py
from dataclasses import dataclass
import math
import re as _re
import threading
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    Prefetch, FusionQuery, Fusion, SparseVector,
)
from retrieval.embedder import embed_query, embed_query_sparse
from config import QDRANT_URL, QDRANT_COLLECTION, QDRANT_TIMEOUT_SECONDS, RETRIEVAL_TOP_K
from domain.cache import get_domain_pack

_client: QdrantClient | None = None
# Guards the first-call construction of _client. Without it, two threads
# entering get_client() simultaneously on a cold start can each create a
# QdrantClient — one wins the assignment and the other becomes a leaked
# socket-holding orphan.
_client_lock = threading.Lock()


def _expand_query(query: str) -> str:
    """Expand domain abbreviations in the query for better lexical recall.

    With an empty domain pack this is a no-op.
    """
    for abbr, full in get_domain_pack().abbreviations.items():
        # Only expand if the full form isn't already present
        if _re.search(rf'\b{_re.escape(abbr)}\b', query) and full.lower() not in query.lower():
            query = _re.sub(rf'\b{_re.escape(abbr)}\b', f"{abbr} {full}", query)
    return query


def get_client() -> QdrantClient:
    global _client
    # Fast path: no lock once the singleton exists.
    if _client is not None:
        return _client
    with _client_lock:
        # Double-checked under the lock so a racer that already constructed
        # the client doesn't get a second QdrantClient assigned over the top.
        if _client is None:
            _client = QdrantClient(url=QDRANT_URL, timeout=QDRANT_TIMEOUT_SECONDS)
        return _client


@dataclass
class SearchResult:
    text: str
    source_file: str
    page_number: int
    section: str
    score: float
    payload: dict

    @property
    def attribution(self) -> str:
        return f"[Source: {self.source_file} | Page {self.page_number} | Section: {self.section}]"


def hybrid_search(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    filters: dict | None = None,
    enable_rerank: bool = True,
) -> list[SearchResult]:
    query = _expand_query(query)
    client = get_client()
    dense_vec = embed_query(query)
    sparse = embed_query_sparse(query)

    qdrant_filter = None
    if filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        qdrant_filter = Filter(must=conditions)

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec.tolist(), using="dense", limit=top_k),
            Prefetch(
                query=SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"],
                ),
                using="sparse",
                limit=top_k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
        timeout=max(1, math.ceil(QDRANT_TIMEOUT_SECONDS)),
    )

    results = [
        SearchResult(
            text=p.payload.get("text", ""),
            source_file=p.payload.get("source_file", ""),
            page_number=p.payload.get("page_number", 0),
            section=p.payload.get("section", ""),
            score=p.score,
            payload=p.payload,
        )
        for p in response.points
    ]

    if enable_rerank and results:
        from retrieval.reranker import rerank  # lazy import to avoid circular dependency
        results = rerank(query, results)

    return results
