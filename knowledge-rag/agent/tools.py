# agent/tools.py
from __future__ import annotations
from typing import Optional
from langchain_core.tools import tool
from retrieval.searcher import hybrid_search
from retrieval.reranker import rerank


@tool
def search_reports(query: str, filters: Optional[dict] = None) -> list[dict]:
    """Search the knowledge base for relevant passages.

    Args:
        query: The search query string.
        filters: Optional dict of metadata filters (e.g. {"category": "policy", "doc_type": "handbook"}).

    Returns:
        List of dicts with keys: text, attribution, score, payload.
    """
    # enable_rerank=False: hybrid_search provides bi-encoder recall; rerank() below
    # applies the cross-encoder with default RERANK_TOP_K / RERANK_MIN_SCORE.
    results = hybrid_search(query, filters=filters, enable_rerank=False)
    ranked = rerank(query, results)
    return [
        {
            "text": r.text,
            "attribution": r.attribution,
            "score": r.score,
            "payload": r.payload,
        }
        for r in ranked
    ]


@tool
def compare_across_reports(query: str, field: str = "", filters: Optional[dict] = None) -> list[dict]:
    """Compare a specific field across multiple documents.

    Args:
        query: The comparison query (e.g. "data retention policy across 2023 and 2024 handbooks").
        field: The field to compare. Optional hint that depends on the domain
            (e.g. "duration", "value", "owner"). Empty string disables hinting.
        filters: Optional dict of metadata filters.

    Returns:
        List of dicts grouped by source file: {"source": filename, "entries": [...]}.
    """
    # Fetch broad candidate set (top 40) without internal reranking so all
    # candidates are available for diversity-aware cross-file comparison.
    results = hybrid_search(query, top_k=40, filters=filters, enable_rerank=False)
    ranked = rerank(query, results, top_k=20)
    by_file: dict[str, list] = {}
    for r in ranked:
        by_file.setdefault(r.source_file, []).append({
            "text": r.text,
            "attribution": r.attribution,
            "payload": r.payload,
        })
    return [{"source": k, "entries": v} for k, v in by_file.items()]
