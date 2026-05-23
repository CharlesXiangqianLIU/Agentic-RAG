# retrieval/reranker.py
import logging
import threading
from FlagEmbedding import FlagReranker
from retrieval.searcher import SearchResult
from config import RERANKER_MODEL, RERANK_TOP_K, RERANK_MIN_SCORE

logger = logging.getLogger(__name__)

_reranker: FlagReranker | None = None
# Serialise reranker calls — same Rust tokenizer thread-safety issue as embedder
_reranker_lock = threading.Lock()


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
    return _reranker


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = RERANK_TOP_K,
    min_score: float = RERANK_MIN_SCORE,
) -> list[SearchResult]:
    """Re-score and re-rank search results using cross-encoder model."""
    if not results:
        return []
    pairs = [[query, r.text] for r in results]
    with _reranker_lock:
        scores = _get_reranker().compute_score(pairs, normalize=True)
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    filtered = [(s, r) for s, r in ranked if s >= min_score]
    if not filtered and ranked:
        logger.warning(
            "[RERANK] All %d results scored below threshold %.2f (best=%.3f) — returning top-%d unfiltered",
            len(ranked), min_score, ranked[0][0], min(top_k, len(ranked)),
        )
        filtered = ranked  # fall back to unfiltered so evidence_map is never empty
    for score, result in filtered[:top_k]:
        result.score = score  # update with reranker score
    return [r for _, r in filtered[:top_k]]
