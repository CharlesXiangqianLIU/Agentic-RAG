# knowledge-rag/agent/nodes/synthesis.py
import hashlib
import logging
import os

from agent.state import AgentState
from config import SYNTHESIS_TOP_K

_logger = logging.getLogger(__name__)


# Default threshold above which two chunks are considered semantic duplicates
# and the lower-scored one is dropped. 0.92 is conservative — paraphrased
# copies merge but distinct facts stay separate. Read at call time so tests
# (and operators) can flip the toggle without re-importing the module.
_DEFAULT_THRESHOLD = "0.92"


def _semantic_dedup_enabled() -> bool:
    return os.getenv("SEMANTIC_DEDUP", "1") != "0"


def _semantic_dedup_threshold() -> float:
    return float(os.getenv("SEMANTIC_DEDUP_THRESHOLD", _DEFAULT_THRESHOLD))


def _chunk_key(chunk: dict) -> str:
    """Stable dedup key: SHA-256 of (source_file + page_number + first 120 chars of text)."""
    source = chunk.get("payload", {}).get("source_file", chunk.get("attribution", ""))
    page = str(chunk.get("payload", {}).get("page_number", ""))
    text_prefix = chunk.get("text", "")[:120]
    raw = f"{source}|{page}|{text_prefix}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def synthesis_node(state: AgentState) -> dict:
    all_chunks: list[dict] = []
    for wr in state.get("worker_results", []):
        all_chunks.extend(wr.get("chunks", []))

    # First pass: exact-identity dedup via hash key.
    seen: set[str] = set()
    unique: list[dict] = []
    for c in all_chunks:
        key = _chunk_key(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # Sort by score descending and cap at SYNTHESIS_TOP_K before the
    # (more expensive) semantic dedup pass so we only embed the candidates
    # that will actually make it into the context.
    unique.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    capped = unique[:SYNTHESIS_TOP_K]
    if len(unique) > SYNTHESIS_TOP_K:
        _logger.info(
            "[SYNTHESIS] Capped evidence_map from %d to %d chunks", len(unique), SYNTHESIS_TOP_K
        )

    if _semantic_dedup_enabled() and len(capped) > 1:
        capped = _semantic_dedup(capped, threshold=_semantic_dedup_threshold())

    evidence_map = {_chunk_key(c): c for c in capped}
    return {"evidence_map": evidence_map}


def _semantic_dedup(chunks: list[dict], threshold: float) -> list[dict]:
    """Drop chunks whose embedding is within `threshold` cosine of a kept chunk.

    Chunks are processed in input order (which the caller has already
    sorted by descending score). The first occurrence of each cluster
    survives; later near-duplicates are dropped.

    Failures in embedding or numeric stack fall through gracefully —
    semantic dedup is an optimisation, not a correctness requirement.
    """
    try:
        import numpy as np
        from retrieval.embedder import embed_texts
    except Exception as exc:
        _logger.warning("[SYNTHESIS] semantic dedup unavailable (%s); using exact dedup only", exc)
        return chunks

    texts = [c.get("text", "") for c in chunks]
    try:
        vectors = np.asarray(embed_texts(texts), dtype="float32")
    except Exception as exc:
        _logger.warning("[SYNTHESIS] semantic dedup embedding failed (%s); using exact dedup only", exc)
        return chunks

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised = vectors / norms
    sims = normalised @ normalised.T  # [N, N] cosine

    kept_indices: list[int] = []
    for i in range(len(chunks)):
        is_duplicate = any(sims[i, j] >= threshold for j in kept_indices)
        if not is_duplicate:
            kept_indices.append(i)

    dropped = len(chunks) - len(kept_indices)
    if dropped > 0:
        _logger.info(
            "[SYNTHESIS] semantic dedup merged %d/%d chunks (threshold=%.2f)",
            dropped, len(chunks), threshold,
        )
    return [chunks[i] for i in kept_indices]
