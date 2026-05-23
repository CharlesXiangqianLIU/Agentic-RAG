import hashlib
import logging
import os
import threading
from functools import lru_cache

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model: BGEM3FlagModel | None = None
# Serialise all model calls — the Rust tokenizer's RefCell is not thread-safe
_model_lock = threading.Lock()


def _detect_device_and_precision() -> tuple[str | None, bool]:
    """Return ``(device, use_fp16)`` for ``BGEM3FlagModel`` construction.

    Resolution order:
      1. Explicit overrides via ``EMBEDDING_DEVICE`` ("cuda", "mps", "cpu")
         and ``EMBEDDING_USE_FP16`` ("1"/"0"). Use these for testing
         specific configurations.
      2. Auto-detect: prefer CUDA → MPS (Apple Silicon) → CPU. Enable
         ``use_fp16`` whenever an accelerator is available; CPU stays fp32
         because fp16 on CPU is slower than fp32.

    Returning ``device=None`` lets FlagEmbedding pick its own default
    (its CPU path), so we only force a device when we have a reason to.
    """
    override = os.getenv("EMBEDDING_DEVICE", "").strip().lower()
    fp16_override = os.getenv("EMBEDDING_USE_FP16")

    if override in ("cuda", "mps", "cpu"):
        device = override
    else:
        device = _autodetect_device()

    if fp16_override is not None:
        use_fp16 = fp16_override == "1"
    else:
        use_fp16 = device in ("cuda", "mps")

    return device, use_fp16


def _autodetect_device() -> str:
    """Best-effort accelerator probe. Falls back to CPU on any error."""
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover — defensive
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover — defensive
        pass
    return "cpu"


def _get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        device, use_fp16 = _detect_device_and_precision()
        kwargs: dict = {"use_fp16": use_fp16}
        if device:
            kwargs["device"] = device
        logger.info(
            "[EMBEDDER] Loading %s on device=%s, fp16=%s",
            EMBEDDING_MODEL, device or "auto", use_fp16,
        )
        _model = BGEM3FlagModel(EMBEDDING_MODEL, **kwargs)
    return _model


def _to_sparse(lexical_weight: dict) -> dict:
    """Convert BGEM3 lexical_weights dict to Qdrant sparse format.

    lexical_weight: {token_str: float} from BGEM3FlagModel
    Returns: {"indices": [int, ...], "values": [float, ...]}
    Uses hash of token string as index (avoids vocabulary management).
    """
    indices = []
    values = []
    for token, weight in lexical_weight.items():
        idx = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little") % (2**31 - 1)
        indices.append(idx)
        values.append(float(weight))
    return {"indices": indices, "values": values}


def embed_texts(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    """Embed a batch of document texts (dense only). Returns (N, D) float32 array, normalized."""
    with _model_lock:
        output = _get_model().encode(texts, batch_size=batch_size, return_dense=True, return_sparse=False,
                                     return_colbert_vecs=False)
    return np.array(output["dense_vecs"], dtype=np.float32)


def embed_texts_sparse(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[dict]:
    """Compute sparse (lexical) embeddings for a batch.
    Returns list of {"indices": [...], "values": [...]} dicts.
    """
    with _model_lock:
        output = _get_model().encode(texts, batch_size=batch_size, return_dense=False, return_sparse=True,
                                     return_colbert_vecs=False)
    return [_to_sparse(lw) for lw in output["lexical_weights"]]


@lru_cache(maxsize=256)
def _embed_query_cached(query: str) -> np.ndarray:
    """Cached dense embedding for a single query string using encode_queries() for
    asymmetric BGE-M3 query encoding. Returns 1-D float32 array."""
    with _model_lock:
        output = _get_model().encode_queries([query], return_dense=True, return_sparse=False,
                                             return_colbert_vecs=False)
    return np.array(output["dense_vecs"][0], dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string (dense). Returns 1-D float32 array, normalized."""
    return _embed_query_cached(query).copy()


@lru_cache(maxsize=256)
def _embed_query_sparse_cached(query: str) -> tuple:
    """Cached sparse embedding for a single query using encode_queries() for
    asymmetric BGE-M3 query encoding. Returns (indices_tuple, values_tuple)."""
    with _model_lock:
        output = _get_model().encode_queries([query], return_dense=False, return_sparse=True,
                                             return_colbert_vecs=False)
    result = _to_sparse(output["lexical_weights"][0])
    return (tuple(result["indices"]), tuple(result["values"]))


def embed_query_sparse(query: str) -> dict:
    """Compute sparse embedding for a single query.
    Returns {"indices": [...], "values": [...]}.
    """
    indices, values = _embed_query_sparse_cached(query)
    return {"indices": list(indices), "values": list(values)}
