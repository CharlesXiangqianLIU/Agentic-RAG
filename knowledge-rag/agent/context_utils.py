# agent/context_utils.py
"""Shared context-building utilities used by answer_node and critic_node."""
import tiktoken as _tiktoken

# Re-export from config so tests can monkeypatch
# ``agent.context_utils.MAX_CONTEXT_TOKENS`` and have ``build_context`` pick
# up the override on the very next call.
from config import MAX_CONTEXT_TOKENS  # noqa: F401 — exported for tests + as the default

_CONTEXT_ENCODER = _tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return token count for text using the shared context encoder."""
    if not text:
        return 0
    return len(_CONTEXT_ENCODER.encode(text))


def truncate_text(text: str, max_tokens: int) -> str:
    """Return text truncated to at most max_tokens using the shared encoder."""
    if not text or max_tokens <= 0:
        return ""
    tokens = _CONTEXT_ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _CONTEXT_ENCODER.decode(tokens[:max_tokens]).strip()


def build_context(chunks: list[dict], max_tokens: int | None = None) -> tuple[str, int]:
    """Build context string from chunks, sorted by score, truncated at max_tokens.

    ``max_tokens`` defaults to the module-level ``MAX_CONTEXT_TOKENS``
    (resolved at call time so tests / runtime overrides take effect).
    Returns ``(context_string, num_chunks_included)``.
    """
    if max_tokens is None:
        # Look up via globals() so monkeypatch.setattr can override it.
        max_tokens = globals().get("MAX_CONTEXT_TOKENS", 8000)
    if max_tokens <= 0:
        return "", 0

    sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
    parts = []
    total_tokens = 0
    for chunk in sorted_chunks:
        part = f"{chunk.get('attribution', '')}\n{chunk.get('text', '')}"
        tokens = count_tokens(part)
        if total_tokens + tokens > max_tokens:
            break
        parts.append(part)
        total_tokens += tokens
    return "\n\n".join(parts), len(parts)
