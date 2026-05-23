# llm/anthropic_provider.py
import logging
import time
from typing import Iterator

import anthropic

from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_TIMEOUT_SECONDS
from llm.base import LLMProvider
from llm.retry import always_retry, status_at_least, with_retry

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = (
    "You are an expert knowledge assistant. "
    "Only state conclusions that are supported by the provided source documents."
)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds

_RETRIABLE = [
    (anthropic.RateLimitError, always_retry),
    (anthropic.APIStatusError, status_at_least(500)),
    (anthropic.APITimeoutError, always_retry),
]


def _with_retry(fn, max_retries: int = _MAX_RETRIES, base_delay: float = _RETRY_BASE_DELAY):
    """Thin wrapper around :func:`llm.retry.with_retry` with Anthropic's rule set."""
    return with_retry(fn, retriable=_RETRIABLE, max_retries=max_retries, base_delay=base_delay)


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or ANTHROPIC_MODEL
        self.client = anthropic.Anthropic(
            api_key=api_key or ANTHROPIC_API_KEY,
            timeout=ANTHROPIC_TIMEOUT_SECONDS,
        )

    def complete(self, messages: list[dict], **kwargs) -> str:
        def _call():
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 4096),
                system=kwargs.get("system", _DEFAULT_SYSTEM),
                messages=messages,
                timeout=kwargs.get("timeout", ANTHROPIC_TIMEOUT_SECONDS),
            )
            return response.content[0].text
        return _with_retry(_call)

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """Stream tokens with retry — but only retry if no tokens have been emitted yet.

        Once any token has been yielded to the caller, an error mid-stream
        must surface immediately; silently retrying would produce a
        corrupted answer.
        """
        for attempt in range(_MAX_RETRIES + 1):
            emitted = False
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=kwargs.get("max_tokens", 4096),
                    system=kwargs.get("system", _DEFAULT_SYSTEM),
                    messages=messages,
                    timeout=kwargs.get("timeout", ANTHROPIC_TIMEOUT_SECONDS),
                ) as s:
                    for token in s.text_stream:
                        emitted = True
                        yield token
                return
            except BaseException as exc:
                predicate = next(
                    (pred for exc_type, pred in _RETRIABLE if isinstance(exc, exc_type)),
                    None,
                )
                if emitted or predicate is None or not predicate(exc) or attempt == _MAX_RETRIES:
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "[RETRY] %s (attempt %d/%d), waiting %.1fs...",
                    type(exc).__name__, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
