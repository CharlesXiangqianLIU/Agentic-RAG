# llm/ollama_provider.py
"""
Ollama local LLM provider — uses the OpenAI-compatible API exposed by Ollama.

To activate:
1. Ensure Ollama is running: `ollama serve` (or the Ollama desktop app)
2. Set in .env:
     LLM_PROVIDER=ollama
     OLLAMA_MODEL=gemma4:27b   # or any model listed by `ollama list`
3. No other code changes required.
"""
import logging
from typing import Iterator

import openai
from openai import OpenAI

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS
from llm.base import LLMProvider
from llm.retry import always_retry, status_at_least, with_retry

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0

_RETRIABLE = [
    (openai.RateLimitError, always_retry),
    (openai.APIStatusError, status_at_least(500)),
    (openai.APITimeoutError, always_retry),
]


def _with_retry(fn, max_retries=_MAX_RETRIES, base_delay=_RETRY_BASE_DELAY):
    return with_retry(fn, retriable=_RETRIABLE, max_retries=max_retries, base_delay=base_delay)


def _build_messages(messages: list[dict], system: str | None = None) -> list[dict]:
    """Merge an optional system prompt into an OpenAI-compatible messages list."""
    if not system:
        return messages
    return [{"role": "system", "content": system}, *messages]


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = None, model: str = None):
        self.model = model or OLLAMA_MODEL
        self.client = OpenAI(
            base_url=base_url or OLLAMA_BASE_URL,
            api_key="ollama",          # Ollama ignores the key but the field is required
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        logger.info("[Ollama] Provider initialised — model=%s base_url=%s",
                    self.model, base_url or OLLAMA_BASE_URL)

    def complete(self, messages: list[dict], **kwargs) -> str:
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=_build_messages(messages, kwargs.get("system")),
                max_tokens=kwargs.get("max_tokens", 4096),
                timeout=kwargs.get("timeout", OLLAMA_TIMEOUT_SECONDS),
            )
            return response.choices[0].message.content

        return _with_retry(_call)

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        def _open_stream():
            return self.client.chat.completions.create(
                model=self.model,
                messages=_build_messages(messages, kwargs.get("system")),
                max_tokens=kwargs.get("max_tokens", 4096),
                stream=True,
                timeout=kwargs.get("timeout", OLLAMA_TIMEOUT_SECONDS),
            )

        stream = _with_retry(_open_stream)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
