# llm/deepseek_provider.py
"""
DeepSeek-R1 via vLLM local deployment — interface-complete stub.

To activate when GPU hardware is ready:
1. Run: vllm serve deepseek-ai/DeepSeek-R1-Distill-32B --port 8000
2. Set in .env: LLM_PROVIDER=deepseek
3. No other code changes required.
"""
import logging
from typing import Iterator

import openai
from openai import OpenAI

from config import VLLM_BASE_URL, VLLM_MODEL, VLLM_TIMEOUT_SECONDS
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
    """Convert a separate system prompt into an OpenAI-compatible messages list."""
    if not system:
        return messages
    return [{"role": "system", "content": system}, *messages]


class DeepSeekProvider(LLMProvider):
    def __init__(self, base_url: str = None, model: str = None):
        self.model = model or VLLM_MODEL
        self.client = OpenAI(
            base_url=base_url or VLLM_BASE_URL,
            api_key="not-required",
            timeout=VLLM_TIMEOUT_SECONDS,
        )

    def complete(self, messages: list[dict], **kwargs) -> str:
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=_build_messages(messages, kwargs.get("system")),
                max_tokens=kwargs.get("max_tokens", 4096),
                timeout=kwargs.get("timeout", VLLM_TIMEOUT_SECONDS),
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
                timeout=kwargs.get("timeout", VLLM_TIMEOUT_SECONDS),
            )

        stream = _with_retry(_open_stream)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
