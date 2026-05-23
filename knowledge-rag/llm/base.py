# llm/base.py
from abc import ABC, abstractmethod
from typing import Iterator


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        """Synchronous completion. Returns the full response text."""
        ...

    @abstractmethod
    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """Streaming completion. Yields text chunks."""
        ...
