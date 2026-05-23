"""Shared exponential-backoff retry helper for LLM providers.

Each provider supplies a list of ``(exception_type, predicate)`` rules.
``predicate(exc)`` returns True if that occurrence is retriable.  Status
errors typically only want to retry on 5xx, while rate-limit and timeout
errors always do — predicates make that distinction declarative instead
of duplicating the same try/except ladder in every provider.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")

log = logging.getLogger(__name__)

#: A list of (exception_type, predicate) pairs that decide whether to retry.
RetryRules = list[tuple[type[BaseException], Callable[[BaseException], bool]]]


def with_retry(
    fn: Callable[[], T],
    *,
    retriable: RetryRules,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Run ``fn`` with exponential backoff on transient errors.

    Retries up to ``max_retries`` times. An exception is retried only if
    ``isinstance(exc, exception_type)`` for some rule AND that rule's
    predicate returns True. The matching rule is searched in order, so
    place more specific exception types earlier.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except BaseException as exc:
            predicate = _match_predicate(exc, retriable)
            if predicate is None or not predicate(exc) or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            log.warning(
                "[RETRY] %s (attempt %d/%d), waiting %.1fs...",
                type(exc).__name__, attempt + 1, max_retries, delay,
            )
            time.sleep(delay)
    # Unreachable: the loop either returns or raises.
    raise RuntimeError("with_retry exhausted without returning or raising")


def _match_predicate(
    exc: BaseException, rules: Iterable[tuple[type[BaseException], Callable[[BaseException], bool]]]
) -> Callable[[BaseException], bool] | None:
    for exc_type, predicate in rules:
        if isinstance(exc, exc_type):
            return predicate
    return None


def status_at_least(threshold: int) -> Callable[[BaseException], bool]:
    """Predicate factory: True when ``exc.status_code >= threshold``.

    Tolerates exceptions that don't expose ``status_code`` (returns False).
    """
    def _predicate(exc: BaseException) -> bool:
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and status >= threshold
    return _predicate


def always_retry(_exc: BaseException) -> bool:
    return True
