"""Lightweight observability for LangGraph nodes: timing + optional LangSmith tracing."""
import logging
import time
from typing import Callable
from agent.state import AgentState

logger = logging.getLogger(__name__)


def timed_node(name: str, fn: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
    """Wrap a graph node function to print wall-clock timing on each invocation."""
    def wrapper(state: AgentState) -> dict:
        start = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - start
        logger.info("[TIMING] %s: %.2fs", name, elapsed)
        return result
    wrapper.__name__ = name
    wrapper.__doc__ = fn.__doc__
    return wrapper
