# tests/test_frontend_timeout.py
"""Tests for graph-level timeout in the frontend."""
import time

from frontend.graph_runner import run_graph_with_timeout


def test_graph_timeout_seconds_is_positive_int():
    """GRAPH_TIMEOUT_SECONDS must be a positive integer."""
    from config import GRAPH_TIMEOUT_SECONDS

    assert isinstance(GRAPH_TIMEOUT_SECONDS, int)
    assert GRAPH_TIMEOUT_SECONDS > 0


class _SlowGraph:
    def stream(self, initial_state, stream_mode="updates"):
        del initial_state, stream_mode
        time.sleep(2)
        yield {"answer": {"final_answer": "too late"}}


class _FastGraph:
    def stream(self, initial_state, stream_mode="updates"):
        del initial_state, stream_mode
        yield {"orchestrate": {"question_type": "lookup"}}
        yield {"answer": {"final_answer": "synthesis complete"}}


def test_graph_timeout_returns_quickly():
    """Slow graph execution should return on timeout without waiting for the worker to finish."""
    start = time.perf_counter()
    result, timed_out = run_graph_with_timeout(_SlowGraph(), {}, timeout_seconds=1)
    elapsed = time.perf_counter() - start

    assert timed_out
    assert result == {}
    assert elapsed < 1.5


def test_graph_runner_merges_results_and_calls_on_event():
    """Completed graph execution should merge node outputs and surface event callbacks."""
    seen = []

    def _on_event(node_name, node_output):
        seen.append((node_name, node_output))

    result, timed_out = run_graph_with_timeout(
        _FastGraph(),
        {},
        timeout_seconds=5,
        on_event=_on_event,
    )

    assert not timed_out
    assert result["final_answer"] == "synthesis complete"
    assert result["question_type"] == "lookup"
    assert [name for name, _ in seen] == ["orchestrate", "answer"]
