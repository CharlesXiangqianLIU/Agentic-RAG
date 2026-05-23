"""Token-level streaming: answer_node + graph_runner.

Two surfaces under test:

* ``answer_node`` forwards each streamed LLM token to a ``_token_sink``
  callable injected into state, while still returning the full
  ``draft_answer`` / ``final_answer`` at completion.
* ``run_graph_streaming`` exposes those tokens to consumers as
  ``("token", str)`` records interleaved with ``("event", ...)`` and
  terminated by exactly one ``("done", merged_state, timed_out)`` item.
"""
from unittest.mock import patch

import pytest

from agent.nodes.answer import answer_node
from frontend.graph_runner import run_graph_streaming, run_graph_with_timeout
from tests.conftest import make_agent_state


# ---------------------------------------------------------------------------
# answer_node forwards tokens to _token_sink
# ---------------------------------------------------------------------------


def _make_evidence_state(**extras):
    evidence = {"k1": {"text": "ground truth.", "attribution": "[Source: a.docx | Page 1 | Section: S1]", "payload": {}}}
    state = make_agent_state(
        question="What does the source say?",
        question_type="lookup",
        evidence_map=evidence,
    )
    state.update(extras)
    return state


def test_answer_node_forwards_tokens_to_sink_when_present():
    captured: list[str] = []
    sink = captured.append

    state = _make_evidence_state(_token_sink=sink)
    with patch("agent.nodes.answer.get_llm_provider") as mock_llm:
        mock_llm.return_value.stream.side_effect = lambda *a, **kw: iter(["The ", "ground ", "truth."])
        result = answer_node(state)

    assert captured == ["The ", "ground ", "truth."]
    assert result["draft_answer"] == "The ground truth."


def test_answer_node_works_without_sink():
    """No _token_sink in state — node still returns the answer."""
    state = _make_evidence_state()
    with patch("agent.nodes.answer.get_llm_provider") as mock_llm:
        mock_llm.return_value.stream.side_effect = lambda *a, **kw: iter(["The ", "ground ", "truth."])
        result = answer_node(state)
    assert result["draft_answer"] == "The ground truth."


def test_answer_node_swallows_sink_errors():
    """If the sink raises, the node must keep streaming and still return."""
    def angry_sink(_token: str) -> None:
        raise RuntimeError("ui crashed")

    state = _make_evidence_state(_token_sink=angry_sink)
    with patch("agent.nodes.answer.get_llm_provider") as mock_llm:
        mock_llm.return_value.stream.side_effect = lambda *a, **kw: iter(["hello"])
        result = answer_node(state)

    assert result["draft_answer"] == "hello"


# ---------------------------------------------------------------------------
# run_graph_streaming
# ---------------------------------------------------------------------------


class _GraphThatStreamsTokens:
    """Minimal graph stub: calls token sink, then emits two node events."""

    def stream(self, initial_state, stream_mode="updates"):
        sink = initial_state.get("_token_sink")
        if sink is not None:
            for tok in ("foo ", "bar"):
                sink(tok)
        yield {"orchestrate": {"question_type": "lookup"}}
        yield {"answer": {"final_answer": "foo bar"}}


class _SilentGraph:
    """Graph stub that emits node events but no tokens."""

    def stream(self, initial_state, stream_mode="updates"):
        del initial_state, stream_mode
        yield {"orchestrate": {"question_type": "lookup"}}
        yield {"answer": {"final_answer": "done"}}


class _SlowSilentGraph:
    """Graph stub that sleeps past the timeout."""

    def stream(self, initial_state, stream_mode="updates"):
        del initial_state, stream_mode
        import time
        time.sleep(2)
        yield {"answer": {"final_answer": "too late"}}


def test_streaming_yields_tokens_events_and_done():
    records = list(run_graph_streaming(_GraphThatStreamsTokens(), {}, timeout_seconds=5))

    tokens = [r[1] for r in records if r[0] == "token"]
    events = [(r[1], r[2]) for r in records if r[0] == "event"]
    done = [r for r in records if r[0] == "done"]

    assert tokens == ["foo ", "bar"]
    assert [name for name, _ in events] == ["orchestrate", "answer"]
    assert len(done) == 1
    _, merged, timed_out = done[0]
    assert merged["final_answer"] == "foo bar"
    assert timed_out is False


def test_streaming_done_record_is_always_last():
    records = list(run_graph_streaming(_GraphThatStreamsTokens(), {}, timeout_seconds=5))
    assert records[-1][0] == "done"
    assert all(r[0] != "done" for r in records[:-1])


def test_streaming_calls_on_event_for_each_node():
    seen: list[str] = []
    list(run_graph_streaming(
        _SilentGraph(), {}, timeout_seconds=5,
        on_event=lambda name, _: seen.append(name),
    ))
    assert seen == ["orchestrate", "answer"]


def test_streaming_timeout_emits_done_with_timed_out_true():
    import time
    start = time.perf_counter()
    records = list(run_graph_streaming(_SlowSilentGraph(), {}, timeout_seconds=1))
    elapsed = time.perf_counter() - start

    assert elapsed < 1.5
    done = [r for r in records if r[0] == "done"]
    assert len(done) == 1
    _, merged, timed_out = done[0]
    assert timed_out is True
    assert merged == {}


def test_streaming_propagates_graph_errors():
    class _BoomGraph:
        def stream(self, initial_state, stream_mode="updates"):
            del initial_state, stream_mode
            raise RuntimeError("graph exploded")
            yield  # pragma: no cover — make this a generator

    with pytest.raises(RuntimeError, match="graph exploded"):
        list(run_graph_streaming(_BoomGraph(), {}, timeout_seconds=5))


# ---------------------------------------------------------------------------
# run_graph_with_timeout wrapper still works
# ---------------------------------------------------------------------------


def test_wrapper_still_returns_merged_state_and_timed_out():
    result, timed_out = run_graph_with_timeout(_SilentGraph(), {}, timeout_seconds=5)
    assert timed_out is False
    assert result["final_answer"] == "done"
    assert result["question_type"] == "lookup"
