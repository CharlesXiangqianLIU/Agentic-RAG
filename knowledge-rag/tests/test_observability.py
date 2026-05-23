"""Tests for observability module: timing metrics and config."""
import config
from agent.observability import timed_node
from agent.state import AgentState
from tests.conftest import make_agent_state


def make_state():
    return make_agent_state(question="q")


def test_timed_node_returns_correct_output():
    """Wrap a function that returns a dict, call it, assert result is correct."""
    def sample_fn(state: AgentState) -> dict:
        return {"key": "val"}

    wrapped = timed_node("sample", sample_fn)
    state = make_state()
    result = wrapped(state)

    assert result == {"key": "val"}


def test_timed_node_preserves_name():
    """Assert timed_node preserves the function name."""
    def sample_fn(state: AgentState) -> dict:
        return {}

    wrapped = timed_node("mynode", sample_fn)
    assert wrapped.__name__ == "mynode"


def test_timed_node_prints_timing(capsys):
    """Use capsys to capture stdout, assert timing message is printed."""
    def sample_fn(state: AgentState) -> dict:
        return {}

    wrapped = timed_node("mynode", sample_fn)
    state = make_state()
    wrapped(state)

    captured = capsys.readouterr()
    assert "[TIMING] mynode:" in captured.out
    assert captured.out.rstrip().endswith("s")


def test_langsmith_config_defaults():
    """Assert LangSmith config defaults when env vars not set."""
    assert config.LANGSMITH_API_KEY == ""
    assert config.LANGCHAIN_PROJECT == "knowledge-rag"


def test_build_graph_does_not_raise():
    """build_graph() completes without error (LangSmith activation path)."""
    from agent.graph import build_graph
    g = build_graph()
    assert g is not None
