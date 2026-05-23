# knowledge-rag/tests/test_worker_node.py
import time
from unittest.mock import patch, MagicMock
import pytest

import agent.nodes.worker as worker_mod
from agent.nodes.worker import worker_node
from domain.loader import DomainPack
from tests.conftest import make_agent_state


@pytest.fixture(autouse=True)
def _domain_fields(monkeypatch):
    """Worker uses domain pack fields for comparison/trend hints. Inject a
    chemistry-flavoured field list so the original chemistry-tuned tests
    continue to exercise the field-hint logic. Tests that need an empty
    pack can still override.
    """
    pack = DomainPack(
        abbreviations={"DCM": "Dichloromethane", "THF": "Tetrahydrofuran"},
        fields=["yield", "temperature", "catalyst", "solvent", "time", "loading"],
    )
    monkeypatch.setattr(worker_mod, "get_domain_pack", lambda: pack)
    yield pack


def make_worker_state(sub_task, agent_type, question="test question"):
    return make_agent_state(
        question=question,
        question_type=agent_type,
        current_sub_task=sub_task,
        current_agent_type=agent_type,
    )


def make_chunk(text="87%", attribution="[Source: test.docx | Page 1 | Section: S1]"):
    return {"text": text, "attribution": attribution, "score": 0.9, "payload": {}}


def test_worker_lookup_calls_search_reports():
    state = make_worker_state("yield Entry 3", "lookup")
    with patch("agent.nodes.worker.search_reports") as mock_search:
        mock_search.invoke.return_value = [make_chunk()]
        result = worker_node(state)
    # _invoke_args omits 'filters' when None — assert just the query payload.
    mock_search.invoke.assert_called_once_with({"query": "yield Entry 3"})
    assert len(result["worker_results"]) == 1
    assert result["worker_results"][0]["agent_type"] == "lookup"
    assert result["worker_results"][0]["chunks"] == [make_chunk()]


def test_worker_comparison_calls_compare_across_reports():
    state = make_worker_state("compare catalysts", "comparison")
    with patch("agent.nodes.worker.compare_across_reports") as mock_compare, \
         patch("agent.nodes.worker.extract_structured_data") as mock_extract:
        mock_compare.invoke.return_value = [
            {"source": "a.docx", "entries": [make_chunk()]}
        ]
        mock_extract.return_value = []
        result = worker_node(state)
    mock_compare.invoke.assert_called_once()
    assert result["worker_results"][0]["agent_type"] == "comparison"
    assert len(result["worker_results"][0]["chunks"]) >= 1


def test_worker_comparison_appends_structured_extraction():
    state = make_worker_state("compare yields", "comparison")
    with patch("agent.nodes.worker.compare_across_reports") as mock_compare, \
         patch("agent.nodes.worker.extract_structured_data") as mock_extract:
        mock_compare.invoke.return_value = [{"source": "a.docx", "entries": [make_chunk()]}]
        mock_extract.return_value = [
            {"field": "yield", "value": "87", "unit": "%", "attribution": "[Source: a.docx]"}
        ]
        result = worker_node(state)
    texts = [c["text"] for c in result["worker_results"][0]["chunks"]]
    assert any("yield=87%" in t for t in texts)


def test_worker_trend_appends_statistical_summary():
    state = make_worker_state("temperature trend", "trend")
    with patch("agent.nodes.worker.search_reports") as mock_search, \
         patch("agent.nodes.worker.statistical_summary") as mock_stats:
        mock_search.invoke.return_value = [make_chunk()]
        mock_stats.return_value = "Statistical summary for 'temperature': n=3, min=70, max=90, mean=80.0, trend=stable"
        result = worker_node(state)
    mock_stats.assert_called_once()
    call_chunks, call_metric, call_independent = mock_stats.call_args.args
    assert call_metric == "temperature"
    assert call_independent == ""
    assert call_chunks[0] == make_chunk()
    assert result["worker_results"][0]["agent_type"] == "trend"
    texts = [c["text"] for c in result["worker_results"][0]["chunks"]]
    assert any("Statistical summary" in t for t in texts)


def test_worker_trend_skips_summary_without_metric_hint():
    state = make_worker_state("DCM usage trend", "trend")
    with patch("agent.nodes.worker.search_reports") as mock_search, \
         patch("agent.nodes.worker.statistical_summary") as mock_stats:
        mock_search.invoke.return_value = [make_chunk()]
        mock_stats.return_value = ""
        result = worker_node(state)
    mock_stats.assert_called_once()
    call_chunks, call_metric, call_independent = mock_stats.call_args.args
    assert call_metric == ""
    assert call_independent == ""
    assert call_chunks[0] == make_chunk()
    assert result["worker_results"][0]["chunks"] == [make_chunk()]


def test_worker_trend_extracts_metric_and_independent_variable_from_question():
    state = make_worker_state(
        "reaction time yield 6 hours optimization",
        "trend",
        question="What is the trend in yield as reaction time increases from 6 to 24 hours?",
    )
    with patch("agent.nodes.worker.search_reports") as mock_search, \
         patch("agent.nodes.worker.statistical_summary") as mock_stats:
        mock_search.invoke.return_value = [make_chunk()]
        mock_stats.return_value = "Statistical summary"
        worker_node(state)
    _, call_metric, call_independent = mock_stats.call_args.args
    assert call_metric == "yield"
    assert call_independent == "time"


def test_worker_reasoning_calls_multi_hop_search():
    state = make_worker_state("why did yield drop", "reasoning", question="causal question")
    with patch("agent.nodes.worker.multi_hop_search") as mock_hop:
        mock_hop.return_value = [make_chunk()]
        result = worker_node(state)
    mock_hop.assert_called_once_with("why did yield drop", ["causal question"], filters=None)
    assert result["worker_results"][0]["agent_type"] == "reasoning"


def test_worker_unknown_type_falls_back_to_lookup():
    state = make_worker_state("some query", "unknown_type")
    with patch("agent.nodes.worker.search_reports") as mock_search:
        mock_search.invoke.return_value = [make_chunk()]
        result = worker_node(state)
    mock_search.invoke.assert_called_once()
    assert result["worker_results"][0]["agent_type"] == "unknown_type"


def test_comparison_strategy_passes_field_hint():
    """Test that _comparison_strategy extracts field hints and passes them to compare_across_reports."""
    state = make_worker_state("compare yield across catalysts", "comparison")
    with patch("agent.nodes.worker.compare_across_reports") as mock_compare, \
         patch("agent.nodes.worker.extract_structured_data") as mock_extract:
        mock_compare.invoke.return_value = [{"source": "a.docx", "entries": [make_chunk()]}]
        mock_extract.return_value = []
        result = worker_node(state)
    # Verify compare_across_reports was called with field="yield"
    mock_compare.invoke.assert_called_once()
    call_args = mock_compare.invoke.call_args[0][0]
    assert call_args["field"] == "yield"
    assert result["worker_results"][0]["agent_type"] == "comparison"


def test_run_with_timeout_returns_empty_on_timeout():
    from agent.nodes.worker import _run_with_timeout
    start = time.perf_counter()
    result = _run_with_timeout(lambda: time.sleep(5), timeout=0)
    elapsed = time.perf_counter() - start
    assert result == []
    assert elapsed < 1.0


def test_run_with_timeout_returns_result_on_success():
    from agent.nodes.worker import _run_with_timeout
    result = _run_with_timeout(lambda: [{"text": "ok"}], timeout=5)
    assert result == [{"text": "ok"}]


def test_worker_passes_filters_to_search():
    """Test that metadata_filters from state are forwarded to search_reports."""
    state = make_worker_state("yield Entry 3", "lookup")
    state["metadata_filters"] = {"project_id": "PRJ-031"}
    with patch("agent.nodes.worker.search_reports") as mock_search:
        mock_search.invoke.return_value = [make_chunk()]
        worker_node(state)
    call_args = mock_search.invoke.call_args[0][0]
    assert call_args.get("filters") == {"project_id": "PRJ-031"}


def test_worker_passes_filters_to_reasoning_search():
    state = make_worker_state("why did yield drop", "reasoning", question="causal question")
    state["metadata_filters"] = {"project_id": "PRJ-031"}
    with patch("agent.nodes.worker.multi_hop_search") as mock_hop:
        mock_hop.return_value = [make_chunk()]
        worker_node(state)
    mock_hop.assert_called_once_with(
        "why did yield drop",
        ["causal question"],
        filters={"project_id": "PRJ-031"},
    )
