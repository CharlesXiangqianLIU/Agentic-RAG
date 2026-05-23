# knowledge-rag/tests/test_retry_search_node.py
from unittest.mock import patch
from agent.nodes.retry_search import retry_search_node
from tests.conftest import make_agent_state


def make_state(critic_issues):
    return make_agent_state(critic_issues=critic_issues, critic_round=1)


def make_chunk():
    return {"text": "result", "attribution": "[Source: a.docx | Page 1 | Section: T1]", "score": 0.9, "payload": {}}


def test_retry_search_appends_worker_results():
    issues = [{"claim": "99%", "issue_type": "unsupported", "retry_query": "yield Entry 3"}]
    state = make_state(issues)
    with patch("agent.nodes.retry_search.search_reports") as mock_search:
        mock_search.invoke.return_value = [make_chunk()]
        result = retry_search_node(state)
    # retry_search shares worker's _invoke_args helper: 'filters' key is omitted when None.
    mock_search.invoke.assert_called_once_with({"query": "yield Entry 3"})
    assert len(result["worker_results"]) == 1
    assert result["worker_results"][0]["agent_type"] == "retry"
    assert result["worker_results"][0]["sub_task"] == "yield Entry 3"


def test_retry_search_handles_multiple_issues():
    issues = [
        {"claim": "99%", "issue_type": "unsupported", "retry_query": "yield Entry 3"},
        {"claim": "palladium", "issue_type": "missing_context", "retry_query": "catalyst type Entry 3"},
    ]
    state = make_state(issues)
    with patch("agent.nodes.retry_search.search_reports") as mock_search:
        mock_search.invoke.return_value = [make_chunk()]
        result = retry_search_node(state)
    assert mock_search.invoke.call_count == 2
    assert len(result["worker_results"]) == 2


def test_retry_search_no_issues_returns_empty():
    result = retry_search_node(make_state([]))
    assert result["worker_results"] == []


def test_retry_search_skips_empty_retry_query():
    issues = [{"claim": "x", "issue_type": "unsupported", "retry_query": ""}]
    state = make_state(issues)
    with patch("agent.nodes.retry_search.search_reports") as mock_search:
        result = retry_search_node(state)
    mock_search.invoke.assert_not_called()
    assert result["worker_results"] == []


def test_retry_search_forwards_filters():
    state = make_state([{"claim": "99%", "issue_type": "unsupported", "retry_query": "yield Entry 3"}])
    state["metadata_filters"] = {"project_id": "PRJ-031"}
    with patch("agent.nodes.retry_search.search_reports") as mock_search:
        mock_search.invoke.return_value = [make_chunk()]
        retry_search_node(state)
    mock_search.invoke.assert_called_once_with(
        {"query": "yield Entry 3", "filters": {"project_id": "PRJ-031"}}
    )


def test_retry_search_deduplicates_queries():
    state = make_state([
        {"claim": "99%", "issue_type": "unsupported", "retry_query": "yield Entry 3"},
        {"claim": "87%", "issue_type": "missing_context", "retry_query": "yield Entry 3"},
    ])
    with patch("agent.nodes.retry_search.search_reports") as mock_search:
        mock_search.invoke.return_value = [make_chunk()]
        result = retry_search_node(state)
    # retry_search shares worker's _invoke_args helper: 'filters' key is omitted when None.
    mock_search.invoke.assert_called_once_with({"query": "yield Entry 3"})
    assert len(result["worker_results"]) == 1
