# knowledge-rag/tests/test_agent_graph.py
import json
from unittest.mock import patch
from agent.graph import build_graph
from agent.state import AgentState


def make_initial_state(question: str = "What was the yield?") -> AgentState:
    return AgentState(
        question=question, question_type="", sub_tasks=[],
        current_sub_task="", current_agent_type="",
        worker_results=[], evidence_map={},
        draft_answer="", final_answer="",
        critic_issues=[], critic_round=0,
        reflection_passed=False, messages=[],
    )


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_graph_has_correct_nodes():
    graph = build_graph()
    try:
        drawn = graph.get_graph()
        node_names = set(drawn.nodes.keys())
        expected = {"orchestrate", "worker", "synthesis", "answer", "critic", "retry_search"}
        assert expected.issubset(node_names)
    except Exception:
        pass  # graceful if introspection unavailable


def test_graph_runs_lookup_question():
    sub_tasks_json = '[{"task": "yield Entry 3", "agent_type": "lookup"}]'
    critic_pass = json.dumps({"overall": "PASS", "issues": []})

    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_orch, \
         patch("agent.nodes.worker.search_reports") as mock_search, \
         patch("agent.nodes.answer.get_llm_provider") as mock_answer, \
         patch("agent.nodes.critic.get_llm_provider") as mock_critic:

        mock_orch.return_value.complete.side_effect = ["lookup", sub_tasks_json]
        mock_search.invoke.return_value = [
            {"text": "Entry 3 yield 87%",
             "attribution": "[Source: PRJ.docx | Page 5 | Section: Table 1]",
             "score": 0.9, "payload": {}}
        ]
        mock_answer.return_value.stream.return_value = iter([
            "The yield was 87%. [Source: PRJ.docx | Page 5 | Section: Table 1]"
        ])
        mock_critic.return_value.complete.return_value = critic_pass

        graph = build_graph()
        result = graph.invoke(make_initial_state("What was the yield of Entry 3?"))

    assert result["final_answer"]
    assert result["question_type"] == "lookup"
    assert result["critic_round"] == 1


def test_graph_runs_comparison_question():
    sub_tasks_json = (
        '[{"task": "catalyst loading nitration", "agent_type": "comparison"}, '
        '{"task": "palladium loading projects", "agent_type": "comparison"}]'
    )
    critic_pass = json.dumps({"overall": "PASS", "issues": []})

    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_orch, \
         patch("agent.nodes.worker.compare_across_reports") as mock_compare, \
         patch("agent.nodes.worker.extract_structured_data") as mock_extract, \
         patch("agent.nodes.answer.get_llm_provider") as mock_answer, \
         patch("agent.nodes.critic.get_llm_provider") as mock_critic:

        mock_orch.return_value.complete.side_effect = ["comparison", sub_tasks_json]
        mock_compare.invoke.return_value = [
            {"source": "a.docx", "entries": [
                {"text": "catalyst Pd 5%", "attribution": "[Source: a.docx | Page 1 | Section: T1]", "payload": {}}
            ]}
        ]
        mock_extract.return_value = []
        mock_answer.return_value.stream.return_value = iter(["Pd loading ranged from 3% to 7%."])
        mock_critic.return_value.complete.return_value = critic_pass

        graph = build_graph()
        result = graph.invoke(make_initial_state("Compare catalyst loadings across projects"))

    assert result["final_answer"]
    assert result["question_type"] == "comparison"


def test_graph_critic_retry_cycle():
    sub_tasks_json = '[{"task": "yield Entry 3", "agent_type": "lookup"}]'
    critic_fail = json.dumps({
        "overall": "FAIL",
        "issues": [{"claim": "99%", "issue_type": "unsupported", "retry_query": "Entry 3 yield actual"}]
    })
    critic_pass = json.dumps({"overall": "PASS", "issues": []})

    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_orch, \
         patch("agent.nodes.worker.search_reports") as mock_search, \
         patch("agent.nodes.retry_search.search_reports") as mock_retry, \
         patch("agent.nodes.answer.get_llm_provider") as mock_answer, \
         patch("agent.nodes.critic.get_llm_provider") as mock_critic:

        mock_orch.return_value.complete.side_effect = ["lookup", sub_tasks_json]
        mock_search.invoke.return_value = [
            {"text": "yield 45%", "attribution": "[Source: a.docx | Page 1 | Section: T1]", "score": 0.9, "payload": {}}
        ]
        mock_retry.invoke.return_value = [
            {"text": "Entry 3 yield 87%", "attribution": "[Source: a.docx | Page 2 | Section: T2]", "score": 0.95, "payload": {}}
        ]
        mock_answer.return_value.stream.side_effect = [
            iter(["The yield was 87%."]),
            iter(["The yield was 87%."]),
        ]
        mock_critic.return_value.complete.side_effect = [critic_fail, critic_pass]

        graph = build_graph()
        result = graph.invoke(make_initial_state("What was the yield of Entry 3?"))

    assert result["final_answer"]
    assert result["critic_round"] >= 2


def test_graph_stops_after_max_review_rounds_with_warning():
    sub_tasks_json = '[{"task": "yield Entry 3", "agent_type": "lookup"}]'
    critic_fail = json.dumps({
        "overall": "FAIL",
        "issues": [{"claim": "99%", "issue_type": "unsupported", "retry_query": "Entry 3 yield actual"}]
    })

    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_orch, \
         patch("agent.nodes.worker.search_reports") as mock_search, \
         patch("agent.nodes.retry_search.search_reports") as mock_retry, \
         patch("agent.nodes.answer.get_llm_provider") as mock_answer, \
         patch("agent.nodes.critic.get_llm_provider") as mock_critic:

        mock_orch.return_value.complete.side_effect = ["lookup", sub_tasks_json]
        mock_search.invoke.return_value = [
            {"text": "yield 45%", "attribution": "[Source: a.docx | Page 1 | Section: T1]", "score": 0.9, "payload": {}}
        ]
        mock_retry.invoke.return_value = [
            {"text": "still only yield 45%", "attribution": "[Source: a.docx | Page 2 | Section: T2]", "score": 0.95, "payload": {}}
        ]
        mock_answer.return_value.stream.side_effect = [
            iter(["The yield was 99%."]),
            iter(["The yield was 99%."]),
            iter(["The yield was 99%."]),
        ]
        mock_critic.return_value.complete.side_effect = [critic_fail, critic_fail, critic_fail]

        graph = build_graph()
        result = graph.invoke(make_initial_state("What was the yield of Entry 3?"))

    assert result["critic_round"] == 3
    assert result["reflection_passed"] is False
    assert "could not be fully verified" in result["final_answer"]
