# knowledge-rag/tests/test_orchestrate_node.py
from unittest.mock import patch
from langgraph.types import Send
from agent.nodes.orchestrate import orchestrate_node, dispatch_workers, _DEFAULT_PLAN_SYSTEM
from tests.conftest import make_agent_state


def make_state(question="What was the yield?"):
    return make_agent_state(question=question, question_type="")


def test_orchestrate_classifies_lookup():
    state = make_state("What was the yield of Entry 3?")
    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.side_effect = [
            "lookup",
            '[{"task": "yield Entry 3", "agent_type": "lookup"}]',
        ]
        result = orchestrate_node(state)
    classify_call = mock_llm.return_value.complete.call_args_list[0]
    plan_call = mock_llm.return_value.complete.call_args_list[1]
    assert "timeout" in classify_call.kwargs
    assert "timeout" in plan_call.kwargs
    assert result["question_type"] == "lookup"
    assert len(result["sub_tasks"]) == 1
    assert result["sub_tasks"][0]["agent_type"] == "lookup"


def test_orchestrate_classifies_comparison():
    state = make_state("Compare catalyst loadings across all nitration projects")
    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.side_effect = [
            "comparison",
            '[{"task": "search catalyst loading nitration", "agent_type": "comparison"}, '
            '{"task": "palladium loading comparison", "agent_type": "comparison"}]',
        ]
        result = orchestrate_node(state)
    assert result["question_type"] == "comparison"
    assert len(result["sub_tasks"]) >= 2


def test_orchestrate_falls_back_on_bad_json():
    state = make_state("What is the yield?")
    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.side_effect = ["lookup", "not valid json at all"]
        result = orchestrate_node(state)
    assert result["sub_tasks"] == [{"task": "What is the yield?", "agent_type": "lookup"}]


def test_orchestrate_invalid_type_defaults_to_lookup():
    state = make_state("Some question")
    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.side_effect = [
            "invalid_type",
            '[{"task": "some query", "agent_type": "lookup"}]',
        ]
        result = orchestrate_node(state)
    assert result["question_type"] == "lookup"


def test_dispatch_workers_returns_send_list():
    state = make_state()
    state["sub_tasks"] = [
        {"task": "yield query", "agent_type": "lookup"},
        {"task": "compare yields", "agent_type": "comparison"},
    ]
    sends = dispatch_workers(state)
    assert len(sends) == 2
    assert all(isinstance(s, Send) for s in sends)
    assert sends[0].node == "worker"
    assert sends[0].arg["current_sub_task"] == "yield query"
    assert sends[1].arg["current_agent_type"] == "comparison"


def test_dispatch_workers_falls_back_when_no_subtasks():
    state = make_state("What is the yield?")
    state["sub_tasks"] = []
    state["question_type"] = "lookup"
    sends = dispatch_workers(state)
    assert len(sends) == 1
    assert sends[0].arg["current_sub_task"] == "What is the yield?"
    assert sends[0].arg["current_agent_type"] == "lookup"


def test_query_rewrite_expands_subtasks_when_enabled(monkeypatch):
    """QUERY_REWRITE=1 replaces each planned sub_task with paraphrases."""
    monkeypatch.setenv("QUERY_REWRITE", "1")
    monkeypatch.setenv("QUERY_REWRITE_MAX_VARIANTS", "3")

    state = make_state(question="What is the policy?")
    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_llm:
        mock_complete = mock_llm.return_value.complete
        # 1st call: classify → "lookup". 2nd: plan → 1 sub_task. 3rd: rewrites.
        mock_complete.side_effect = [
            "lookup",
            '[{"task": "policy details", "agent_type": "lookup"}]',
            '["company policy summary", "policy guidelines overview"]',
        ]
        result = orchestrate_node(state)

    tasks = [t["task"] for t in result["sub_tasks"]]
    assert "policy details" in tasks
    assert "company policy summary" in tasks
    assert "policy guidelines overview" in tasks
    # All carry the original agent_type.
    assert all(t["agent_type"] == "lookup" for t in result["sub_tasks"])


def test_query_rewrite_disabled_by_default(monkeypatch):
    monkeypatch.delenv("QUERY_REWRITE", raising=False)
    state = make_state(question="What is the policy?")
    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.side_effect = [
            "lookup",
            '[{"task": "policy details", "agent_type": "lookup"}]',
        ]
        result = orchestrate_node(state)

    # Only the classify + plan calls happened — no rewrite call.
    assert mock_llm.return_value.complete.call_count == 2
    assert [t["task"] for t in result["sub_tasks"]] == ["policy details"]


def test_query_rewrite_caps_at_max_subtasks(monkeypatch):
    """Big plans + N variants must not explode beyond _QUERY_REWRITE_MAX_SUBTASKS."""
    monkeypatch.setenv("QUERY_REWRITE", "1")
    monkeypatch.setenv("QUERY_REWRITE_MAX_VARIANTS", "3")
    monkeypatch.setenv("QUERY_REWRITE_MAX_SUBTASKS", "4")

    state = make_state(question="big plan")
    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_llm:
        # 4-task comparison plan -> 4*3 = 12 expanded; capped at 4.
        mock_llm.return_value.complete.side_effect = [
            "comparison",
            '[{"task": "a", "agent_type": "comparison"},'
            ' {"task": "b", "agent_type": "comparison"},'
            ' {"task": "c", "agent_type": "comparison"},'
            ' {"task": "d", "agent_type": "comparison"}]',
            # rewrite responses repeat — each returns 2 paraphrases
            '["a1", "a2"]', '["b1", "b2"]', '["c1", "c2"]', '["d1", "d2"]',
        ]
        result = orchestrate_node(state)

    assert len(result["sub_tasks"]) == 4  # capped


def test_query_rewrite_falls_back_on_bad_json(monkeypatch):
    """If the rewrite LLM returns garbage, keep the original sub_task."""
    monkeypatch.setenv("QUERY_REWRITE", "1")

    state = make_state(question="What is the policy?")
    with patch("agent.nodes.orchestrate.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.side_effect = [
            "lookup",
            '[{"task": "original task", "agent_type": "lookup"}]',
            "no json here, totally broken",
        ]
        result = orchestrate_node(state)

    assert [t["task"] for t in result["sub_tasks"]] == ["original task"]


def test_plan_system_contains_few_shot_examples():
    """Smoke test: the default plan prompt must carry the structural pieces the
    orchestrator depends on (examples header, JSON sub-task schema, one example
    per question type). Domain-specific phrasing belongs in a domain pack
    (see `domain/examples/`), so the default prompt is intentionally generic.
    """
    assert "Examples" in _DEFAULT_PLAN_SYSTEM, "must contain few-shot examples header"
    assert '"task"' in _DEFAULT_PLAN_SYSTEM and '"agent_type"' in _DEFAULT_PLAN_SYSTEM, (
        "must show the JSON sub-task schema"
    )
    for question_type in ("lookup", "comparison", "trend", "reasoning"):
        assert question_type in _DEFAULT_PLAN_SYSTEM, f"{question_type} example must be present"
