# knowledge-rag/tests/test_state.py
import operator
from agent.state import AgentState, WorkerResult, CriticIssue


def test_worker_result_has_required_fields():
    wr = WorkerResult(sub_task="yield query", agent_type="lookup", chunks=[])
    assert wr["sub_task"] == "yield query"
    assert wr["agent_type"] == "lookup"
    assert wr["chunks"] == []


def test_critic_issue_has_required_fields():
    issue = CriticIssue(claim="99%", issue_type="unsupported", retry_query="yield Entry 3")
    assert issue["claim"] == "99%"
    assert issue["retry_query"] == "yield Entry 3"


def test_agent_state_has_new_fields():
    state = AgentState(
        question="q",
        question_type="",
        sub_tasks=[],
        current_sub_task="",
        current_agent_type="",
        worker_results=[],
        evidence_map={},
        draft_answer="",
        final_answer="",
        critic_issues=[],
        critic_round=0,
        reflection_passed=False,
        messages=[],
    )
    assert "worker_results" in state
    assert "evidence_map" in state
    assert "critic_round" in state
    assert "current_sub_task" in state


def test_worker_results_uses_add_reducer():
    a = [WorkerResult(sub_task="q1", agent_type="lookup", chunks=[])]
    b = [WorkerResult(sub_task="q2", agent_type="trend", chunks=[])]
    merged = operator.add(a, b)
    assert len(merged) == 2
