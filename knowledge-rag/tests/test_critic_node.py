# knowledge-rag/tests/test_critic_node.py
import json
from unittest.mock import patch
from agent.nodes.critic import critic_node
from tests.conftest import make_agent_state


def make_state(draft_answer, evidence_map, critic_round=0):
    return make_agent_state(draft_answer=draft_answer, evidence_map=evidence_map, critic_round=critic_round)


def test_critic_pass():
    state = make_state(
        draft_answer="The yield was 87%.",
        evidence_map={"k1": {"text": "Entry 3 yield was 87%", "attribution": ""}},
    )
    with patch("agent.nodes.critic.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = '{"overall": "PASS", "issues": []}'
        result = critic_node(state)
    assert "timeout" in mock_llm.return_value.complete.call_args.kwargs
    assert result["reflection_passed"] is True
    assert result["critic_round"] == 1
    assert result["critic_issues"] == []


def test_critic_fail_with_issues():
    state = make_state(
        draft_answer="The yield was 99%.",
        evidence_map={"k1": {"text": "Entry 3 yield was 45%", "attribution": ""}},
    )
    with patch("agent.nodes.critic.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = json.dumps({
            "overall": "FAIL",
            "issues": [{
                "claim": "yield was 99%",
                "issue_type": "unsupported",
                "retry_query": "yield Entry 3",
            }]
        })
        result = critic_node(state)
    assert result["reflection_passed"] is False
    assert len(result["critic_issues"]) == 1
    assert result["critic_issues"][0]["retry_query"] == "yield Entry 3"


def test_critic_forces_pass_at_round_3():
    state = make_state(
        draft_answer="The yield was 99%.",
        evidence_map={"k1": {"text": "yield 45%", "attribution": ""}},
        critic_round=2,  # this call will be round 3
    )
    state["final_answer"] = "The yield was 99%."
    with patch("agent.nodes.critic.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = json.dumps({
            "overall": "FAIL",
            "issues": [{"claim": "x", "issue_type": "unsupported", "retry_query": "q"}]
        })
        result = critic_node(state)
    assert result["reflection_passed"] is False
    assert result["critic_round"] == 3
    assert "could not be fully verified" in result["final_answer"]


def test_critic_increments_round():
    state = make_state("answer", {"k": {"text": "t", "attribution": ""}}, critic_round=1)
    with patch("agent.nodes.critic.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = '{"overall": "PASS", "issues": []}'
        result = critic_node(state)
    assert result["critic_round"] == 2


def test_critic_fails_closed_on_malformed_json():
    state = make_state("answer", {"k": {"text": "t", "attribution": ""}})
    with patch("agent.nodes.critic.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = "this is not JSON at all"
        result = critic_node(state)
    assert result["reflection_passed"] is False
    assert len(result["critic_issues"]) == 1


def test_critic_uses_all_evidence_not_just_five():
    evidence_map = {
        f"k{i}": {"text": f"chunk {i} content", "attribution": f"src{i}"} for i in range(10)
    }
    state = make_state("draft", evidence_map)
    captured_prompt = []
    def mock_complete(messages, system="", max_tokens=512, timeout=None):
        captured_prompt.append(messages[0]["content"])
        return '{"overall": "PASS", "issues": []}'

    with patch("agent.nodes.critic.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.side_effect = mock_complete
        critic_node(state)

    for i in range(10):
        assert f"chunk {i} content" in captured_prompt[0]


def test_critic_flags_red_flag_phrases_on_malformed_json():
    state = make_state(
        draft_answer="I cannot find the data in the sources.",
        evidence_map={"k": {"text": "t", "attribution": ""}},
    )
    with patch("agent.nodes.critic.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = "this is not JSON at all"
        result = critic_node(state)
    assert result["reflection_passed"] is False
    assert len(result["critic_issues"]) == 1
    assert result["critic_issues"][0]["issue_type"] == "missing_context"


def test_critic_marks_clean_answer_unverified_on_malformed_json():
    state = make_state(
        draft_answer="The yield was 87% according to the source.",
        evidence_map={"k": {"text": "t", "attribution": ""}},
    )
    with patch("agent.nodes.critic.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = "this is not JSON at all"
        result = critic_node(state)
    assert result["reflection_passed"] is False
    assert result["critic_issues"][0]["issue_type"] == "unsupported"


def test_critic_truncates_large_context(monkeypatch):
    """critic_node should not pass unlimited context to LLM.

    The default budget scales with the active LLM's window, so explicitly
    clamp it to a small value for this test to force truncation.
    """
    import agent.context_utils as ctx_utils
    monkeypatch.setattr(ctx_utils, "MAX_CONTEXT_TOKENS", 8000)

    long_text = "reaction details " * 200  # ~600 tokens per chunk
    evidence_map = {
        f"k{i}": {"text": long_text, "attribution": f"src{i}", "score": 0.5, "payload": {}}
        for i in range(20)  # 20 * ~600 = ~12000 tokens > 8000 limit
    }
    state = make_state("The yield was 87%.", evidence_map)

    captured_prompts = []
    def mock_complete(messages, system="", max_tokens=512, timeout=None):
        captured_prompts.append(messages[0]["content"])
        return '{"overall": "PASS", "issues": []}'

    with patch("agent.nodes.critic.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.side_effect = mock_complete
        critic_node(state)

    full_context_len = len(long_text) * 20
    assert len(captured_prompts[0]) < full_context_len
