# tests/test_answer_node_v2.py
from unittest.mock import patch
from agent.state import AgentState
from agent.nodes.answer import (
    answer_node,
    _build_context,
    _select_system_prompt,
    _max_tokens_for,
    _build_history_prefix,
    _DEFAULT_SYSTEM,
    _DEFAULT_COMPARISON_SYSTEM,
    _DEFAULT_TREND_SYSTEM,
)
from tests.conftest import make_agent_state


def make_state(evidence_map):
    return make_agent_state(evidence_map=evidence_map)


def test_answer_reads_from_evidence_map():
    evidence_map = {
        "k1": {"text": "Entry 3 | 87%",
               "attribution": "[Source: PRJ-031.docx | Page 12 | Section: Table 1]",
               "payload": {}}
    }
    state = make_state(evidence_map)
    with patch("agent.nodes.answer.get_llm_provider") as mock_llm:
        mock_llm.return_value.stream.side_effect = lambda *a, **kw: iter([
            "The yield was 87%. [Source: PRJ-031.docx | Page 12 | Section: Table 1]"
        ])
        result = answer_node(state)
    assert "final_answer" in result
    assert "draft_answer" in result
    assert result["final_answer"]


def test_answer_uses_all_evidence_chunks_in_context():
    evidence_map = {
        f"k{i}": {"text": f"chunk text {i}", "attribution": f"src{i}", "payload": {}}
        for i in range(5)
    }
    state = make_state(evidence_map)
    captured = []
    def mock_stream(messages, system="", max_tokens=2048, timeout=None):
        captured.append(messages[0]["content"])
        return iter(["The answer."])
    with patch("agent.nodes.answer.get_llm_provider") as mock_llm:
        mock_llm.return_value.stream.side_effect = mock_stream
        answer_node(state)
    for i in range(5):
        assert f"chunk text {i}" in captured[0]


def test_answer_safety_check_flags_unsupported_number():
    evidence_map = {"k1": {"text": "Entry 3 yield was 45%", "attribution": "", "payload": {}}}
    state = make_state(evidence_map)
    with patch("agent.nodes.answer.get_llm_provider") as mock_llm:
        mock_llm.return_value.stream.side_effect = lambda *a, **kw: iter(["The yield was 99%."])
        result = answer_node(state)
    assert "[UNSUPPORTED" in result["final_answer"]


def test_answer_truncates_large_context():
    # 20 chunks of ~1100 tokens each — must exceed the budget so the
    # builder truncates. We pass an explicit small budget instead of
    # relying on the model-aware default (which can be 80 000+ tokens
    # for claude-sonnet-4-6 and would fit all 20 chunks comfortably).
    long_text = "The reaction was performed with reagents X and Y at high temperature. Details include complex procedures and multiple reagents. " * 50
    evidence_map = {
        f"k{i}": {
            "text": long_text,
            "attribution": f"[Source: file{i}.docx | Page 1 | Section: Procedure]",
            "score": 0.5 + (i * 0.01),
            "payload": {}
        }
        for i in range(20)
    }
    chunks = list(evidence_map.values())
    context, n_included = _build_context(chunks, max_tokens=8000)
    assert n_included < 20
    assert n_included > 0
    assert len(context) > 0


def test_answer_sorts_by_score():
    # Create 3 chunks with different scores
    chunks = [
        {"text": "Low score chunk", "attribution": "[Source: low.docx]", "score": 0.3},
        {"text": "High score chunk", "attribution": "[Source: high.docx]", "score": 0.9},
        {"text": "Medium score chunk", "attribution": "[Source: medium.docx]", "score": 0.5},
    ]
    context, n_included = _build_context(chunks)
    # All 3 should fit
    assert n_included == 3
    # High score chunk should come first in context
    high_idx = context.find("High score chunk")
    med_idx = context.find("Medium score chunk")
    low_idx = context.find("Low score chunk")
    assert high_idx < med_idx < low_idx


def test_answer_includes_all_chunks_in_safety_check():
    # Create evidence map with many chunks
    evidence_map = {
        f"k{i}": {
            "text": f"Evidence {i} with number {i*10}%",
            "attribution": f"[Source: file{i}.docx]",
            "score": 0.5,
            "payload": {}
        }
        for i in range(10)
    }
    state = make_state(evidence_map)
    chunks = list(evidence_map.values())

    with patch("agent.nodes.answer.get_llm_provider") as mock_llm, \
         patch("agent.nodes.answer.check_answer") as mock_check:
        mock_llm.return_value.stream.side_effect = lambda *a, **kw: iter(["Some answer"])
        mock_check.return_value = "Final answer"

        answer_node(state)

        # Verify check_answer was called with all chunks
        mock_check.assert_called_once()
        call_args = mock_check.call_args
        # check_answer receives (draft_answer, chunks)
        chunks_arg = call_args[0][1]
        assert len(chunks_arg) == 10


def test_max_tokens_comparison_returns_4096():
    assert _max_tokens_for("comparison") == 4096


def test_max_tokens_trend_returns_4096():
    assert _max_tokens_for("trend") == 4096


def test_max_tokens_lookup_returns_2048():
    assert _max_tokens_for("lookup") == 2048


def test_select_system_prompt_comparison():
    """_select_system_prompt returns the comparison default for 'comparison' type."""
    result = _select_system_prompt("comparison")
    assert result == _DEFAULT_COMPARISON_SYSTEM
    assert "Markdown table" in result


def test_select_system_prompt_trend():
    """_select_system_prompt returns the trend default for 'trend' type."""
    result = _select_system_prompt("trend")
    assert result == _DEFAULT_TREND_SYSTEM
    assert "numbered list" in result


def test_select_system_prompt_default():
    """_select_system_prompt falls back to the generic default for 'lookup'."""
    result = _select_system_prompt("lookup")
    assert result == _DEFAULT_SYSTEM
    assert "source passages" in result
    assert "Markdown table" not in result


def test_answer_includes_filter_scope_in_system_prompt():
    """When metadata_filters is set, system prompt should include scope note."""
    evidence_map = {
        "k1": {"text": "Entry 3 yield 87%", "attribution": "[Source: PRJ-031.docx | Page 1 | Section: T1]", "payload": {}}
    }
    state = AgentState(
        question="What was the yield?",
        question_type="lookup",
        sub_tasks=[],
        current_sub_task="", current_agent_type="",
        worker_results=[], evidence_map=evidence_map,
        draft_answer="", final_answer="",
        critic_issues=[], critic_round=0, reflection_passed=False, messages=[],
        metadata_filters={"project_id": "PRJ-031"},
    )
    captured_systems = []
    def mock_stream(messages, system="", max_tokens=2048, timeout=None):
        captured_systems.append(system)
        return iter(["The yield was 87%."])

    with patch("agent.nodes.answer.get_llm_provider") as mock_llm:
        mock_llm.return_value.stream.side_effect = mock_stream
        answer_node(state)

    assert len(captured_systems) == 1
    assert "project_id=PRJ-031" in captured_systems[0]
    assert "scoped" in captured_systems[0].lower()


def test_answer_no_scope_note_when_no_filters():
    """When metadata_filters is empty/absent, system prompt should not have scope note."""
    evidence_map = {"k1": {"text": "yield 87%", "attribution": "", "payload": {}}}
    state = make_state(evidence_map)  # make_state doesn't set metadata_filters
    captured_systems = []
    def mock_stream(messages, system="", max_tokens=2048, timeout=None):
        captured_systems.append(system)
        return iter(["The yield was 87%."])
    with patch("agent.nodes.answer.get_llm_provider") as mock_llm:
        mock_llm.return_value.stream.side_effect = mock_stream
        answer_node(state)
    assert "scoped" not in captured_systems[0].lower()


def test_build_history_prefix_empty():
    assert _build_history_prefix([]) == ""


def test_build_history_prefix_single_turn():
    history = [{"question": "What catalyst?", "answer": "Pd(OAc)2."}]
    result = _build_history_prefix(history)
    assert "What catalyst?" in result
    assert "Pd(OAc)2." in result
    assert result.startswith("Prior conversation context:")


def test_build_history_prefix_caps_at_two_turns():
    history = [
        {"question": f"Q{i}", "answer": f"A{i}"}
        for i in range(5)
    ]
    result = _build_history_prefix(history)
    # Only last 2 turns should appear
    assert "Q4" in result and "Q3" in result
    assert "Q0" not in result


def test_answer_injects_conversation_history_into_prompt():
    """Conversation history should appear in the user message sent to the LLM."""
    evidence_map = {"k1": {"text": "yield 87%", "attribution": "", "payload": {}}}
    history = [{"question": "Which project?", "answer": "PRJ-031."}]
    state = make_agent_state(evidence_map=evidence_map, conversation_history=history)
    captured_messages = []
    def mock_stream(messages, system="", max_tokens=2048, timeout=None):
        captured_messages.append(messages[0]["content"])
        return iter(["87%."])
    with patch("agent.nodes.answer.get_llm_provider") as mock_llm:
        mock_llm.return_value.stream.side_effect = mock_stream
        answer_node(state)
    assert "Which project?" in captured_messages[0]
    assert "PRJ-031." in captured_messages[0]


def test_answer_forwards_request_timeout():
    evidence_map = {"k1": {"text": "yield 87%", "attribution": "", "payload": {}}}
    state = make_state(evidence_map)
    with patch("agent.nodes.answer.get_llm_provider") as mock_llm:
        mock_llm.return_value.stream.side_effect = lambda *a, **kw: iter(["87%."])
        answer_node(state)
    assert "timeout" in mock_llm.return_value.stream.call_args.kwargs


def test_answer_no_history_prefix_when_absent():
    """When conversation_history is absent, prompt should not include history prefix."""
    evidence_map = {"k1": {"text": "yield 87%", "attribution": "", "payload": {}}}
    state = make_state(evidence_map)  # no conversation_history key
    captured_messages = []
    def mock_stream(messages, system="", max_tokens=2048, timeout=None):
        captured_messages.append(messages[0]["content"])
        return iter(["87%."])
    with patch("agent.nodes.answer.get_llm_provider") as mock_llm:
        mock_llm.return_value.stream.side_effect = mock_stream
        answer_node(state)
    assert "Prior conversation context:" not in captured_messages[0]


def test_build_history_prefix_truncates_long_answers():
    history = [{
        "question": "What happened?",
        "answer": "yield " * 2000,
    }]
    result = _build_history_prefix(history)
    assert result.startswith("Prior conversation context:")
    assert len(result) < len("yield " * 2000)
