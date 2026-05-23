"""Conversation history summarisation in answer._build_history_prefix."""
from unittest.mock import patch

import agent.nodes.answer as answer_mod
from agent.nodes.answer import _build_history_prefix


def _turn(q: str, a: str) -> dict:
    return {"question": q, "answer": a}


def _history(n: int) -> list[dict]:
    return [_turn(f"Q{i}", f"A{i}") for i in range(n)]


def test_summary_disabled_returns_verbatim_only(monkeypatch):
    monkeypatch.delenv("HISTORY_SUMMARY", raising=False)
    with patch.object(answer_mod, "get_llm_provider") as mock_llm:
        prefix = _build_history_prefix(_history(5))
    assert mock_llm.call_count == 0  # LLM never invoked when disabled
    assert "Earlier conversation summary" not in prefix
    assert "Prior conversation context" in prefix


def test_summary_runs_when_history_exceeds_recent_window(monkeypatch):
    monkeypatch.setenv("HISTORY_SUMMARY", "1")
    monkeypatch.setattr(answer_mod, "ANSWER_HISTORY_TURNS", 2)

    with patch.object(answer_mod, "get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = "Summary: user is asking about leave policies."
        prefix = _build_history_prefix(_history(6))

    mock_llm.return_value.complete.assert_called_once()
    assert "Earlier conversation summary" in prefix
    assert "Summary: user is asking about leave policies." in prefix
    # The last 2 turns are still verbatim.
    assert "Q4" in prefix and "Q5" in prefix


def test_summary_not_called_when_only_recent_turns(monkeypatch):
    """If the history fits within ANSWER_HISTORY_TURNS, no summary is needed."""
    monkeypatch.setenv("HISTORY_SUMMARY", "1")
    monkeypatch.setattr(answer_mod, "ANSWER_HISTORY_TURNS", 5)

    with patch.object(answer_mod, "get_llm_provider") as mock_llm:
        prefix = _build_history_prefix(_history(3))

    mock_llm.return_value.complete.assert_not_called()
    assert "Earlier conversation summary" not in prefix


def test_summary_failure_falls_back_to_verbatim(monkeypatch):
    monkeypatch.setenv("HISTORY_SUMMARY", "1")
    monkeypatch.setattr(answer_mod, "ANSWER_HISTORY_TURNS", 2)

    with patch.object(answer_mod, "get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.side_effect = RuntimeError("LLM down")
        prefix = _build_history_prefix(_history(6))

    # No summary block, but verbatim turns still survive.
    assert "Earlier conversation summary" not in prefix
    assert "Q4" in prefix and "Q5" in prefix


def test_empty_history_returns_empty_string():
    assert _build_history_prefix([]) == ""
