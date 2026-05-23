# knowledge-rag/tests/conftest.py
import os

import pytest
from pathlib import Path
from agent.state import AgentState

# Disable the semantic-dedup pass in synthesis by default for the whole
# test session. It calls the bge-m3 embedder, which is slow and may try
# to download model weights — neither is acceptable in unit tests. Tests
# that need to exercise the pass can re-enable via monkeypatch.setenv.
os.environ.setdefault("SEMANTIC_DEDUP", "0")

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def tmp_docx():
    return str(FIXTURES_DIR / "sample.docx")


def make_agent_state(**kwargs) -> AgentState:
    """Factory for AgentState with sensible defaults. Override any field via kwargs."""
    defaults = dict(
        question="What does section 2 cover?",
        question_type="lookup",
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
    defaults.update(kwargs)
    return AgentState(**defaults)
