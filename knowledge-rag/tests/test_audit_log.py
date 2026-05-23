"""Audit log table in frontend/persistence.py."""
import pytest

from frontend.persistence import (
    clear_history,
    read_audit,
    save_message,
    write_audit,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "audit.db"


def test_write_and_read_round_trip(db_path):
    write_audit(
        "What is the policy?", "The policy is X.",
        question_type="lookup", db_path=db_path,
        evidence=[{"text": "X is the policy", "attribution": "h.docx"}],
        metadata_filters={"category": "policy"},
    )
    rows = read_audit(db_path=db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["question"] == "What is the policy?"
    assert row["answer"] == "The policy is X."
    assert row["question_type"] == "lookup"
    assert row["evidence"] == [{"text": "X is the policy", "attribution": "h.docx"}]
    assert row["metadata_filters"] == {"category": "policy"}
    assert row["has_unsupported"] is False
    assert row["created_at"]  # timestamp populated


def test_has_unsupported_set_when_answer_contains_tag(db_path):
    write_audit(
        "What yield?", "[UNSUPPORTED: 87%] was reported.",
        db_path=db_path,
    )
    rows = read_audit(db_path=db_path)
    assert rows[0]["has_unsupported"] is True


def test_read_audit_returns_newest_first(db_path):
    for i in range(3):
        write_audit(f"q{i}", f"a{i}", db_path=db_path)
    rows = read_audit(db_path=db_path)
    questions = [r["question"] for r in rows]
    assert questions == ["q2", "q1", "q0"]


def test_read_audit_respects_limit(db_path):
    for i in range(10):
        write_audit(f"q{i}", f"a{i}", db_path=db_path)
    assert len(read_audit(limit=3, db_path=db_path)) == 3


def test_clear_history_does_not_touch_audit_log(db_path):
    """Audit log is compliance data — must survive a user History clear."""
    save_message(role="user", content="hello", db_path=db_path)
    write_audit("q", "a", db_path=db_path)

    clear_history(db_path=db_path)

    assert len(read_audit(db_path=db_path)) == 1  # audit still there


def test_write_audit_accepts_no_optional_args(db_path):
    """Minimum-viable call shape: just question + answer."""
    rid = write_audit("q", "a", db_path=db_path)
    assert isinstance(rid, int) and rid > 0
    row = read_audit(db_path=db_path)[0]
    assert row["evidence"] == []
    assert row["metadata_filters"] == {}
    assert row["user_label"] == ""
