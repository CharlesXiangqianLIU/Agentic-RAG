"""Tests for frontend conversation history persistence."""
import sqlite3
from pathlib import Path
from frontend.persistence import load_history, save_message, clear_history, _get_conn


def test_save_and_load_messages(tmp_path: Path):
    """Save 2 messages (user + assistant), load_history returns them in order with correct fields."""
    db_path = tmp_path / "test.db"

    # Save user message
    user_id = save_message(
        role="user",
        content="What is the optimal temperature?",
        question="What is the optimal temperature?",
        db_path=db_path,
    )
    assert user_id == 1

    # Save assistant message with chunks
    chunks = [
        {
            "text": "The optimal temperature is 80 °C.",
            "attribution": "[Source: file | Page 1 | Section: Optimization Study]",
        }
    ]
    assistant_id = save_message(
        role="assistant",
        content="The optimal temperature is 80 °C based on the study results.",
        question="What is the optimal temperature?",
        chunks=chunks,
        db_path=db_path,
    )
    assert assistant_id == 2

    # Load history
    history = load_history(db_path=db_path)

    assert len(history) == 2

    # Check user message
    assert history[0]["id"] == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is the optimal temperature?"
    assert history[0]["question"] == "What is the optimal temperature?"
    assert history[0]["chunks"] == []

    # Check assistant message
    assert history[1]["id"] == 2
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "The optimal temperature is 80 °C based on the study results."
    assert history[1]["question"] == "What is the optimal temperature?"
    assert len(history[1]["chunks"]) == 1
    assert history[1]["chunks"][0]["text"] == "The optimal temperature is 80 °C."


def test_load_history_empty_db(tmp_path: Path):
    """Load_history on fresh DB returns []."""
    db_path = tmp_path / "empty.db"
    history = load_history(db_path=db_path)
    assert history == []


def test_save_returns_incrementing_ids(tmp_path: Path):
    """Two saves return sequential ids."""
    db_path = tmp_path / "incremental.db"

    id1 = save_message(role="user", content="Q1", db_path=db_path)
    id2 = save_message(role="assistant", content="A1", db_path=db_path)
    id3 = save_message(role="user", content="Q2", db_path=db_path)

    assert id1 == 1
    assert id2 == 2
    assert id3 == 3


def test_clear_history(tmp_path: Path):
    """After clear_history, load_history returns []."""
    db_path = tmp_path / "clearable.db"

    # Add some messages
    save_message(role="user", content="Q1", db_path=db_path)
    save_message(role="assistant", content="A1", db_path=db_path)

    # Verify they're there
    history = load_history(db_path=db_path)
    assert len(history) == 2

    # Clear
    clear_history(db_path=db_path)

    # Verify empty
    history = load_history(db_path=db_path)
    assert history == []


def test_chunks_serialized_correctly(tmp_path: Path):
    """Save a message with chunks list, load back, chunks are deserialized."""
    db_path = tmp_path / "chunks_test.db"

    chunks = [
        {
            "text": "Result 1",
            "attribution": "[Source: file1 | Page 1]",
            "page_number": 1,
            "section": "Results",
        },
        {
            "text": "Result 2",
            "attribution": "[Source: file2 | Page 2]",
            "page_number": 2,
            "section": "Discussion",
        },
    ]

    save_message(
        role="assistant",
        content="Summary of results",
        question="What were the results?",
        chunks=chunks,
        db_path=db_path,
    )

    history = load_history(db_path=db_path)
    assert len(history) == 1

    loaded_chunks = history[0]["chunks"]
    assert len(loaded_chunks) == 2
    assert loaded_chunks[0]["text"] == "Result 1"
    assert loaded_chunks[0]["attribution"] == "[Source: file1 | Page 1]"
    assert loaded_chunks[0]["page_number"] == 1
    assert loaded_chunks[1]["text"] == "Result 2"
    assert loaded_chunks[1]["section"] == "Discussion"


def test_save_and_load_question_type(tmp_path: Path):
    """Save message with question_type, load_history, assert question_type is preserved."""
    db_path = tmp_path / "question_type_test.db"

    # Save user message
    save_message(
        role="user",
        content="Compare the two reactions.",
        question="Compare the two reactions.",
        db_path=db_path,
    )

    # Save assistant message with question_type
    save_message(
        role="assistant",
        content="The two reactions differ in...",
        question="Compare the two reactions.",
        question_type="comparison",
        db_path=db_path,
    )

    # Load and verify
    history = load_history(db_path=db_path)
    assert len(history) == 2

    # User message should have empty question_type
    assert history[0]["question_type"] == ""

    # Assistant message should have question_type
    assert history[1]["question_type"] == "comparison"


def test_migration_adds_question_type_column(tmp_path: Path):
    """Create old schema DB without question_type, run migration, verify column exists and old rows have empty question_type."""
    db_path = tmp_path / "migration_test.db"

    # Create old schema without question_type column
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            question TEXT DEFAULT '',
            chunks_json TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    # Insert a row with old schema
    conn.execute(
        "INSERT INTO history (role, content, question) VALUES (?, ?, ?)",
        ("assistant", "Old message", "Old question"),
    )
    conn.commit()
    conn.close()

    # Now open a managed connection — _get_conn is now a contextmanager,
    # entering it triggers the migration path.
    with _get_conn(db_path):
        pass

    # Load history — should have the question_type column now
    entries = load_history(db_path=db_path)
    assert len(entries) == 1
    assert entries[0]["role"] == "assistant"
    assert entries[0]["content"] == "Old message"
    assert entries[0]["question_type"] == ""  # Should be empty for old rows
