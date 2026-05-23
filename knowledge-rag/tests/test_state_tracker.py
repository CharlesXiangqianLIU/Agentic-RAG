# tests/test_state_tracker.py
"""Tests for ingestion state tracking (MD5-based change detection)."""
import json
from pathlib import Path

from ingestion.state_tracker import compute_md5, load_state, save_state, is_changed


def test_compute_md5_returns_hex_string(tmp_path: Path):
    """compute_md5 returns a 32-char hex string for a real file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    md5 = compute_md5(test_file)

    assert isinstance(md5, str)
    assert len(md5) == 32
    assert all(c in "0123456789abcdef" for c in md5)


def test_compute_md5_consistent(tmp_path: Path):
    """compute_md5 returns the same hash for identical file content."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    md5_1 = compute_md5(test_file)
    md5_2 = compute_md5(test_file)

    assert md5_1 == md5_2


def test_load_state_empty_when_missing(tmp_path: Path):
    """load_state returns {} when state file doesn't exist."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    state = load_state(docs_dir)

    assert state == {}


def test_load_state_from_existing_file(tmp_path: Path):
    """load_state returns dict from existing state file."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    state_file = docs_dir / ".ingestion_state.json"

    expected_state = {
        "file1.docx": "abc123def456",
        "file2.docx": "xyz789uvw012",
    }
    with open(state_file, "w") as f:
        json.dump(expected_state, f)

    state = load_state(docs_dir)

    assert state == expected_state


def test_save_state_creates_file(tmp_path: Path):
    """save_state creates the file with correct content."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    test_state = {
        "file1.docx": "abc123def456",
        "file2.docx": "xyz789uvw012",
    }
    save_state(docs_dir, test_state)

    state_file = docs_dir / ".ingestion_state.json"
    assert state_file.exists()

    with open(state_file) as f:
        loaded_state = json.load(f)

    assert loaded_state == test_state


def test_save_state_overwrites_existing(tmp_path: Path):
    """save_state overwrites an existing state file."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    old_state = {"old_file.docx": "old_hash"}
    save_state(docs_dir, old_state)

    new_state = {"new_file.docx": "new_hash"}
    save_state(docs_dir, new_state)

    state_file = docs_dir / ".ingestion_state.json"
    with open(state_file) as f:
        loaded_state = json.load(f)

    assert loaded_state == new_state


def test_is_changed_true_for_new_file(tmp_path: Path):
    """is_changed returns True for a new file (not in state)."""
    test_file = tmp_path / "test.docx"
    test_file.write_text("content")

    state = {}  # empty state

    assert is_changed(test_file, state) is True


def test_is_changed_false_when_md5_matches(tmp_path: Path):
    """is_changed returns False when MD5 matches stored value."""
    test_file = tmp_path / "test.docx"
    test_file.write_text("content")

    md5 = compute_md5(test_file)
    state = {"test.docx": md5}

    assert is_changed(test_file, state) is False


def test_is_changed_true_when_file_modified(tmp_path: Path):
    """is_changed returns True when file content changes."""
    test_file = tmp_path / "test.docx"
    test_file.write_text("original content")

    old_md5 = compute_md5(test_file)
    state = {"test.docx": old_md5}

    # Modify the file
    test_file.write_text("modified content")

    assert is_changed(test_file, state) is True


def test_is_changed_true_when_hash_missing(tmp_path: Path):
    """is_changed returns True when filename is in state but hash is different."""
    test_file = tmp_path / "test.docx"
    test_file.write_text("content")

    state = {"test.docx": "wrong_hash"}

    assert is_changed(test_file, state) is True
