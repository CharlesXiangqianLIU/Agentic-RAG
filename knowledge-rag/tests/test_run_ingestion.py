"""Tests for concurrent ingestion in run_ingestion.py."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test 1: files are processed concurrently (both files ingested)
# ---------------------------------------------------------------------------

def test_main_processes_files_concurrently(tmp_path, capsys):
    """Both .docx files in temp dir must be ingested when none are skipped."""
    # Create two fake .docx files (parse is mocked, so empty bytes are fine)
    file_a = tmp_path / "report_a.docx"
    file_b = tmp_path / "report_b.docx"
    file_a.write_bytes(b"")
    file_b.write_bytes(b"")

    with (
        patch("ingestion.run_ingestion.ingest_file", return_value=5) as mock_ingest,
        patch("ingestion.run_ingestion.compute_md5", return_value="abc123"),
        patch("ingestion.run_ingestion.load_state", return_value={}),
        patch("ingestion.run_ingestion.save_state"),
        patch("ingestion.run_ingestion.is_changed", return_value=True),
        patch("retrieval.embedder.embed_texts", return_value=[[0.0]]),
        patch("ingestion.indexer.ensure_collection"),
        patch("sys.argv", ["run_ingestion", "--docs-dir", str(tmp_path)]),
    ):
        from ingestion.run_ingestion import main
        main()

    assert mock_ingest.call_count == 2
    called_paths = {call.args[0].name for call in mock_ingest.call_args_list}
    assert called_paths == {"report_a.docx", "report_b.docx"}

    captured = capsys.readouterr()
    assert "10 chunks" in captured.out or "5 chunks" in captured.out


# ---------------------------------------------------------------------------
# Test 2: unchanged files are skipped
# ---------------------------------------------------------------------------

def test_main_skips_unchanged_files(tmp_path, capsys):
    """When is_changed returns False, ingest_file must not be called."""
    fake_docx = tmp_path / "unchanged.docx"
    fake_docx.write_bytes(b"")

    with (
        patch("ingestion.run_ingestion.ingest_file") as mock_ingest,
        patch("ingestion.run_ingestion.load_state", return_value={"unchanged.docx": "deadbeef"}),
        patch("ingestion.run_ingestion.save_state"),
        patch("ingestion.run_ingestion.is_changed", return_value=False),
        patch("retrieval.embedder.embed_texts", return_value=[[0.0]]),
        patch("ingestion.indexer.ensure_collection"),
        patch("sys.argv", ["run_ingestion", "--docs-dir", str(tmp_path)]),
    ):
        from ingestion.run_ingestion import main
        main()

    mock_ingest.assert_not_called()

    captured = capsys.readouterr()
    assert "[SKIP]" in captured.out


# ---------------------------------------------------------------------------
# Test 3: errors in ingest_file are handled gracefully
# ---------------------------------------------------------------------------

def test_process_file_handles_errors_gracefully(tmp_path):
    """When ingest_file raises, _process_file must return an ERROR line and 0 chunks."""
    fake_docx = tmp_path / "bad_file.docx"
    fake_docx.write_bytes(b"")

    with (
        patch("ingestion.run_ingestion.ingest_file", side_effect=Exception("boom")),
        patch("ingestion.run_ingestion.is_changed", return_value=True),
    ):
        from ingestion.run_ingestion import _process_file
        line, chunk_count = _process_file(
            path=fake_docx,
            i=1,
            total=1,
            force=False,
            state={},
        )

    assert "ERROR" in line
    assert "boom" in line
    assert chunk_count == 0
