# ingestion/state_tracker.py
"""Tracks ingestion state (MD5 hashes) to enable incremental re-ingestion."""
import hashlib
import json
from pathlib import Path

_STATE_FILENAME = ".ingestion_state.json"


def compute_md5(path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(docs_dir: Path) -> dict[str, str]:
    """Load {filename: md5} state from docs_dir/.ingestion_state.json."""
    state_file = docs_dir / _STATE_FILENAME
    if not state_file.exists():
        return {}
    with open(state_file) as f:
        return json.load(f)


def save_state(docs_dir: Path, state: dict[str, str]) -> None:
    """Persist {filename: md5} state to docs_dir/.ingestion_state.json."""
    state_file = docs_dir / _STATE_FILENAME
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def is_changed(path: Path, state: dict[str, str]) -> bool:
    """Return True if file is new or its MD5 differs from stored hash."""
    current_md5 = compute_md5(path)
    return state.get(path.name) != current_md5
