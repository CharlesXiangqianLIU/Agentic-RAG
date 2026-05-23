"""Skip-when-unavailable scaffolding for the e2e suite.

These tests are NOT collected by the default ``pytest tests/`` run.
They need:

  * A running Qdrant (e.g. ``docker compose -f docker-compose.test.yml up -d qdrant``).
  * The bge-m3 embedder model files (~2 GB; downloaded automatically on
    first use into ``~/.cache/huggingface/``).

Run with:

    make e2e          # convenience wrapper
    pytest tests/e2e/ -v

Override the Qdrant URL via ``E2E_QDRANT_URL`` (default: http://localhost:6334).
"""
from __future__ import annotations

import os
import uuid

import pytest


_E2E_QDRANT_URL = os.getenv("E2E_QDRANT_URL", "http://localhost:6334")


def _qdrant_reachable(url: str) -> bool:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=url, timeout=2.0)
        client.get_collections()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """If Qdrant isn't reachable, skip every test in this directory."""
    if _qdrant_reachable(_E2E_QDRANT_URL):
        return
    reason = (
        f"e2e: Qdrant not reachable at {_E2E_QDRANT_URL} — "
        "start it with `make e2e-up` and re-run."
    )
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "e2e" in item.nodeid:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def e2e_qdrant_url() -> str:
    return _E2E_QDRANT_URL


@pytest.fixture
def e2e_collection_name() -> str:
    """Unique collection per test so parallel runs don't trample each other."""
    return f"e2e_{uuid.uuid4().hex[:8]}"
