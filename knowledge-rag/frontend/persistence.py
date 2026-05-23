"""Persistence layer for chat history + audit log.

By default uses local SQLite at ``~/.knowledge_rag_history.db``. Set
``DATABASE_URL`` to a ``postgresql://...`` connection string to switch
to Postgres (requires the optional ``psycopg`` dependency — install with
``pip install -e ".[postgres]"``).

Two tables live in either backend:

* ``history`` — chat-style messages rendered back to the UI.
* ``audit_log`` — every Q&A turn with its retrieved evidence, the
  question type, whether the safety check flagged anything, and a
  timestamp. Intended for HR / legal / compliance domains that need a
  record of what the system told whom.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_DB_PATH = Path.home() / ".knowledge_rag_history.db"


# ---------------------------------------------------------------------------
# Backend detection + adapter
# ---------------------------------------------------------------------------


def _is_postgres_url(db_path: Path | str | None) -> bool:
    """Return True if ``db_path`` looks like a Postgres connection string."""
    candidate = os.fspath(db_path) if db_path is not None else ""
    return candidate.startswith("postgres://") or candidate.startswith("postgresql://")


def _resolve_dsn(db_path: Path | str | None) -> Path | str:
    """Pick the effective DSN. Env var ``DATABASE_URL`` wins over the default,
    and an explicit ``db_path`` argument wins over both (tests pass paths
    directly).
    """
    if db_path is not None and db_path != _DB_PATH:
        return db_path
    env_url = os.getenv("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    return _DB_PATH


class _ConnAdapter:
    """Uniform wrapper over sqlite3 and psycopg connections.

    Both expose ``execute(sql, params=())`` and ``commit()`` / ``close()``.
    The adapter translates ``?`` placeholders to ``%s`` for Postgres and
    captures ``cursor.lastrowid`` semantics uniformly via ``insert_returning_id``.
    """

    def __init__(self, raw, kind: str):
        self.raw = raw
        self.kind = kind  # "sqlite" or "postgres"

    @staticmethod
    def _translate(sql: str, kind: str) -> str:
        return sql.replace("?", "%s") if kind == "postgres" else sql

    def execute(self, sql: str, params: tuple | list = ()):
        sql = self._translate(sql, self.kind)
        if self.kind == "postgres":
            cur = self.raw.cursor()
            cur.execute(sql, params)
        else:
            # sqlite3.Connection.execute returns a fresh Cursor.
            cur = self.raw.execute(sql, params)
        return cur

    def insert_returning_id(self, sql: str, params: tuple | list, returning_col: str = "id") -> int:
        """Run an INSERT and return the auto-generated id of the new row."""
        if self.kind == "postgres":
            cur = self.raw.cursor()
            cur.execute(self._translate(sql, self.kind) + f" RETURNING {returning_col}", params)
            row = cur.fetchone()
            return int(row[0])
        cur = self.raw.execute(sql, params)
        return cur.lastrowid

    def fetchall(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        cur = self.execute(sql, params)
        return cur.fetchall()

    def commit(self):
        self.raw.commit()

    def close(self):
        self.raw.close()


# ---------------------------------------------------------------------------
# Schema (per backend)
# ---------------------------------------------------------------------------


_SQLITE_CREATE_HISTORY = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    question TEXT DEFAULT '',
    chunks_json TEXT DEFAULT '[]',
    question_type TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_SQLITE_CREATE_AUDIT = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_label TEXT DEFAULT '',
    question TEXT NOT NULL,
    question_type TEXT DEFAULT '',
    answer TEXT NOT NULL,
    evidence_json TEXT DEFAULT '[]',
    has_unsupported INTEGER DEFAULT 0,
    metadata_filters_json TEXT DEFAULT '{}'
)
"""

_POSTGRES_CREATE_HISTORY = """
CREATE TABLE IF NOT EXISTS history (
    id BIGSERIAL PRIMARY KEY,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    question TEXT DEFAULT '',
    chunks_json TEXT DEFAULT '[]',
    question_type TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_POSTGRES_CREATE_AUDIT = """
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_label TEXT DEFAULT '',
    question TEXT NOT NULL,
    question_type TEXT DEFAULT '',
    answer TEXT NOT NULL,
    evidence_json TEXT DEFAULT '[]',
    has_unsupported INTEGER DEFAULT 0,
    metadata_filters_json TEXT DEFAULT '{}'
)
"""


# Track which DSNs have already been migrated in this process.
# Avoids running ALTER TABLE on every connection open.
_migrated: set[str] = set()


@contextmanager
def _get_conn(db_path: Path | str | None = _DB_PATH) -> Iterator[_ConnAdapter]:
    dsn = _resolve_dsn(db_path)
    key = str(dsn)

    if _is_postgres_url(dsn):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "DATABASE_URL points at Postgres but the 'psycopg' package is not installed. "
                "Install with: pip install -e \".[postgres]\""
            ) from exc
        raw = psycopg.connect(str(dsn))
        adapter = _ConnAdapter(raw, kind="postgres")
        adapter.execute(_POSTGRES_CREATE_HISTORY)
        adapter.execute(_POSTGRES_CREATE_AUDIT)
        adapter.commit()
        _migrate_if_needed(adapter, key)
    else:
        raw = sqlite3.connect(str(dsn))
        adapter = _ConnAdapter(raw, kind="sqlite")
        adapter.execute(_SQLITE_CREATE_HISTORY)
        adapter.execute(_SQLITE_CREATE_AUDIT)
        adapter.commit()
        _migrate_if_needed(adapter, key)

    try:
        yield adapter
    finally:
        adapter.close()


def _migrate_if_needed(adapter: _ConnAdapter, key: str) -> None:
    if key in _migrated:
        return
    try:
        adapter.execute("ALTER TABLE history ADD COLUMN question_type TEXT DEFAULT ''")
        adapter.commit()
    except Exception:
        # Column already exists (sqlite) or already added (postgres).
        pass
    _migrated.add(key)


# ---------------------------------------------------------------------------
# Public API — unchanged signatures
# ---------------------------------------------------------------------------


def load_history(db_path: Path | str | None = _DB_PATH) -> list[dict]:
    """Load all history rows as list of dicts, matching ``st.session_state.history``."""
    with _get_conn(db_path) as conn:
        rows = conn.fetchall(
            "SELECT id, role, content, question, chunks_json, question_type FROM history ORDER BY id"
        )
    result: list[dict] = []
    for row_id, role, content, question, chunks_json, question_type in rows:
        result.append({
            "id": row_id,
            "role": role,
            "content": content,
            "question": question,
            "chunks": json.loads(chunks_json) if chunks_json else [],
            "question_type": question_type or "",
        })
    return result


def save_message(
    role: str,
    content: str,
    question: str = "",
    chunks: list[dict] | None = None,
    question_type: str = "",
    db_path: Path | str | None = _DB_PATH,
) -> int:
    """Insert one message row and return its auto-assigned id."""
    chunks_json = json.dumps(chunks or [])
    with _get_conn(db_path) as conn:
        row_id = conn.insert_returning_id(
            "INSERT INTO history (role, content, question, chunks_json, question_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (role, content, question, chunks_json, question_type),
        )
        conn.commit()
        return row_id


def clear_history(db_path: Path | str | None = _DB_PATH) -> None:
    """Delete all rows from ``history`` (for testing / user reset).

    Does NOT touch ``audit_log`` — purging history is a user-driven UI action
    while audit_log is a compliance record that must outlive UI resets.
    """
    with _get_conn(db_path) as conn:
        conn.execute("DELETE FROM history")
        conn.commit()


def write_audit(
    question: str,
    answer: str,
    *,
    question_type: str = "",
    evidence: list[dict] | None = None,
    metadata_filters: dict | None = None,
    user_label: str = "",
    db_path: Path | str | None = _DB_PATH,
) -> int:
    """Persist one Q&A turn to ``audit_log`` and return its row id.

    ``evidence`` is the list of chunk dicts that were used to answer.
    Stored as JSON so the schema doesn't have to mirror the chunk shape
    exactly. ``has_unsupported`` is auto-derived by scanning the answer
    for ``[UNSUPPORTED`` — set by the safety post-processor.
    """
    has_unsupported = 1 if "[UNSUPPORTED" in (answer or "") else 0
    evidence_json = json.dumps(evidence or [], default=str)
    filters_json = json.dumps(metadata_filters or {}, default=str)

    with _get_conn(db_path) as conn:
        row_id = conn.insert_returning_id(
            "INSERT INTO audit_log "
            "(user_label, question, question_type, answer, evidence_json, has_unsupported, metadata_filters_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_label, question, question_type, answer, evidence_json, has_unsupported, filters_json),
        )
        conn.commit()
        return row_id


def read_audit(limit: int = 100, db_path: Path | str | None = _DB_PATH) -> list[dict]:
    """Return the most recent ``limit`` audit-log rows, newest first."""
    with _get_conn(db_path) as conn:
        rows = conn.fetchall(
            "SELECT id, created_at, user_label, question, question_type, answer, "
            "evidence_json, has_unsupported, metadata_filters_json "
            "FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    result: list[dict] = []
    for (
        row_id, created_at, user_label, question, question_type,
        answer, evidence_json, has_unsupported, filters_json,
    ) in rows:
        result.append({
            "id": row_id,
            "created_at": created_at,
            "user_label": user_label,
            "question": question,
            "question_type": question_type,
            "answer": answer,
            "evidence": json.loads(evidence_json) if evidence_json else [],
            "has_unsupported": bool(has_unsupported),
            "metadata_filters": json.loads(filters_json) if filters_json else {},
        })
    return result
