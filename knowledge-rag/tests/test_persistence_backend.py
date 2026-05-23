"""Backend selection in frontend/persistence.py.

We only exercise the selection logic and the SQLite path here. The
Postgres path is only reachable when ``psycopg`` is installed; that's
verified by the import-error message instead.
"""
import pytest

from frontend.persistence import (
    _is_postgres_url,
    _resolve_dsn,
)


def test_is_postgres_url_recognises_both_schemes():
    assert _is_postgres_url("postgres://user:pw@host/db")
    assert _is_postgres_url("postgresql://user:pw@host/db")


def test_is_postgres_url_rejects_other_schemes():
    assert not _is_postgres_url(None)
    assert not _is_postgres_url("")
    assert not _is_postgres_url("/tmp/foo.db")
    assert not _is_postgres_url("sqlite:///path")


def test_resolve_dsn_explicit_db_path_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://shouldnotwin")
    chosen = _resolve_dsn(tmp_path / "explicit.db")
    assert chosen == tmp_path / "explicit.db"


def test_resolve_dsn_env_var_wins_over_default(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://envwin")
    chosen = _resolve_dsn(None)
    assert chosen == "postgresql://envwin"


def test_resolve_dsn_default_when_nothing_set(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from frontend.persistence import _DB_PATH
    assert _resolve_dsn(None) == _DB_PATH


def test_postgres_url_without_psycopg_raises_helpful_error(monkeypatch):
    """If psycopg isn't installed and DATABASE_URL is postgres, the error
    must name the install hint instead of a bare ImportError."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://nohost/nodb")

    # Pretend psycopg is missing.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psycopg":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from frontend.persistence import _get_conn

    with pytest.raises(RuntimeError, match="pip install -e"):
        with _get_conn(None):
            pass
