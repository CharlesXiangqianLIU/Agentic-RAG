"""Shared retry helper in llm/retry.py."""
import pytest

from llm.retry import always_retry, status_at_least, with_retry


class _MyError(Exception):
    pass


class _StatusError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"status {status_code}")


def test_returns_value_on_first_success():
    calls = {"n": 0}

    def _fn():
        calls["n"] += 1
        return "ok"

    assert with_retry(_fn, retriable=[(_MyError, always_retry)]) == "ok"
    assert calls["n"] == 1


def test_retries_on_matching_exception(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)  # don't actually wait
    calls = {"n": 0}

    def _fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _MyError("transient")
        return "third time lucky"

    result = with_retry(_fn, retriable=[(_MyError, always_retry)], base_delay=0.001)
    assert result == "third time lucky"
    assert calls["n"] == 3


def test_does_not_retry_on_unlisted_exception():
    calls = {"n": 0}

    def _fn():
        calls["n"] += 1
        raise RuntimeError("not in rules")

    with pytest.raises(RuntimeError):
        with_retry(_fn, retriable=[(_MyError, always_retry)])
    assert calls["n"] == 1


def test_predicate_can_short_circuit_specific_subtypes(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    def _fn():
        calls["n"] += 1
        raise _StatusError(400)  # client error, predicate says don't retry

    with pytest.raises(_StatusError):
        with_retry(_fn, retriable=[(_StatusError, status_at_least(500))], base_delay=0.001)
    assert calls["n"] == 1


def test_status_at_least_retries_5xx(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    raised = [_StatusError(503), _StatusError(503), None]

    def _fn():
        nxt = raised.pop(0)
        if nxt is not None:
            raise nxt
        return "recovered"

    assert with_retry(
        _fn, retriable=[(_StatusError, status_at_least(500))],
        base_delay=0.001, max_retries=3,
    ) == "recovered"


def test_status_at_least_handles_missing_attribute():
    # If the exception type doesn't carry status_code, predicate returns False -> don't retry.
    class _NoStatus(Exception):
        pass

    with pytest.raises(_NoStatus):
        with_retry(
            lambda: (_ for _ in ()).throw(_NoStatus()),
            retriable=[(_NoStatus, status_at_least(500))],
        )


def test_exhausts_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    def _fn():
        calls["n"] += 1
        raise _MyError("permanent")

    with pytest.raises(_MyError):
        with_retry(_fn, retriable=[(_MyError, always_retry)], max_retries=2, base_delay=0.001)
    assert calls["n"] == 3  # initial + 2 retries
