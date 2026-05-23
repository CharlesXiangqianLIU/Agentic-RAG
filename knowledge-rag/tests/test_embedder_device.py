"""Tests for embedder device + precision detection.

The embedder itself takes ~3 GB of memory to instantiate, so we only
test the small selection helpers — _detect_device_and_precision() and
_autodetect_device(). The actual BGEM3FlagModel constructor is never
called in this file.
"""
import sys
from unittest.mock import MagicMock

import pytest

from retrieval.embedder import _autodetect_device, _detect_device_and_precision


def _install_fake_torch(monkeypatch, *, cuda: bool, mps: bool) -> None:
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = cuda
    if mps:
        fake_torch.backends.mps.is_available.return_value = True
    else:
        # Either no mps attribute or mps available -> False
        fake_torch.backends.mps.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_explicit_device_override_wins(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.delenv("EMBEDDING_USE_FP16", raising=False)
    _install_fake_torch(monkeypatch, cuda=True, mps=False)  # cuda would otherwise win
    device, use_fp16 = _detect_device_and_precision()
    assert device == "cpu"
    assert use_fp16 is False  # CPU defaults to fp32


def test_autodetect_prefers_cuda(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("EMBEDDING_USE_FP16", raising=False)
    _install_fake_torch(monkeypatch, cuda=True, mps=True)
    device, use_fp16 = _detect_device_and_precision()
    assert device == "cuda"
    assert use_fp16 is True


def test_autodetect_falls_back_to_mps_when_no_cuda(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("EMBEDDING_USE_FP16", raising=False)
    _install_fake_torch(monkeypatch, cuda=False, mps=True)
    device, use_fp16 = _detect_device_and_precision()
    assert device == "mps"
    assert use_fp16 is True


def test_autodetect_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("EMBEDDING_USE_FP16", raising=False)
    _install_fake_torch(monkeypatch, cuda=False, mps=False)
    device, use_fp16 = _detect_device_and_precision()
    assert device == "cpu"
    assert use_fp16 is False


def test_fp16_override_forces_on(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("EMBEDDING_USE_FP16", "1")
    _install_fake_torch(monkeypatch, cuda=False, mps=False)
    _, use_fp16 = _detect_device_and_precision()
    assert use_fp16 is True


def test_fp16_override_forces_off(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")
    monkeypatch.setenv("EMBEDDING_USE_FP16", "0")
    _install_fake_torch(monkeypatch, cuda=True, mps=False)
    _, use_fp16 = _detect_device_and_precision()
    assert use_fp16 is False


def test_unknown_device_value_falls_through_to_autodetect(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DEVICE", "nonsense")
    monkeypatch.delenv("EMBEDDING_USE_FP16", raising=False)
    _install_fake_torch(monkeypatch, cuda=False, mps=False)
    device, use_fp16 = _detect_device_and_precision()
    assert device == "cpu"
    assert use_fp16 is False


def test_autodetect_handles_missing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    # Force the import inside _autodetect_device to fail by making the
    # entry None (which raises ImportError on `import torch`).
    # We can't easily simulate ImportError via sys.modules, so use sentinel.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _autodetect_device() == "cpu"
