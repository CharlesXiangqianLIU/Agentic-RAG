"""Model-aware MAX_CONTEXT_TOKENS defaulting in config.py."""
import importlib

import pytest


def _reload_config(monkeypatch, **env_overrides):
    """Apply env overrides, then re-import config so module-level defaults re-resolve."""
    for k, v in env_overrides.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import config
    return importlib.reload(config)


def test_explicit_env_var_wins(monkeypatch):
    cfg = _reload_config(monkeypatch, MAX_CONTEXT_TOKENS="12345", LLM_PROVIDER="anthropic")
    assert cfg.MAX_CONTEXT_TOKENS == 12345


def test_anthropic_default_picks_large_window(monkeypatch):
    cfg = _reload_config(
        monkeypatch,
        MAX_CONTEXT_TOKENS=None,
        LLM_PROVIDER="anthropic",
        ANTHROPIC_MODEL="claude-sonnet-4-6",
    )
    # 40% of 200_000 = 80_000
    assert cfg.MAX_CONTEXT_TOKENS == 80_000


def test_deepseek_default(monkeypatch):
    cfg = _reload_config(
        monkeypatch,
        MAX_CONTEXT_TOKENS=None,
        LLM_PROVIDER="deepseek",
        VLLM_MODEL="deepseek-ai/DeepSeek-R1-Distill-32B",
    )
    # 40% of 64_000 = 25_600
    assert cfg.MAX_CONTEXT_TOKENS == 25_600


def test_unknown_model_falls_back_to_8000(monkeypatch):
    cfg = _reload_config(
        monkeypatch,
        MAX_CONTEXT_TOKENS=None,
        LLM_PROVIDER="anthropic",
        ANTHROPIC_MODEL="some-unknown-model-x",
    )
    assert cfg.MAX_CONTEXT_TOKENS == 8000


def test_small_window_model_does_not_go_below_8000(monkeypatch):
    """Even for a 8_192-window gemma the budget stays at the 8_000 floor."""
    cfg = _reload_config(
        monkeypatch,
        MAX_CONTEXT_TOKENS=None,
        LLM_PROVIDER="ollama",
        OLLAMA_MODEL="gemma4:27b",
    )
    assert cfg.MAX_CONTEXT_TOKENS == 8000  # max(8000, 8192*0.4=3276)


# Restore default state for the rest of the test session.
@pytest.fixture(autouse=True, scope="module")
def _restore_config_at_module_end():
    yield
    import config
    importlib.reload(config)
