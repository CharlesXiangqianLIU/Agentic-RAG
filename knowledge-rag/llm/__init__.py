# llm/__init__.py
from config import LLM_PROVIDER
from llm.base import LLMProvider


def get_llm_provider() -> LLMProvider:
    """Factory: returns the active LLM provider based on LLM_PROVIDER env var."""
    if LLM_PROVIDER == "deepseek":
        from llm.deepseek_provider import DeepSeekProvider
        return DeepSeekProvider()
    if LLM_PROVIDER == "ollama":
        from llm.ollama_provider import OllamaProvider
        return OllamaProvider()
    from llm.anthropic_provider import AnthropicProvider
    return AnthropicProvider()
