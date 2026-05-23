# tests/test_llm_providers.py
import inspect
import pytest
from unittest.mock import MagicMock, patch
import anthropic
import openai
from llm.base import LLMProvider
from llm.anthropic_provider import AnthropicProvider
from llm.deepseek_provider import DeepSeekProvider


def test_anthropic_provider_is_llm_provider():
    provider = AnthropicProvider(api_key="dummy")
    assert isinstance(provider, LLMProvider)


def test_deepseek_provider_is_llm_provider():
    provider = DeepSeekProvider(base_url="http://localhost:8000/v1")
    assert isinstance(provider, LLMProvider)


def test_providers_share_same_complete_interface():
    a_sig = inspect.signature(AnthropicProvider.complete)
    d_sig = inspect.signature(DeepSeekProvider.complete)
    assert list(a_sig.parameters.keys()) == list(d_sig.parameters.keys())


def test_providers_share_same_stream_interface():
    a_sig = inspect.signature(AnthropicProvider.stream)
    d_sig = inspect.signature(DeepSeekProvider.stream)
    assert list(a_sig.parameters.keys()) == list(d_sig.parameters.keys())


def test_anthropic_provider_complete_returns_string():
    with patch("llm.anthropic_provider.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value.content = [MagicMock(text="Test answer")]
        provider = AnthropicProvider(api_key="dummy")
        result = provider.complete([{"role": "user", "content": "Hello"}])
        assert isinstance(result, str)
        assert result == "Test answer"


def test_anthropic_provider_sets_client_timeout():
    with patch("llm.anthropic_provider.anthropic.Anthropic") as mock_anthropic_cls:
        AnthropicProvider(api_key="dummy")
    assert "timeout" in mock_anthropic_cls.call_args.kwargs


def test_get_llm_provider_returns_anthropic_by_default():
    with patch("llm.LLM_PROVIDER", "anthropic"):
        from llm import get_llm_provider
        with patch("llm.anthropic_provider.anthropic"):
            provider = get_llm_provider()
            assert isinstance(provider, AnthropicProvider)


def test_complete_retries_on_rate_limit_error():
    """Test that complete() retries on RateLimitError."""
    with patch("llm.retry.time.sleep"):  # skip actual sleep delays
        with patch("llm.anthropic_provider.anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client

            # Raise RateLimitError twice, then succeed
            rate_limit_err = anthropic.RateLimitError(
                message="rate limit",
                response=MagicMock(headers={}, status_code=429),
                body={},
            )
            mock_client.messages.create.side_effect = [
                rate_limit_err,
                rate_limit_err,
                MagicMock(content=[MagicMock(text="Success")]),
            ]

            provider = AnthropicProvider(api_key="dummy")
            result = provider.complete([{"role": "user", "content": "Hello"}])

            assert result == "Success"
            assert mock_client.messages.create.call_count == 3


def test_complete_retries_on_server_error():
    """Test that complete() retries on APIStatusError with 5xx status."""
    with patch("llm.retry.time.sleep"):  # skip actual sleep delays
        with patch("llm.anthropic_provider.anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client

            # Raise 503 server error twice, then succeed
            api_err_503 = anthropic.APIStatusError(
                message="server error",
                response=MagicMock(headers={}, status_code=503),
                body={},
            )
            mock_client.messages.create.side_effect = [
                api_err_503,
                api_err_503,
                MagicMock(content=[MagicMock(text="Success")]),
            ]

            provider = AnthropicProvider(api_key="dummy")
            result = provider.complete([{"role": "user", "content": "Hello"}])

            assert result == "Success"
            assert mock_client.messages.create.call_count == 3


def test_complete_does_not_retry_on_client_error():
    """Test that complete() does NOT retry on APIStatusError with 4xx status."""
    with patch("llm.retry.time.sleep"):  # skip actual sleep delays
        with patch("llm.anthropic_provider.anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client

            # Raise 400 client error (should not retry)
            api_err_400 = anthropic.APIStatusError(
                message="bad request",
                response=MagicMock(headers={}, status_code=400),
                body={},
            )
            mock_client.messages.create.side_effect = api_err_400

            provider = AnthropicProvider(api_key="dummy")

            with pytest.raises(anthropic.APIStatusError):
                provider.complete([{"role": "user", "content": "Hello"}])

            # Should only be called once, no retries
            assert mock_client.messages.create.call_count == 1


def test_deepseek_complete_retries_on_rate_limit():
    """Test that DeepSeekProvider.complete() retries on RateLimitError."""
    with patch("llm.retry.time.sleep"):
        with patch("llm.deepseek_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client

            rate_limit_err = openai.RateLimitError(
                message="rate limit",
                response=MagicMock(headers={}, status_code=429),
                body={},
            )
            success_response = MagicMock()
            success_response.choices[0].message.content = "Success"
            mock_client.chat.completions.create.side_effect = [
                rate_limit_err,
                rate_limit_err,
                success_response,
            ]

            provider = DeepSeekProvider(base_url="http://localhost:8000/v1")
            result = provider.complete([{"role": "user", "content": "Hello"}])

            assert result == "Success"
            assert mock_client.chat.completions.create.call_count == 3


def test_deepseek_complete_forwards_system_prompt():
    with patch("llm.deepseek_provider.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        success_response = MagicMock()
        success_response.choices[0].message.content = "Success"
        mock_client.chat.completions.create.return_value = success_response

        provider = DeepSeekProvider(base_url="http://localhost:8000/v1")
        provider.complete(
            [{"role": "user", "content": "Hello"}],
            system="You are a planner.",
        )

        call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert call_messages[0] == {"role": "system", "content": "You are a planner."}
        assert call_messages[1] == {"role": "user", "content": "Hello"}


def test_anthropic_complete_forwards_timeout():
    with patch("llm.anthropic_provider.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value.content = [MagicMock(text="ok")]

        provider = AnthropicProvider(api_key="dummy")
        provider.complete([{"role": "user", "content": "Hello"}], timeout=12)

        assert mock_client.messages.create.call_args.kwargs["timeout"] == 12


def test_deepseek_provider_sets_client_timeout():
    with patch("llm.deepseek_provider.OpenAI") as mock_openai_cls:
        DeepSeekProvider(base_url="http://localhost:8000/v1")
    assert "timeout" in mock_openai_cls.call_args.kwargs


def test_deepseek_complete_retries_on_server_error():
    """Test that DeepSeekProvider.complete() retries on APIStatusError with 5xx status."""
    with patch("llm.retry.time.sleep"):
        with patch("llm.deepseek_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client

            api_err_503 = openai.APIStatusError(
                message="server error",
                response=MagicMock(headers={}, status_code=503),
                body={},
            )
            success_response = MagicMock()
            success_response.choices[0].message.content = "Success"
            mock_client.chat.completions.create.side_effect = [
                api_err_503,
                api_err_503,
                success_response,
            ]

            provider = DeepSeekProvider(base_url="http://localhost:8000/v1")
            result = provider.complete([{"role": "user", "content": "Hello"}])

            assert result == "Success"
            assert mock_client.chat.completions.create.call_count == 3


def test_stream_retries_on_rate_limit_error():
    """Test that stream() retries on RateLimitError."""
    with patch("llm.retry.time.sleep"):
        with patch("llm.anthropic_provider.anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client

            rate_limit_err = anthropic.RateLimitError(
                message="rate limit",
                response=MagicMock(headers={}, status_code=429),
                body={},
            )
            # context manager mock: __enter__ yields mock_stream_obj
            mock_stream_ctx = MagicMock()
            mock_stream_ctx.__enter__ = MagicMock(return_value=MagicMock(text_stream=iter(["token1", "token2"])))
            mock_stream_ctx.__exit__ = MagicMock(return_value=False)

            mock_client.messages.stream.side_effect = [
                rate_limit_err,
                mock_stream_ctx,
            ]

            provider = AnthropicProvider(api_key="dummy")
            result = list(provider.stream([{"role": "user", "content": "Hello"}]))

            assert result == ["token1", "token2"]
            assert mock_client.messages.stream.call_count == 2


def test_stream_collects_all_tokens():
    """stream() returns an iterator of all tokens from the stream."""
    with patch("llm.anthropic_provider.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(
            return_value=MagicMock(text_stream=iter(["The ", "yield ", "was ", "87%."]))
        )
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.messages.stream.return_value = mock_stream_ctx

        provider = AnthropicProvider(api_key="dummy")
        result = "".join(provider.stream([{"role": "user", "content": "Q"}]))
        assert result == "The yield was 87%."


def test_anthropic_stream_forwards_timeout():
    with patch("llm.anthropic_provider.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=MagicMock(text_stream=iter(["ok"])))
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.messages.stream.return_value = mock_stream_ctx

        provider = AnthropicProvider(api_key="dummy")
        list(provider.stream([{"role": "user", "content": "Q"}], timeout=9))

        assert mock_client.messages.stream.call_args.kwargs["timeout"] == 9


def test_deepseek_stream_retries_on_rate_limit():
    """DeepSeekProvider.stream() retries on RateLimitError, matching Anthropic parity."""
    with patch("llm.retry.time.sleep"):
        with patch("llm.deepseek_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client

            rate_limit_err = openai.RateLimitError(
                message="rate limit",
                response=MagicMock(headers={}, status_code=429),
                body={},
            )

            def make_chunk(content):
                chunk = MagicMock()
                chunk.choices[0].delta.content = content
                return chunk

            mock_client.chat.completions.create.side_effect = [
                rate_limit_err,
                [make_chunk("The "), make_chunk("answer.")],
            ]

            provider = DeepSeekProvider(base_url="http://localhost:8000/v1")
            result = "".join(provider.stream([{"role": "user", "content": "Hello"}]))

            assert result == "The answer."
            assert mock_client.chat.completions.create.call_count == 2


def test_deepseek_stream_collects_all_tokens():
    """DeepSeekProvider.stream() collects all non-None delta tokens."""
    with patch("llm.deepseek_provider.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        def make_chunk(content):
            chunk = MagicMock()
            chunk.choices[0].delta.content = content
            return chunk

        mock_client.chat.completions.create.return_value = [
            make_chunk("token1"),
            make_chunk(None),   # None deltas should be skipped
            make_chunk("token2"),
        ]

        provider = DeepSeekProvider(base_url="http://localhost:8000/v1")
        result = list(provider.stream([{"role": "user", "content": "Q"}]))
        assert result == ["token1", "token2"]


def test_deepseek_stream_forwards_system_prompt():
    with patch("llm.deepseek_provider.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        chunk = MagicMock()
        chunk.choices[0].delta.content = "token"
        mock_client.chat.completions.create.return_value = [chunk]

        provider = DeepSeekProvider(base_url="http://localhost:8000/v1")
        list(provider.stream([{"role": "user", "content": "Q"}], system="You are a critic."))

        call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert call_messages[0] == {"role": "system", "content": "You are a critic."}
        assert call_messages[1] == {"role": "user", "content": "Q"}


def test_deepseek_complete_forwards_timeout():
    with patch("llm.deepseek_provider.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        success_response = MagicMock()
        success_response.choices[0].message.content = "Success"
        mock_client.chat.completions.create.return_value = success_response

        provider = DeepSeekProvider(base_url="http://localhost:8000/v1")
        provider.complete([{"role": "user", "content": "Hello"}], timeout=7)

        assert mock_client.chat.completions.create.call_args.kwargs["timeout"] == 7


def test_deepseek_stream_forwards_timeout():
    with patch("llm.deepseek_provider.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        chunk = MagicMock()
        chunk.choices[0].delta.content = "token"
        mock_client.chat.completions.create.return_value = [chunk]

        provider = DeepSeekProvider(base_url="http://localhost:8000/v1")
        list(provider.stream([{"role": "user", "content": "Q"}], timeout=7))

        assert mock_client.chat.completions.create.call_args.kwargs["timeout"] == 7


def test_deepseek_complete_does_not_retry_on_client_error():
    """Test that DeepSeekProvider.complete() does NOT retry on APIStatusError with 4xx status."""
    with patch("llm.retry.time.sleep"):
        with patch("llm.deepseek_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client

            api_err_400 = openai.APIStatusError(
                message="bad request",
                response=MagicMock(headers={}, status_code=400),
                body={},
            )
            mock_client.chat.completions.create.side_effect = api_err_400

            provider = DeepSeekProvider(base_url="http://localhost:8000/v1")

            with pytest.raises(openai.APIStatusError):
                provider.complete([{"role": "user", "content": "Hello"}])

            assert mock_client.chat.completions.create.call_count == 1
