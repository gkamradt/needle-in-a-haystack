"""Contract tests for the OpenAI provider plugin (no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from needlehaystack.config.schema import ModelConfig
from needlehaystack.core.types import Usage
from needlehaystack.providers.base import ModelProvider
from needlehaystack.providers.openai import OpenAIProvider, _build_openai


def _fake_openai_response(text: str = "hello", prompt: int = 12, completion: int = 7) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def _provider_with_mock_client(
    extra_kwargs: dict | None = None,
) -> tuple[OpenAIProvider, AsyncMock]:
    create = AsyncMock(return_value=_fake_openai_response())
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = OpenAIProvider(
        id="openai-test",
        request_model="gpt-test",
        client=client,
        extra_kwargs=extra_kwargs or {},
    )
    return provider, create


def test_satisfies_protocol() -> None:
    provider, _ = _provider_with_mock_client()
    assert isinstance(provider, ModelProvider)


def test_complete_calls_sdk_with_messages_and_returns_text_and_usage() -> None:
    provider, create = _provider_with_mock_client()
    completion = asyncio.run(provider.complete(system="you are a system", user="hello"))

    create.assert_awaited_once()
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["messages"] == [
        {"role": "system", "content": "you are a system"},
        {"role": "user", "content": "hello"},
    ]
    assert completion.text == "hello"
    assert completion.usage == Usage(input_tokens=12, output_tokens=7)


def test_complete_omits_system_when_empty() -> None:
    provider, create = _provider_with_mock_client()
    asyncio.run(provider.complete(system="", user="just user"))

    kwargs = create.call_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "just user"}]


def test_extra_kwargs_forwarded_verbatim() -> None:
    provider, create = _provider_with_mock_client(
        extra_kwargs={"temperature": 0.7, "max_tokens": 128, "top_p": 0.95},
    )
    asyncio.run(provider.complete(system="", user="hi"))

    kwargs = create.call_args.kwargs
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 128
    assert kwargs["top_p"] == 0.95


def test_response_without_usage_returns_none_usage() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
        usage=None,
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = OpenAIProvider(id="x", request_model="m", client=client)

    completion = asyncio.run(provider.complete(system="", user="hi"))
    assert completion.text == "x"
    assert completion.usage is None


def test_factory_strips_stream_and_forwards_request_kwargs() -> None:
    cfg = ModelConfig.model_validate(
        {
            "id": "gpt-4o-test",
            "runtime": {"sdk": "openai-python", "api": "chat_completions"},
            "client": {"api_key_env": "OPENAI_API_KEY_TEST"},
            "request": {
                "model": "gpt-4o",
                "temperature": 0.0,
                "max_tokens": 1024,
                "stream": True,
                "top_p": 0.9,
            },
        }
    )
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = "sentinel-client"
        provider = _build_openai(cfg)

    assert isinstance(provider, OpenAIProvider)
    assert provider.id == "gpt-4o-test"
    assert provider.request_model == "gpt-4o"
    assert provider.client == "sentinel-client"
    # `model` and `stream` are excluded; everything else forwarded.
    assert provider.extra_kwargs == {"temperature": 0.0, "max_tokens": 1024, "top_p": 0.9}
