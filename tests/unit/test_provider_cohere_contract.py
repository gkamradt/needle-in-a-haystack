"""Contract tests for the Cohere provider plugin (no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from needlehaystack.config.schema import ModelConfig
from needlehaystack.core.types import Usage
from needlehaystack.providers.base import ModelProvider
from needlehaystack.providers.cohere import CohereProvider, _build_cohere


def _fake_cohere_response(
    text: str = "hi from cohere",
    input_count: int = 30,
    output_count: int = 5,
) -> object:
    return SimpleNamespace(
        text=text,
        meta=SimpleNamespace(
            tokens=SimpleNamespace(input_count=input_count, output_count=output_count)
        ),
    )


def _provider_with_mock_client(
    extra_kwargs: dict | None = None,
) -> tuple[CohereProvider, AsyncMock]:
    chat = AsyncMock(return_value=_fake_cohere_response())
    client = SimpleNamespace(chat=chat)
    provider = CohereProvider(
        id="cohere-test",
        request_model="command-test",
        client=client,
        extra_kwargs=extra_kwargs or {},
    )
    return provider, chat


def test_satisfies_protocol() -> None:
    provider, _ = _provider_with_mock_client()
    assert isinstance(provider, ModelProvider)


def test_complete_adapts_system_user_into_preamble_and_message() -> None:
    provider, chat = _provider_with_mock_client()
    completion = asyncio.run(provider.complete(system="be brief", user="hi"))

    chat.assert_awaited_once()
    kwargs = chat.call_args.kwargs
    assert kwargs["model"] == "command-test"
    assert kwargs["message"] == "hi"
    assert kwargs["preamble"] == "be brief"
    assert completion.text == "hi from cohere"
    assert completion.usage == Usage(input_tokens=30, output_tokens=5)


def test_complete_omits_preamble_when_system_empty() -> None:
    provider, chat = _provider_with_mock_client()
    asyncio.run(provider.complete(system="", user="hi"))
    assert "preamble" not in chat.call_args.kwargs


def test_extra_kwargs_forwarded() -> None:
    provider, chat = _provider_with_mock_client(extra_kwargs={"temperature": 0.2})
    asyncio.run(provider.complete(system="", user="hi"))
    assert chat.call_args.kwargs["temperature"] == 0.2


def test_missing_usage_returns_none() -> None:
    response = SimpleNamespace(text="x", meta=None)
    chat = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=chat)
    provider = CohereProvider(id="x", request_model="m", client=client)
    completion = asyncio.run(provider.complete(system="", user="hi"))
    assert completion.usage is None


def test_single_async_client_reused_across_calls() -> None:
    """One client instance per provider; multiple `complete` calls share it."""
    provider, chat = _provider_with_mock_client()
    asyncio.run(provider.complete(system="", user="a"))
    asyncio.run(provider.complete(system="", user="b"))
    assert chat.await_count == 2
    # All calls went through the same chat object on the same client.
    assert chat is provider.client.chat


def test_factory_creates_single_client_and_strips_stream() -> None:
    cfg = ModelConfig.model_validate(
        {
            "id": "cohere-command-test",
            "runtime": {"sdk": "cohere-python", "api": "chat"},
            "client": {"api_key_env": "COHERE_API_KEY_TEST"},
            "request": {"model": "command-r", "temperature": 0.3, "stream": True},
        }
    )
    with patch("cohere.AsyncClient") as mock_cls:
        mock_cls.return_value = "sentinel-client"
        provider = _build_cohere(cfg)

    assert isinstance(provider, CohereProvider)
    assert provider.client == "sentinel-client"
    assert mock_cls.call_count == 1  # construction not deferred per-call
    assert provider.extra_kwargs == {"temperature": 0.3}
