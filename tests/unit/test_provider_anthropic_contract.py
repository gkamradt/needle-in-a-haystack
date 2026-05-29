"""Contract tests for the Anthropic provider plugin (no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from needlehaystack.config.schema import ModelConfig
from needlehaystack.core.types import Usage
from needlehaystack.providers.anthropic import AnthropicProvider, _build_anthropic
from needlehaystack.providers.base import ModelProvider


def _block(type_: str, text: str = "") -> object:
    return SimpleNamespace(type=type_, text=text)


def _fake_anthropic_response(
    *,
    blocks: list | None = None,
    input_tokens: int = 100,
    output_tokens: int = 25,
) -> object:
    return SimpleNamespace(
        content=blocks if blocks is not None else [_block("text", "hi from claude")],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _provider_with_mock_client(
    extra_kwargs: dict | None = None,
) -> tuple[AnthropicProvider, AsyncMock]:
    create = AsyncMock(return_value=_fake_anthropic_response())
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    provider = AnthropicProvider(
        id="anthropic-test",
        request_model="claude-test",
        client=client,
        extra_kwargs=extra_kwargs or {"max_tokens": 1024},
    )
    return provider, create


def test_satisfies_protocol() -> None:
    provider, _ = _provider_with_mock_client()
    assert isinstance(provider, ModelProvider)


def test_complete_uses_messages_api_with_system_as_top_level_kwarg() -> None:
    provider, create = _provider_with_mock_client()
    completion = asyncio.run(provider.complete(system="You are concise.", user="hi"))

    create.assert_awaited_once()
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "claude-test"
    assert kwargs["system"] == "You are concise."
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert completion.text == "hi from claude"
    assert completion.usage == Usage(input_tokens=100, output_tokens=25)


def test_complete_omits_system_when_empty() -> None:
    provider, create = _provider_with_mock_client()
    asyncio.run(provider.complete(system="", user="hi"))
    assert "system" not in create.call_args.kwargs


def test_extra_kwargs_pass_through_thinking_and_output_config() -> None:
    provider, create = _provider_with_mock_client(
        extra_kwargs={
            "max_tokens": 12000,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        }
    )
    asyncio.run(provider.complete(system="", user="hi"))
    kwargs = create.call_args.kwargs
    assert kwargs["max_tokens"] == 12000
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "medium"}


def test_text_extraction_concatenates_only_text_blocks() -> None:
    response = _fake_anthropic_response(
        blocks=[
            _block("thinking", "this is reasoning that should be ignored"),
            _block("text", "Hello"),
            _block("text", " world."),
            _block("tool_use"),  # no text attr — should be ignored
        ]
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    provider = AnthropicProvider(id="x", request_model="m", client=client, extra_kwargs={})

    completion = asyncio.run(provider.complete(system="", user="hi"))
    assert completion.text == "Hello world."


def test_factory_builds_from_full_example_model_config() -> None:
    cfg = ModelConfig.model_validate(
        {
            "id": "opus-medium",
            "runtime": {"sdk": "anthropic-python", "api": "messages"},
            "client": {"api_key_env": "ANTHROPIC_API_KEY_TEST"},
            "request": {
                "model": "claude-opus-4",
                "max_tokens": 120000,
                "stream": True,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "medium"},
            },
            "pricing": {"input": 5.0, "output": 25.0},
        }
    )
    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_cls.return_value = "sentinel-client"
        provider = _build_anthropic(cfg)

    assert isinstance(provider, AnthropicProvider)
    assert provider.id == "opus-medium"
    assert provider.request_model == "claude-opus-4"
    # stream is stripped, model is hoisted, the rest flows.
    assert provider.extra_kwargs == {
        "max_tokens": 120000,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
    }
