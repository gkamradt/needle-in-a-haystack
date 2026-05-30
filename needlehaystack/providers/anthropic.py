"""Anthropic provider plugin (Messages API).

Registered under `(runtime.sdk, runtime.api) = ("anthropic-python", "messages")`.

The provider forwards the model config's `request:` block as kwargs to
`messages.create`, with two notable behaviors:

- Anthropic's API takes `system` as a top-level kwarg (not a message),
  so we pass it that way.
- Provider-specific knobs like `thinking` and `output_config` from the
  example config flow through untouched, courtesy of `extra="allow"` on
  `ModelRequest`.

Streaming is stripped from the forwarded kwargs — same rationale as the
OpenAI provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.types import Usage
from .base import Completion
from .registry import register_provider

if TYPE_CHECKING:
    from ..config.schema import ModelConfig


@dataclass(slots=True)
class AnthropicProvider:
    """`ModelProvider` backed by the Anthropic Python SDK Messages API."""

    id: str
    request_model: str
    client: Any  # anthropic.AsyncAnthropic
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    async def complete(self, system: str, user: str) -> Completion:
        # `system` is a top-level kwarg in the Messages API. Omit it
        # entirely when empty so we don't send `system=""` (which is
        # technically valid but ugly).
        call_kwargs: dict[str, Any] = dict(self.extra_kwargs)
        call_kwargs["model"] = self.request_model
        call_kwargs["messages"] = [{"role": "user", "content": user}]
        if system:
            call_kwargs["system"] = system

        resp = await self.client.messages.create(**call_kwargs)
        text = _extract_text(resp)
        usage: Usage | None = None
        if getattr(resp, "usage", None) is not None:
            usage = Usage(
                input_tokens=int(resp.usage.input_tokens),
                output_tokens=int(resp.usage.output_tokens),
            )
        return Completion(text=text, usage=usage)


def _extract_text(resp: Any) -> str:
    """Concatenate the `text` blocks of an Anthropic Messages response.

    Responses come back as a list of content blocks (`text`, `thinking`,
    `tool_use`, etc.). We only emit `text` blocks; thinking content stays
    on the response object for callers that want it.
    """
    content = getattr(resp, "content", None) or []
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


def _build_anthropic(config: ModelConfig) -> AnthropicProvider:
    """Factory: build an `AnthropicProvider` from a parsed `ModelConfig`."""
    from anthropic import AsyncAnthropic  # lazy import

    api_key = os.environ.get(config.client.api_key_env) if config.client.api_key_env else None
    client = AsyncAnthropic(api_key=api_key)
    extras = config.request.model_dump(
        exclude={"model", "stream"},
        exclude_none=True,
    )
    return AnthropicProvider(
        id=config.id,
        request_model=config.request.model,
        client=client,
        extra_kwargs=extras,
    )


register_provider("anthropic-python", "messages", _build_anthropic)
