"""OpenAI provider plugin (Chat Completions API).

Registered under `(runtime.sdk, runtime.api) = ("openai-python", "chat_completions")`.

The provider holds a single `AsyncOpenAI` client and forwards the model
config's `request:` block as kwargs to `chat.completions.create`. The
runner only sees `complete(system, user) -> Completion`.

Streaming is not honored — `stream` is stripped from the forwarded
kwargs. The runner's contract is "one prompt, one response, with usage";
aggregating a stream into that shape is future work.
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
class OpenAIProvider:
    """`ModelProvider` backed by the OpenAI Python SDK's Chat Completions."""

    id: str
    request_model: str
    client: Any  # openai.AsyncOpenAI — typed as Any so tests can inject mocks
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    async def complete(self, system: str, user: str) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        resp = await self.client.chat.completions.create(
            model=self.request_model,
            messages=messages,
            **self.extra_kwargs,
        )

        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage: Usage | None = None
        if getattr(resp, "usage", None) is not None:
            usage = Usage(
                input_tokens=int(resp.usage.prompt_tokens),
                output_tokens=int(resp.usage.completion_tokens),
            )
        return Completion(text=text, usage=usage)


def _build_openai(config: ModelConfig) -> OpenAIProvider:
    """Factory: build an `OpenAIProvider` from a parsed `ModelConfig`."""
    from openai import AsyncOpenAI  # lazy: only imported when the plugin runs

    api_key = os.environ.get(config.client.api_key_env) if config.client.api_key_env else None
    client = AsyncOpenAI(api_key=api_key)
    # Forward every `request:` key except the ones we handle explicitly.
    # `stream` is stripped because our `complete()` returns a single
    # aggregated `Completion`, not a stream.
    extras = config.request.model_dump(
        exclude={"model", "stream"},
        exclude_none=True,
    )
    return OpenAIProvider(
        id=config.id,
        request_model=config.request.model,
        client=client,
        extra_kwargs=extras,
    )


register_provider("openai-python", "chat_completions", _build_openai)
