"""Cohere provider plugin (chat API).

Registered under `(runtime.sdk, runtime.api) = ("cohere-python", "chat")`.

Cohere's v1 chat endpoint takes the user turn as `message=` and the
system instructions as `preamble=`. We adapt the `system`/`user` shape
the runner uses to that signature; everything else from `request:` (e.g.
`temperature`) passes through unchanged.

A single `AsyncClient` is constructed at factory time and reused across
calls.
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
class CohereProvider:
    """`ModelProvider` backed by the Cohere Python SDK's chat API."""

    id: str
    request_model: str
    client: Any  # cohere.AsyncClient
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    async def complete(self, system: str, user: str) -> Completion:
        call_kwargs: dict[str, Any] = dict(self.extra_kwargs)
        call_kwargs["model"] = self.request_model
        call_kwargs["message"] = user
        if system:
            call_kwargs["preamble"] = system

        resp = await self.client.chat(**call_kwargs)
        text = getattr(resp, "text", "") or ""
        usage = _extract_usage(resp)
        return Completion(text=text, usage=usage)


def _extract_usage(resp: Any) -> Usage | None:
    """Cohere reports usage via `meta.tokens.{input,output}_count` in v5."""
    meta = getattr(resp, "meta", None)
    if meta is None:
        return None
    tokens = getattr(meta, "tokens", None)
    if tokens is None:
        return None
    try:
        return Usage(
            input_tokens=int(tokens.input_count),
            output_tokens=int(tokens.output_count),
        )
    except AttributeError:
        return None


def _build_cohere(config: ModelConfig) -> CohereProvider:
    """Factory: build a `CohereProvider` from a parsed `ModelConfig`."""
    from cohere import AsyncClient  # lazy import

    api_key = os.environ.get(config.client.api_key_env) if config.client.api_key_env else None
    client = AsyncClient(api_key=api_key)
    extras = config.request.model_dump(
        exclude={"model", "stream"},
        exclude_none=True,
    )
    return CohereProvider(
        id=config.id,
        request_model=config.request.model,
        client=client,
        extra_kwargs=extras,
    )


register_provider("cohere-python", "chat", _build_cohere)
