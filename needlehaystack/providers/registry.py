"""Registry of provider plugins keyed by `(runtime.sdk, runtime.api)`.

A "provider plugin" is a factory: it takes a parsed `ModelConfig` and
returns a configured `ModelProvider`. Real provider plugins
(`anthropic-python` / `messages`, `openai-python` / `responses`, …)
land in Phase 7. The only built-in for v2.0 is the `fake/fake` provider
that backs every test config.

Third parties register their own via `register_provider(...)`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import ModelProvider
from .fake import FakeProvider

if TYPE_CHECKING:
    from ..config.schema import ModelConfig


# (sdk, api) → factory
ProviderKey = tuple[str, str]
ProviderFactory = Callable[["ModelConfig"], ModelProvider]
PROVIDER_REGISTRY: dict[ProviderKey, ProviderFactory] = {}


def register_provider(sdk: str, api: str, factory: ProviderFactory) -> None:
    """Register a provider plugin under `(sdk, api)`.

    Raises `ValueError` on duplicate keys — same reasoning as the task
    registry: duplicates usually indicate a copy-paste bug.
    """
    key = (sdk, api)
    if key in PROVIDER_REGISTRY:
        raise ValueError(f"provider plugin {key!r} is already registered")
    PROVIDER_REGISTRY[key] = factory


def unregister_provider(sdk: str, api: str) -> None:
    """Remove a registration. Used in tests."""
    PROVIDER_REGISTRY.pop((sdk, api), None)


def build_provider(config: ModelConfig) -> ModelProvider:
    """Construct a provider for `config`.

    Lazy import of the config schema avoids a circular import; the
    registry module itself doesn't depend on pydantic at import time.
    """
    key = (config.runtime.sdk, config.runtime.api)
    factory = PROVIDER_REGISTRY.get(key)
    if factory is None:
        raise KeyError(
            f"no provider registered for runtime {key!r}; registered: {sorted(PROVIDER_REGISTRY)}"
        )
    return factory(config)


def assert_provider_registered(config: ModelConfig) -> None:
    """Verify a plugin exists for `config.runtime` without constructing it.

    Used by `niah validate` so config validation never requires API
    credentials or a working SDK install.
    """
    key = (config.runtime.sdk, config.runtime.api)
    if key not in PROVIDER_REGISTRY:
        raise KeyError(
            f"no provider registered for runtime {key!r}; registered: {sorted(PROVIDER_REGISTRY)}"
        )


# ---------------------------------------------------------------------------
# Built-in: fake
# ---------------------------------------------------------------------------


def _build_fake(config: ModelConfig) -> ModelProvider:
    """Factory for the in-process FakeProvider used by tests and CI."""
    # FakeProvider reads its mode + canned response from `request:` so
    # tests can drive it entirely from YAML.
    extra = config.request.model_dump(exclude={"model", "max_tokens", "temperature", "stream"})
    return FakeProvider(
        id=config.id,
        request_model=config.request.model,
        mode=extra.get("mode", "canned"),
        response_text=extra.get("response_text", "ok"),
        fail_until=extra.get("fail_until", 0),
        fail_message=extra.get("fail_message", "synthetic failure"),
    )


register_provider("fake", "fake", _build_fake)
