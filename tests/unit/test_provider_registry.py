"""Tests for the provider registry's `(sdk, api)` resolution."""

from __future__ import annotations

import pytest

from needlehaystack.providers import (
    PROVIDER_REGISTRY,
    AnthropicProvider,
    CohereProvider,
    FakeProvider,
    OpenAIProvider,
    build_provider,
    register_provider,
    unregister_provider,
)


def test_all_builtin_pairs_registered() -> None:
    for key in [
        ("fake", "fake"),
        ("openai-python", "chat_completions"),
        ("anthropic-python", "messages"),
        ("cohere-python", "chat"),
    ]:
        assert key in PROVIDER_REGISTRY, f"missing built-in registration for {key}"


def test_unknown_pair_raises_with_registered_keys() -> None:
    from needlehaystack.config.schema import ModelConfig

    cfg = ModelConfig.model_validate(
        {
            "id": "x",
            "runtime": {"sdk": "made-up-sdk", "api": "nope"},
            "request": {"model": "m"},
        }
    )
    with pytest.raises(KeyError, match="no provider registered") as e:
        build_provider(cfg)
    # Friendly error message names the registered keys.
    msg = str(e.value)
    for known in ("fake", "openai-python", "anthropic-python", "cohere-python"):
        assert known in msg


def test_duplicate_registration_raises() -> None:
    def _factory(_):  # type: ignore[no-untyped-def]
        return None  # pragma: no cover

    with pytest.raises(ValueError, match="already registered"):
        register_provider("fake", "fake", _factory)


def test_third_party_registration_round_trip() -> None:
    from needlehaystack.config.schema import ModelConfig
    from needlehaystack.providers.fake import FakeProvider as _Fake

    def _factory(config):  # type: ignore[no-untyped-def]
        return _Fake(id=config.id, request_model=config.request.model)

    try:
        register_provider("ext-sdk", "ext-api", _factory)
        cfg = ModelConfig.model_validate(
            {
                "id": "ext",
                "runtime": {"sdk": "ext-sdk", "api": "ext-api"},
                "request": {"model": "ext-model"},
            }
        )
        provider = build_provider(cfg)
        assert isinstance(provider, _Fake)
        assert provider.id == "ext"
    finally:
        unregister_provider("ext-sdk", "ext-api")


def test_built_in_classes_are_importable_at_package_top_level() -> None:
    """The four provider classes are re-exported from `needlehaystack.providers`
    so users / tests can construct them directly without poking submodules.
    """
    for cls in (FakeProvider, OpenAIProvider, AnthropicProvider, CohereProvider):
        assert isinstance(cls, type)
