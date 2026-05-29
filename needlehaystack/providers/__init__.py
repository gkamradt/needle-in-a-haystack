"""Provider implementations for the v2 runner.

The Protocol (`ModelProvider`) lives in `base.py`. Concrete provider
classes for OpenAI, Anthropic, and Cohere are introduced in Phase 7.
"""

from .base import Completion, ModelProvider
from .fake import FakeProvider, make_echo_prompt
from .registry import (
    PROVIDER_REGISTRY,
    build_provider,
    register_provider,
    unregister_provider,
)

__all__ = [
    "PROVIDER_REGISTRY",
    "Completion",
    "FakeProvider",
    "ModelProvider",
    "build_provider",
    "make_echo_prompt",
    "register_provider",
    "unregister_provider",
]
