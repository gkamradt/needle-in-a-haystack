"""Provider implementations for the v2 runner.

The Protocol (`ModelProvider`) lives in `base.py`. Real provider plugins
(`openai`, `anthropic`, `cohere`) self-register on import by calling
`register_provider(sdk, api, factory)` at module load time. Their SDK
dependencies are imported lazily inside each factory so importing this
package never requires the real SDKs at module load time.
"""

# Import the submodules for their side effect of registering provider
# plugins. Order doesn't matter; each module owns a distinct (sdk, api)
# key. Aliased to `_*` to make it obvious these imports are for side
# effects, not for re-export.
from . import anthropic as _anthropic  # noqa: F401
from . import cohere as _cohere  # noqa: F401
from . import openai as _openai  # noqa: F401
from .anthropic import AnthropicProvider
from .base import Completion, ModelProvider
from .cohere import CohereProvider
from .fake import FakeProvider, make_echo_prompt
from .openai import OpenAIProvider
from .registry import (
    PROVIDER_REGISTRY,
    build_provider,
    register_provider,
    unregister_provider,
)

__all__ = [
    "PROVIDER_REGISTRY",
    "AnthropicProvider",
    "CohereProvider",
    "Completion",
    "FakeProvider",
    "ModelProvider",
    "OpenAIProvider",
    "build_provider",
    "make_echo_prompt",
    "register_provider",
    "unregister_provider",
]
