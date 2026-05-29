"""Provider implementations for the v2 runner.

The Protocol (`ModelProvider`) lives in `base.py`. Concrete provider
classes for OpenAI, Anthropic, and Cohere are introduced in Phase 7.
"""

from .base import Completion, ModelProvider
from .fake import FakeProvider, make_echo_prompt

__all__ = [
    "Completion",
    "FakeProvider",
    "ModelProvider",
    "make_echo_prompt",
]
