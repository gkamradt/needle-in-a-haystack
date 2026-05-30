"""Provider Protocol used by the v2 runner.

This is the new, narrow contract: a provider is constructed from a parsed
`ModelConfig` (Phase 6) and exposes a single `complete(system, user)`
coroutine that returns a `Completion`. Everything model-specific
(`temperature`, `thinking`, `output_config`, `reasoning_effort`, …) lives
on the model config; providers don't take per-call kwargs.

The legacy `model.py` ABC stays in the tree alongside this file until
Phase 7 deletes the v1 testers that import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..core.types import Usage


@dataclass(slots=True)
class Completion:
    """The result of a single provider call.

    `text` is the model's response. `usage` is `None` for providers /
    APIs that don't report token usage.
    """

    text: str
    usage: Usage | None


@runtime_checkable
class ModelProvider(Protocol):
    """The v2 provider contract.

    Implementations are constructed from a parsed `ModelConfig` and own
    the SDK client and the merged `request:` kwargs. The runner only
    sees `complete(system, user)`.
    """

    id: str
    """Human-friendly id from the model-config YAML (e.g.
    `"anthropic-opus-4-medium"`). Appears in result rows."""

    request_model: str
    """The literal model string sent to the SDK (e.g. `"claude-opus-4"`)."""

    async def complete(self, system: str, user: str) -> Completion:
        """Run one chat-style completion and return the response.

        Implementations should raise on transport / API errors. The runner
        is responsible for retries and error logging.
        """
        ...
