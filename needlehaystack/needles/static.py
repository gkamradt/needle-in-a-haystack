"""A needle generator that always returns the same configured needle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..core.types import Needle


@dataclass(slots=True)
class StaticNeedle:
    """Wraps a fixed needle. Ignores `seed`.

    `texts` is a list because multi-needle tasks insert multiple snippets
    for one logical needle. For a single needle, pass a one-element list.
    """

    name: ClassVar[str] = "static"

    texts: list[str]
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def generate(self, seed: int | None = None) -> Needle:
        # Copy the metadata so callers can mutate the returned Needle
        # without affecting future generations.
        return Needle(
            texts=list(self.texts),
            expected_answer=self.expected_answer,
            metadata=dict(self.metadata),
        )
