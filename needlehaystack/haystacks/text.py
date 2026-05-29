"""Haystack made by repeating a passed-in string to length."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..core import tokens


@dataclass(slots=True)
class RepeatingTextHaystack:
    """Repeats `text` (with a separator) until token count is satisfied.

    Useful for tests and for users who want a controlled, content-free
    haystack (e.g. lorem ipsum) instead of essays.
    """

    name: ClassVar[str] = "text"

    text: str
    separator: str = "\n\n"

    def load(self, min_tokens: int) -> str:
        if not self.text:
            raise ValueError("RepeatingTextHaystack: `text` must be non-empty")
        unit = self.text + self.separator
        unit_tokens = tokens.count(unit)
        if unit_tokens == 0:  # pragma: no cover - defensive
            raise ValueError("RepeatingTextHaystack: `text` encodes to zero tokens")
        # `+ 1` so we always overshoot and the runner can truncate cleanly.
        repeats = (min_tokens // unit_tokens) + 1
        return unit * repeats

    def descriptor(self) -> dict[str, str]:
        return {"type": self.name, "text_preview": self.text[:60]}
