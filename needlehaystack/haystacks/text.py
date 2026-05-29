"""Haystack made by repeating a passed-in string to length."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar

from ..core import tokens

# Soft threshold: the source `text` is stored verbatim on every result
# row's recipe (for byte-exact reconstruction). The "unit" is meant to be
# small — if it's not, the user almost certainly wants `FilesHaystack`
# instead, which only stores a path.
LARGE_TEXT_THRESHOLD_BYTES = 10_000


@dataclass(slots=True)
class RepeatingTextHaystack:
    """Repeats `text` (with a separator) until token count is satisfied.

    Useful for tests and for users who want a controlled, content-free
    haystack (e.g. lorem ipsum) instead of essays.

    Note on storage: the source `text` is the **unit** that gets repeated
    to fill the target context length. It's saved verbatim on every
    result row so reconstruction can rebuild the exact context the model
    saw. Keep the unit small. If you have a large document, use
    `FilesHaystack` — its descriptor stores only the path.
    """

    name: ClassVar[str] = "text"

    text: str
    separator: str = "\n\n"

    def __post_init__(self) -> None:
        size = len(self.text.encode("utf-8"))
        if size > LARGE_TEXT_THRESHOLD_BYTES:
            warnings.warn(
                f"RepeatingTextHaystack: `text` is {size:,} bytes "
                f"(threshold {LARGE_TEXT_THRESHOLD_BYTES:,}). This text is "
                f"saved verbatim on every result row. For large content, "
                f"use FilesHaystack instead — its descriptor only stores "
                f"the path.",
                stacklevel=2,
            )

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
        # We store the source `text` (and separator) in full because the
        # reconstruction recipe needs to rebuild the exact context the
        # model saw. For `RepeatingTextHaystack` the source is the *unit*
        # that gets repeated, not the rendered context, so it's small.
        return {"type": self.name, "text": self.text, "separator": self.separator}
