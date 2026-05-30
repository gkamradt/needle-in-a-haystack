"""Protocol for haystack sources."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class HaystackSource(Protocol):
    """A source of background text large enough to satisfy a target length.

    Implementations return a string with **at least** `min_tokens` tokens
    under the project's shared encoding. The runner truncates to the
    exact length later.
    """

    name: ClassVar[str]

    def load(self, min_tokens: int) -> str:
        """Return background text with at least `min_tokens` tokens."""
        ...

    def descriptor(self) -> dict[str, str]:
        """Return a small dict identifying this source for the result recipe.

        Must include `type` matching `name`. Anything else (path, etc.)
        is implementation-defined and ends up on the result row.
        """
        ...
