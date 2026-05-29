"""Protocol for scorers."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from ..core.types import Needle, Score


@runtime_checkable
class Scorer(Protocol):
    """Compares a model response to a needle and returns a `Score`."""

    name: ClassVar[str]

    def score(self, response: str, needle: Needle) -> Score: ...
