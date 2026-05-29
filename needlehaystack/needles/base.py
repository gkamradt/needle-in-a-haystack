"""Protocol for needle generators."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from ..core.types import Needle


@runtime_checkable
class NeedleGenerator(Protocol):
    """Produces a `Needle` for one cell of the sweep.

    `seed` is passed by the runner so callers can reproduce a row from
    its recorded seed. Implementations that don't need randomness can
    ignore the argument.
    """

    name: ClassVar[str]

    def generate(self, seed: int | None = None) -> Needle: ...
