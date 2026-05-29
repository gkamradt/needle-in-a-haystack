"""Protocol for inserters plus shared helpers (period snapping).

All inserter implementations work in **token space** (using the shared
`cl100k_base` encoding) and return `NeedlePlacement` objects that record
exactly where each needle landed in the final, post-insertion token
stream. The runner uses those placements to build the result row's
reconstruction recipe.
"""

from __future__ import annotations

from typing import ClassVar, NamedTuple, Protocol, runtime_checkable

from ..core import tokens
from ..core.types import NeedlePlacement

# Maximum number of tokens we'll walk backward looking for a sentence
# boundary before giving up and inserting at the raw point. Bounded so
# the lookup is O(MAX_PERIOD_SNAP) per needle.
MAX_PERIOD_SNAP: int = 50


class InsertResult(NamedTuple):
    """Output of `Inserter.insert`.

    `new_tokens` is the final token list (haystack with all needles
    spliced in). `placements` records where each needle landed; one
    entry per needle, in needle order.
    """

    new_tokens: list[int]
    placements: list[NeedlePlacement]


@runtime_checkable
class Inserter(Protocol):
    """Splice one-or-more needles into a token stream."""

    name: ClassVar[str]

    def insert(
        self,
        context_tokens: list[int],
        needle_texts: list[str],
        depth_percent: float,
    ) -> InsertResult:
        """Splice the needles into `context_tokens` and return the result.

        `depth_percent` semantics depend on the inserter (e.g. single
        depth vs. starting depth for even spread).
        """
        ...


def snap_to_period(
    context_tokens: list[int],
    insertion_point: int,
    *,
    max_snap: int = MAX_PERIOD_SNAP,
) -> int:
    """Walk `insertion_point` backward to the nearest preceding period.

    Bounded by `max_snap`. If no period is found within the window the
    original point is returned — the v1 code looped unbounded and could
    walk all the way to index 0.

    `insertion_point` is interpreted as "insert *before* the token at
    this index"; we check that the token immediately *preceding* the
    insertion point is a period.
    """
    if insertion_point <= 0:
        return 0
    period_tokens = set(tokens.encode("."))
    lower_bound = max(0, insertion_point - max_snap)
    while insertion_point > lower_bound:
        if context_tokens[insertion_point - 1] in period_tokens:
            return insertion_point
        insertion_point -= 1
    return insertion_point
