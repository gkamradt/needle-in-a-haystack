"""Insert N needles at caller-supplied depth percentages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ..core import tokens
from ..core.types import NeedlePlacement
from .base import InsertResult, snap_to_period


@dataclass(slots=True)
class ExplicitDepthsInserter:
    """Place one needle at each of `depth_percents`.

    `len(depth_percents)` must equal the number of needle texts passed
    to `insert`. The runner-level `depth_percent` argument is ignored —
    it's the inserter's job to respect what the caller asked for.

    Depths may be in any order; the inserter sorts them ascending
    internally before splicing so token-index bookkeeping stays simple.
    """

    name: ClassVar[str] = "explicit"

    depth_percents: list[float] = field(default_factory=list)
    snap_to_periods: bool = True

    def __post_init__(self) -> None:
        for d in self.depth_percents:
            if not 0.0 <= d <= 100.0:
                raise ValueError(f"every entry in depth_percents must be in [0, 100]; got {d}")

    def insert(
        self,
        context_tokens: list[int],
        needle_texts: list[str],
        depth_percent: float,
    ) -> InsertResult:
        del depth_percent  # explicit inserter ignores the sweep-level depth
        if len(needle_texts) != len(self.depth_percents):
            raise ValueError(
                f"ExplicitDepthsInserter got {len(needle_texts)} needle texts but "
                f"{len(self.depth_percents)} depth_percents; they must match"
            )

        pre_len = len(context_tokens)

        # Pair each needle with its requested depth, sort by depth
        # ascending so we can splice left-to-right.
        items = sorted(
            zip(needle_texts, self.depth_percents, strict=True),
            key=lambda pair: pair[1],
        )

        new_tokens = list(context_tokens)
        # Track placements keyed by input order so we return them in the
        # same order the caller passed needle_texts.
        order = {id(text): i for i, text in enumerate(needle_texts)}
        placements_by_input_index: list[NeedlePlacement | None] = [None] * len(needle_texts)
        accumulated_shift = 0

        for text, depth in items:
            if depth >= 100.0:
                raw_index = pre_len
            else:
                raw_index = int(pre_len * (depth / 100.0))
                if self.snap_to_periods:
                    raw_index = snap_to_period(context_tokens, raw_index)
            actual_depth = (raw_index / pre_len * 100.0) if pre_len > 0 else 0.0
            final_index = raw_index + accumulated_shift
            needle_tokens = tokens.encode(text)
            new_tokens = new_tokens[:final_index] + needle_tokens + new_tokens[final_index:]
            placements_by_input_index[order[id(text)]] = NeedlePlacement(
                text=text,
                insertion_token_index=final_index,
                actual_depth_percent=actual_depth,
            )
            accumulated_shift += len(needle_tokens)

        # By construction every slot is filled.
        placements = [p for p in placements_by_input_index if p is not None]
        assert len(placements) == len(needle_texts)
        return InsertResult(new_tokens=new_tokens, placements=placements)
