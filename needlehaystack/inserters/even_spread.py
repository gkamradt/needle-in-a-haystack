"""Insert N needles spread evenly through the remaining context.

Given N needles and a starting `depth_percent`, places the first needle
at `depth_percent`, then distributes the rest evenly through the
remaining context (depth_percent through 100%).

This is the bug-free replacement for the v1 multi-needle inserter, which
had two issues:

1. It recomputed `actual_depth_percent` against the *post-insertion*
   token list, so each needle's reported depth was off by however much
   the earlier needles had inflated the count.
2. The `depth == 100` branch silently failed to record placements or
   advance the depth for subsequent needles.

This implementation computes target indices against the *pre-insertion*
length up front, then walks through them in order, adjusting only the
insertion index (not the recorded depth) as each splice shifts later
positions to the right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..core import tokens
from ..core.types import NeedlePlacement
from .base import InsertResult, snap_to_period


@dataclass(slots=True)
class EvenSpreadInserter:
    """Spread N needles evenly from `depth_percent` to 100%."""

    name: ClassVar[str] = "even_spread"

    snap_to_periods: bool = True

    def insert(
        self,
        context_tokens: list[int],
        needle_texts: list[str],
        depth_percent: float,
    ) -> InsertResult:
        if not needle_texts:
            raise ValueError("EvenSpreadInserter needs at least one needle text")
        if not 0.0 <= depth_percent <= 100.0:
            raise ValueError(f"depth_percent must be in [0, 100]; got {depth_percent}")

        pre_len = len(context_tokens)
        n = len(needle_texts)

        # Target depth for each needle: linear ramp from `depth_percent`
        # to 100%. We deliberately stop short of 100 itself (interval is
        # over n slots, not n-1) so the last needle doesn't always sit
        # at the very end.
        interval = (100.0 - depth_percent) / n
        target_depths = [depth_percent + i * interval for i in range(n)]

        # Compute pre-insertion indices and snap each to a sentence
        # boundary against the *original* context. This guarantees the
        # depths we report match the original token positions.
        planned: list[tuple[str, int, float]] = []
        for text, depth in zip(needle_texts, target_depths, strict=True):
            if depth >= 100.0:
                raw_index = pre_len
            else:
                raw_index = int(pre_len * (depth / 100.0))
                if self.snap_to_periods:
                    raw_index = snap_to_period(context_tokens, raw_index)
            actual_depth = (raw_index / pre_len * 100.0) if pre_len > 0 else 0.0
            planned.append((text, raw_index, actual_depth))

        # Insertion order matters: splice in ascending pre-insertion
        # index order so later indices stay valid after each splice.
        # (`planned` is already in that order because target_depths is
        # monotonically non-decreasing.)
        new_tokens = list(context_tokens)
        placements: list[NeedlePlacement] = []
        accumulated_shift = 0
        for text, raw_index, actual_depth in planned:
            needle_tokens = tokens.encode(text)
            final_index = raw_index + accumulated_shift
            new_tokens = new_tokens[:final_index] + needle_tokens + new_tokens[final_index:]
            placements.append(
                NeedlePlacement(
                    text=text,
                    insertion_token_index=final_index,
                    actual_depth_percent=actual_depth,
                )
            )
            accumulated_shift += len(needle_tokens)

        return InsertResult(new_tokens=new_tokens, placements=placements)
