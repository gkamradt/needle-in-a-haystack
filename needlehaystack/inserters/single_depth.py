"""Insert a single needle at one specified depth percentage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..core import tokens
from ..core.types import NeedlePlacement
from .base import InsertResult, snap_to_period


@dataclass(slots=True)
class SingleDepthInserter:
    """Insert exactly one needle text at `depth_percent` of the context.

    Behavior at the edges:

    - `depth_percent == 0`: inserted at the start.
    - `depth_percent == 100`: appended at the end.
    - Otherwise: insertion point is `len * depth/100`, then snapped
      back to the nearest preceding period within a bounded window.

    Multiple needle texts are not supported; pass a one-element list.
    Multi-needle insertion is the job of `EvenSpreadInserter`.
    """

    name: ClassVar[str] = "single_depth"

    snap_to_periods: bool = True

    def insert(
        self,
        context_tokens: list[int],
        needle_texts: list[str],
        depth_percent: float,
    ) -> InsertResult:
        if len(needle_texts) != 1:
            raise ValueError(
                f"SingleDepthInserter takes exactly 1 needle text; got {len(needle_texts)}"
            )
        if not 0.0 <= depth_percent <= 100.0:
            raise ValueError(f"depth_percent must be in [0, 100]; got {depth_percent}")

        needle_text = needle_texts[0]
        needle_tokens = tokens.encode(needle_text)
        pre_len = len(context_tokens)

        if depth_percent == 100.0:
            insertion_point = pre_len
        elif depth_percent == 0.0:
            insertion_point = 0
        else:
            insertion_point = int(pre_len * (depth_percent / 100.0))
            if self.snap_to_periods:
                insertion_point = snap_to_period(context_tokens, insertion_point)

        # Compute `actual_depth_percent` against the pre-insertion length
        # so it stays comparable across needles / runs.
        actual_depth = (insertion_point / pre_len * 100.0) if pre_len > 0 else 0.0

        new_tokens = (
            context_tokens[:insertion_point] + needle_tokens + context_tokens[insertion_point:]
        )

        placement = NeedlePlacement(
            text=needle_text,
            insertion_token_index=insertion_point,
            actual_depth_percent=actual_depth,
        )
        return InsertResult(new_tokens=new_tokens, placements=[placement])
