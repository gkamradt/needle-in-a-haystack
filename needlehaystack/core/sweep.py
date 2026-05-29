"""Sweep grid helpers: turn a spec into concrete lengths / depths.

A sweep "spec" comes from a config and may be either a dict-style
description (`{min, max, num, scale}`) or an explicit list of values.
This module accepts both shapes and returns concrete `list[int]` or
`list[float]` for the runner to iterate over.

Two scales are supported:

- `"linear"` — evenly spaced between `min` and `max` inclusive.
- `"sigmoid"` — bunched at both ends, sparser in the middle. Useful for
  depth sweeps where the interesting transitions tend to happen near
  the edges (0% and 100%). Endpoints are guaranteed to land on `min`
  and `max` exactly.

`build_lengths` returns ints (token counts), `build_depths` returns
floats (percentages).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

Scale = Literal["linear", "sigmoid"]


@dataclass(slots=True)
class Sweep:
    """Resolved sweep grid: every cell is `(length, depth, seed)`."""

    lengths: list[int]
    depths: list[float]
    # `[None]` means a single un-seeded run per cell. Multiple seeds
    # multiply the cell count — same length/depth, different RNG state.
    seeds: list[int | None] = field(default_factory=lambda: [None])

    def __post_init__(self) -> None:
        if not self.lengths:
            raise ValueError("Sweep needs at least one length")
        if not self.depths:
            raise ValueError("Sweep needs at least one depth")
        if not self.seeds:
            raise ValueError("Sweep needs at least one seed (use [None] for unseeded)")

    def cells(self) -> list[tuple[int, float, int | None]]:
        """Cartesian product of `(length, depth, seed)`."""
        return [(L, d, s) for L in self.lengths for d in self.depths for s in self.seeds]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_lengths(
    min: int,  # noqa: A002 — mirrors config field name
    max: int,  # noqa: A002
    num: int,
    scale: Scale = "linear",
) -> list[int]:
    """Build `num` context lengths between `min` and `max` inclusive."""
    return [int(round(v)) for v in _build_values(float(min), float(max), num, scale)]


def build_depths(
    min: float,  # noqa: A002
    max: float,  # noqa: A002
    num: int,
    scale: Scale = "linear",
) -> list[float]:
    """Build `num` depth percentages between `min` and `max` inclusive."""
    return _build_values(min, max, num, scale)


def from_spec(spec: Any, *, kind: Literal["length", "depth"]) -> list[int] | list[float]:
    """Accept either an explicit list or a `{min,max,num,scale}` dict.

    Lists are passed through (with type-appropriate casting). Dicts are
    delegated to the matching builder.
    """
    if isinstance(spec, list):
        if kind == "length":
            return [int(v) for v in spec]
        return [float(v) for v in spec]
    if isinstance(spec, dict):
        kwargs = {
            "min": spec["min"],
            "max": spec["max"],
            "num": spec["num"],
            "scale": spec.get("scale", "linear"),
        }
        if kind == "length":
            return build_lengths(**kwargs)
        return build_depths(**kwargs)
    raise TypeError(
        f"sweep spec for {kind!r} must be a list or {{min,max,num,scale}} dict; "
        f"got {type(spec).__name__}"
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_values(lo: float, hi: float, num: int, scale: Scale) -> list[float]:
    if num < 1:
        raise ValueError(f"num must be >= 1; got {num}")
    if num == 1:
        return [lo]
    if hi < lo:
        raise ValueError(f"max ({hi}) must be >= min ({lo})")

    ts = [i / (num - 1) for i in range(num)]  # 0.0 → 1.0 inclusive

    if scale == "linear":
        return [lo + t * (hi - lo) for t in ts]

    if scale == "sigmoid":
        # Renormalized logistic so f(0)=0 and f(1)=1 exactly.
        k = 6.0
        s_lo = _sigmoid(-k / 2)
        s_hi = _sigmoid(k / 2)
        span = s_hi - s_lo
        normalized = [(_sigmoid(k * (t - 0.5)) - s_lo) / span for t in ts]
        return [lo + n * (hi - lo) for n in normalized]

    raise ValueError(f"unknown scale {scale!r}; expected 'linear' or 'sigmoid'")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
