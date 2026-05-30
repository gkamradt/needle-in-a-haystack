"""Exact-match (substring) scorer.

Handles both single-needle and multi-needle cases:

- **Single fact:** checks whether `needle.expected_answer` appears in
  the response (case-insensitive, whitespace-trimmed). 1.0 or 0.0.
- **Multi-fact:** if the needle carries an `expected_fragments` list in
  `needle.metadata`, scores the fraction of those fragments found in
  the response. Used by multi-needle tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..core.types import Needle, Score


@dataclass(slots=True)
class ExactMatchScorer:
    """Case-insensitive substring scorer."""

    name: ClassVar[str] = "exact_match"

    def score(self, response: str, needle: Needle) -> Score:
        fragments = needle.metadata.get("expected_fragments")
        if not fragments:
            fragments = [needle.expected_answer]

        if not fragments:
            # Defensive: nothing to score against → treat as miss.
            return Score(value=0.0, details={"matches": 0, "total": 0})

        haystack = _normalize(response)
        matches = sum(1 for f in fragments if _normalize(f) in haystack)
        value = matches / len(fragments)
        return Score(
            value=value,
            details={"matches": matches, "total": len(fragments)},
        )


def _normalize(s: str) -> str:
    return s.strip().lower()
