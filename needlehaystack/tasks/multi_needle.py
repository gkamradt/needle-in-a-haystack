"""Multi-fact needle task: N sentences spread through the context.

The model must reproduce (or otherwise mention) every fragment to score
1.0. `ExactMatchScorer` returns a fractional score when only some
fragments appear, via `needle.metadata["expected_fragments"]`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ..core.types import Needle, NeedlePlacement, Score
from ..inserters.even_spread import EvenSpreadInserter
from ..scorers.exact_match import ExactMatchScorer


def _default_needle_texts() -> list[str]:
    """Default trio used in the original repo's multi-needle demo."""
    return [
        "Figs are one of the secret ingredients needed to build the perfect pizza.",
        "Prosciutto is one of the secret ingredients needed to build the perfect pizza.",
        "Goat cheese is one of the secret ingredients needed to build the perfect pizza.",
    ]


def _default_fragments() -> list[str]:
    return ["figs", "prosciutto", "goat cheese"]


@dataclass(slots=True)
class MultiNeedleTask:
    """Insert N needle texts evenly from `depth_percent` to 100%."""

    name: ClassVar[str] = "multi"
    inserter_name: str = "even_spread"

    needle_texts: list[str] = field(default_factory=_default_needle_texts)
    expected_fragments: list[str] = field(default_factory=_default_fragments)
    question_template: str = "What are the secret ingredients needed to build the perfect pizza?"
    snap_to_periods: bool = True

    def __post_init__(self) -> None:
        if not self.needle_texts:
            raise ValueError("MultiNeedleTask needs at least one needle text")
        if not self.expected_fragments:
            raise ValueError("MultiNeedleTask needs at least one expected fragment")

    def generate_needle(self, seed: int | None) -> Needle:
        del seed  # static needle, no randomness
        return Needle(
            texts=list(self.needle_texts),
            expected_answer=" / ".join(self.expected_fragments),
            metadata={"expected_fragments": list(self.expected_fragments)},
        )

    def insert(
        self,
        context_tokens: list[int],
        needle: Needle,
        depth_percent: float,
    ) -> tuple[list[int], list[NeedlePlacement]]:
        result = EvenSpreadInserter(snap_to_periods=self.snap_to_periods).insert(
            context_tokens, needle.texts, depth_percent
        )
        return result.new_tokens, result.placements

    def question(self, needle: Needle) -> str:
        del needle
        return self.question_template

    def score(self, response: str, needle: Needle) -> Score:
        return ExactMatchScorer().score(response, needle)
