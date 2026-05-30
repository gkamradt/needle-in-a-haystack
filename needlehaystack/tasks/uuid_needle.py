"""UUID single-needle task.

Generates a fresh (or seeded) UUID, embeds it in a sentence, inserts at
the requested depth, and asks the model to repeat the UUID. Scoring is
exact-match against the UUID string (case-insensitive).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..core.types import Needle, NeedlePlacement, Score
from ..inserters.single_depth import SingleDepthInserter
from ..needles.uuid_single import UuidSingle
from ..scorers.exact_match import ExactMatchScorer


@dataclass(slots=True)
class UuidNeedleTask:
    """Insert a fresh UUID at `depth_percent` and ask the model to repeat it."""

    name: ClassVar[str] = "uuid"
    inserter_name: str = "single_depth"

    needle_template: str = "The secret code is {uuid}."
    question_template: str = "What is the secret code?"
    snap_to_periods: bool = True

    def generate_needle(self, seed: int | None) -> Needle:
        return UuidSingle(template=self.needle_template).generate(seed)

    def insert(
        self,
        context_tokens: list[int],
        needle: Needle,
        depth_percent: float,
    ) -> tuple[list[int], list[NeedlePlacement]]:
        result = SingleDepthInserter(snap_to_periods=self.snap_to_periods).insert(
            context_tokens, needle.texts, depth_percent
        )
        return result.new_tokens, result.placements

    def question(self, needle: Needle) -> str:
        del needle
        return self.question_template

    def score(self, response: str, needle: Needle) -> Score:
        return ExactMatchScorer().score(response, needle)
