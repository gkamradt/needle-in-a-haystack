"""Single-fact needle task: one static sentence at one depth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..core.types import Needle, NeedlePlacement, Score
from ..inserters.single_depth import SingleDepthInserter
from ..needles.static import StaticNeedle
from ..scorers.exact_match import ExactMatchScorer


@dataclass(slots=True)
class SingleNeedleTask:
    """Insert one configured sentence at `depth_percent` and ask one question.

    Default needle is the classic San Francisco / sandwich / Dolores Park
    fact used by the original repo, kept so the v1 demo still works under
    the v2 pipeline.
    """

    name: ClassVar[str] = "single"
    inserter_name: str = "single_depth"

    needle_text: str = (
        "The best thing to do in San Francisco is eat a sandwich "
        "and sit in Dolores Park on a sunny day."
    )
    expected_answer: str = "eat a sandwich and sit in Dolores Park"
    question_template: str = "What is the best thing to do in San Francisco?"
    snap_to_periods: bool = True

    def generate_needle(self, seed: int | None) -> Needle:
        return StaticNeedle(
            texts=[self.needle_text],
            expected_answer=self.expected_answer,
        ).generate(seed)

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
        del needle  # single-needle question is fixed
        return self.question_template

    def score(self, response: str, needle: Needle) -> Score:
        return ExactMatchScorer().score(response, needle)
