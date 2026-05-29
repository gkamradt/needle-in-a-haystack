"""UUID-chain task.

Generates a chain `A → B → C → …` of fresh UUIDs, spreads the link
statements through the context, and asks the model what value is
associated with the starting UUID. **The question deliberately does
not reveal the chain structure** — the model must discover the hops on
its own. That discovery is the interesting behavior we're measuring.

Scoring uses `ChainMatchScorer`, which credits the model for the
furthest hop it reached (so a partial traversal still scores >0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..core.types import Needle, NeedlePlacement, Score
from ..inserters.even_spread import EvenSpreadInserter
from ..needles.uuid_chain import UuidChain
from ..scorers.chain_match import ChainMatchScorer


@dataclass(slots=True)
class UuidChainTask:
    """Insert a UUID chain spread through the context."""

    name: ClassVar[str] = "uuid_chain"
    inserter_name: str = "even_spread"

    chain_length: int = 5
    link_template: str = "{prev} maps to {next}."
    # The default question never says "chain", "hop", "follow", or "link" —
    # we want the model to figure out it has to traverse on its own.
    question_template: str = "What is the value associated with {start}?"
    snap_to_periods: bool = True

    def generate_needle(self, seed: int | None) -> Needle:
        return UuidChain(
            chain_length=self.chain_length,
            link_template=self.link_template,
        ).generate(seed)

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
        start = needle.metadata.get("start")
        if not start:
            raise ValueError(
                "UuidChainTask.question needs needle.metadata['start']; "
                f"got keys: {sorted(needle.metadata.keys())}"
            )
        return self.question_template.format(start=start)

    def score(self, response: str, needle: Needle) -> Score:
        return ChainMatchScorer().score(response, needle)
