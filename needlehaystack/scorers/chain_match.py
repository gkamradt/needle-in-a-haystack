"""Scorer for UUID-chain tasks.

The needle's `metadata["chain"]` is the full ordered list of UUIDs the
model would have to hop through to reach the final answer. We walk the
chain looking at how far through it the response progressed.

The reported `value` is `(furthest_index_found + 1) / len(chain)`, so:

- A response containing the final UUID scores 1.0.
- A response that only mentions the first 2 UUIDs of a 5-link chain
  scores 0.4.
- A response that mentions only UUIDs not in the chain scores 0.0.

`furthest_index_found` is the largest index `i` such that `chain[i]`
appears as a substring of the response. We don't require the earlier
UUIDs to appear — finding `chain[i]` alone is sufficient evidence that
the model got that far.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..core.types import Needle, Score


@dataclass(slots=True)
class ChainMatchScorer:
    """Score how far through a UUID chain the model progressed."""

    name: ClassVar[str] = "chain_match"

    def score(self, response: str, needle: Needle) -> Score:
        chain = needle.metadata.get("chain")
        if not chain:
            raise ValueError(
                "ChainMatchScorer needs needle.metadata['chain']; "
                f"got metadata keys: {sorted(needle.metadata.keys())}"
            )

        normalized = response.lower()
        furthest = -1
        for i, node in enumerate(chain):
            if node.lower() in normalized:
                furthest = i

        hops_correct = furthest + 1  # nodes reached, counting from the start
        value = hops_correct / len(chain)
        return Score(
            value=value,
            details={
                "hops_correct": hops_correct,
                "chain_length": len(chain),
            },
        )
