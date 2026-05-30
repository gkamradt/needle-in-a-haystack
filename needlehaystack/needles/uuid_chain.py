"""Generate a chain of UUIDs and the link statements that connect them.

For `chain_length = N` the generator produces:

- N fresh UUIDs `[A, B, C, ..., Z]`
- N-1 link statements: `"A maps to B."`, `"B maps to C."`, ...

The needle's `texts` is the list of link statements (one per inserted
snippet). The `expected_answer` is the final UUID — the model must
follow the chain to find it. The full chain is stored in `metadata`.

The default question (set by the task in Phase 4) deliberately does NOT
reveal the chain structure — the model has to discover the hops.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import ClassVar

from ..core.types import Needle


@dataclass(slots=True)
class UuidChain:
    """Generates a chain of UUIDs and the snippets that link them."""

    name: ClassVar[str] = "uuid_chain"

    chain_length: int = 5
    link_template: str = "{prev} maps to {next}."

    def __post_init__(self) -> None:
        if self.chain_length < 2:
            raise ValueError(f"UuidChain.chain_length must be >= 2; got {self.chain_length}")

    def generate(self, seed: int | None = None) -> Needle:
        chain = _generate_chain(self.chain_length, seed)
        link_texts = [
            self.link_template.format(prev=chain[i], next=chain[i + 1])
            for i in range(len(chain) - 1)
        ]
        return Needle(
            texts=link_texts,
            expected_answer=chain[-1],
            metadata={"chain": chain, "start": chain[0]},
        )


def _generate_chain(n: int, seed: int | None) -> list[str]:
    if seed is None:
        return [str(uuid.uuid4()) for _ in range(n)]
    rng = random.Random(seed)
    return [str(uuid.UUID(int=rng.getrandbits(128), version=4)) for _ in range(n)]
