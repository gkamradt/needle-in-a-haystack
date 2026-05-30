"""Generate a fresh UUID needle."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import ClassVar

from ..core.types import Needle


@dataclass(slots=True)
class UuidSingle:
    """Generates one fresh UUID and wraps it in a sentence.

    The default template is `"The secret code is {uuid}."`. Change the
    template if you want different surrounding text, but keep `{uuid}`
    as the placeholder.
    """

    name: ClassVar[str] = "uuid_single"

    template: str = "The secret code is {uuid}."

    def generate(self, seed: int | None = None) -> Needle:
        u = _uuid_for_seed(seed)
        text = self.template.format(uuid=u)
        return Needle(
            texts=[text],
            expected_answer=u,
            metadata={"uuid": u},
        )


def _uuid_for_seed(seed: int | None) -> str:
    """Return a UUID-v4-shaped string.

    If `seed` is given, the result is deterministic (derived from a
    seeded `random.Random`). Otherwise a fresh `uuid.uuid4()` is used.
    Deterministic mode is important for resumable / reproducible runs.
    """
    if seed is None:
        return str(uuid.uuid4())
    rng = random.Random(seed)
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))
