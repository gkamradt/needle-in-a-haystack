"""Protocol for result stores.

A result store knows how to append one `TestResult` at a time and how
to answer "have we already written this cell?" for resume support. The
default implementation is JSONL (Phase 5); third parties could write a
sqlite, parquet, or remote-DB backend without touching the runner.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.types import TestResult

# 5-tuple uniquely identifying one cell of one run. Used for resume.
ResultKey = tuple[str, str, int, float, int | None]
# fields:           run_name, model_id, ctx_len, depth, seed


def key_for(result: TestResult) -> ResultKey:
    """Build the resume key from a TestResult."""
    return (
        result.run_name,
        result.model_id,
        result.context_length,
        result.target_depth_percent,
        result.seed,
    )


@runtime_checkable
class ResultStore(Protocol):
    """Append-only sink for `TestResult` rows."""

    async def append(self, result: TestResult) -> None:
        """Persist one result row. Implementations must be coroutine-safe."""
        ...

    def already_has(self, key: ResultKey) -> bool:
        """Return True if a row with `key` is already persisted."""
        ...
