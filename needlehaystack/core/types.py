"""Plain dataclasses passed between the runner, tasks, and stores.

These are dumb containers. Anything with behavior (encoding, scoring,
inserting) lives in its own module.

The shape mirrors the v2 result schema documented in
`docs/refactor/overview.md`. Keep them aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(slots=True)
class Needle:
    """A piece of information inserted into a haystack.

    `texts` is a list because some tasks (multi-needle, UUID chain) insert
    more than one snippet for a single logical "needle". For a plain single
    needle the list has one entry.

    `expected_answer` is the literal string the scorer compares the model's
    response against. For UUID chains it's the final UUID in the chain.

    `metadata` is task-specific bookkeeping (e.g. the full chain for a
    UUID chain task). It rides along on the result row so the JSONL is
    self-describing.
    """

    texts: list[str]
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NeedlePlacement:
    """Where one needle text actually ended up after insertion.

    `insertion_token_index` is measured in the **post-insertion** token
    stream — i.e. the position at which this needle starts in the final
    context. That's what reconstruction needs.

    `actual_depth_percent` is computed from the pre-insertion token length
    (see `inserters/`). This is the bug-free version of the v1 multi-needle
    code.
    """

    text: str
    insertion_token_index: int
    actual_depth_percent: float


@dataclass(slots=True)
class Recipe:
    """Everything a reader needs to reconstruct the rendered context.

    The runner builds this from the haystack descriptor + the inserter's
    returned placements. It's deliberately small (kilobytes regardless of
    context size) so result rows don't blow up the JSONL file.
    """

    haystack: dict[str, Any]
    inserter: str
    needle_placements: list[NeedlePlacement]
    final_context_token_count: int


@dataclass(slots=True)
class Usage:
    """Token usage returned by the provider call."""

    input_tokens: int
    output_tokens: int


@dataclass(slots=True)
class TestCase:
    """One cell of the sweep grid."""

    # Tell pytest not to try to collect this dataclass as a test class.
    __test__: ClassVar[bool] = False

    context_length: int
    depth_percent: float
    seed: int | None = None


@dataclass(slots=True)
class Score:
    """The outcome of running a scorer against a model response.

    `value` is in [0.0, 1.0]. Scorers may put extra detail (e.g.
    `hops_correct`) in `details` for downstream analysis.
    """

    value: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TestResult:
    """One row in the run's JSONL file.

    Schema version 2. Never contains the rendered context — use the
    `recipe` to regenerate it if needed.
    """

    # Tell pytest not to try to collect this dataclass as a test class.
    __test__: ClassVar[bool] = False

    schema_version: int
    run_name: str
    model_id: str
    model_request_name: str
    task_type: str
    context_length: int
    target_depth_percent: float
    recipe: Recipe
    needle_metadata: dict[str, Any]
    expected_answer: str
    prompt_question: str
    response: str | None
    score: Score | None
    usage: Usage | None
    cost_usd: float | None
    duration_seconds: float
    status: str  # "ok" | "error"
    error: str | None
    timestamp_utc: str
    seed: int | None
