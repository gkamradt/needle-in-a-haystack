"""Pydantic config schema for v2.

Two YAMLs drive a run:

1. **Run config** — what to do (`task`, `haystack`, `sweep`, `runner`,
   `store`) and which model to do it with. References a model by id or
   by file path.
2. **Model config** — *how* to call one model (SDK, request kwargs,
   pricing). Reusable across runs and shared via `configs/models/`.

Both are deliberately permissive on the provider-specific
`request:` / `agent:` blocks: pydantic captures the common keys and
passes the rest through verbatim, so adding `thinking` / `output_config`
/ `reasoning_effort` for a new model never requires a schema change.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Model config (standalone, one YAML per model in `configs/models/`)
# ---------------------------------------------------------------------------


class ModelRuntime(BaseModel):
    """How to talk to this model at the transport layer.

    `(sdk, api)` together name the provider plugin in the registry.
    """

    sdk: str
    api: str
    state: str | None = "manual_rolling"


class ModelClient(BaseModel):
    """SDK client construction inputs (mostly auth)."""

    api_key_env: str | None = None
    base_url_env: str | None = None

    model_config = ConfigDict(extra="allow")


class ModelRequest(BaseModel):
    """Per-call request payload.

    Common keys are typed; anything else (thinking, output_config,
    reasoning_effort, top_p, …) rides through under `extra="allow"`.
    """

    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = None

    model_config = ConfigDict(extra="allow")


class ModelAgent(BaseModel):
    """Optional agent-loop knobs. Free-form."""

    model_config = ConfigDict(extra="allow")


class ModelPricing(BaseModel):
    """USD per million tokens. Optional; without it `cost_usd` stays null."""

    input: float
    output: float


class ModelConfig(BaseModel):
    """A reusable model spec."""

    id: str
    agent: ModelAgent = Field(default_factory=ModelAgent)
    runtime: ModelRuntime
    client: ModelClient = Field(default_factory=ModelClient)
    request: ModelRequest
    pricing: ModelPricing | None = None


# ---------------------------------------------------------------------------
# Run config — task / haystack / sweep / runner / store
# ---------------------------------------------------------------------------


class TaskSpec(BaseModel):
    """`task.type` picks a registered task; the rest is task-class kwargs."""

    type: str
    model_config = ConfigDict(extra="allow")

    def kwargs(self) -> dict[str, Any]:
        """Return everything except `type`, for the task constructor."""
        return self.model_dump(exclude={"type"}, exclude_none=True)


class HaystackSpec(BaseModel):
    """`haystack.type` picks `files` or `text`; rest passes through."""

    type: Literal["files", "text"]
    model_config = ConfigDict(extra="allow")

    def kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"type"}, exclude_none=True)


class SweepRange(BaseModel):
    """Range-style sweep spec: `{min, max, num, scale?}`."""

    min: float
    max: float
    num: int
    scale: Literal["linear", "sigmoid"] = "linear"


# A sweep dimension is either an explicit list of values or a SweepRange.
SweepDim = list[float] | list[int] | SweepRange


class SweepSpec(BaseModel):
    """Lengths × depths × optional seeds."""

    context_lengths: SweepDim
    depth_percents: SweepDim
    seeds: list[int | None] = Field(default_factory=lambda: [None])  # type: ignore[arg-type]


class RunnerSpec(BaseModel):
    """Runner orchestration knobs (mirrors `Runner` dataclass)."""

    concurrency: int = 1
    retries: int = 0
    sleep_between_seconds: float = 0.0
    resume: bool = True


class StoreSpec(BaseModel):
    """`store.type` picks `jsonl` (only backend in v2.0)."""

    type: Literal["jsonl"] = "jsonl"
    path: str


class RunConfig(BaseModel):
    """Top-level run YAML."""

    run_name: str
    # Either a path to a model YAML or a bare id resolved against the
    # `model_search_dirs` argument to `load_run`.
    model: str
    task: TaskSpec
    haystack: HaystackSpec
    sweep: SweepSpec
    runner: RunnerSpec = Field(default_factory=RunnerSpec)
    store: StoreSpec
