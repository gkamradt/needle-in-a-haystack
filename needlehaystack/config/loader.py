"""YAML loader + factory functions that turn config into live objects.

Two public entry points:

- `load_run(path)` — parse a run YAML into a `RunConfig`.
- `load_model(path_or_id, search_dirs)` — parse a model YAML into a
  `ModelConfig`, resolving bare ids against a list of search dirs.

The `build_*` helpers turn the parsed specs into the concrete objects
the runner needs (task, haystack, provider, store, sweep).

Errors are raised as `ConfigError` with messages that name what's wrong
and what was expected. We don't surface raw pydantic stack traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..core.sweep import Sweep, from_spec
from ..haystacks.base import HaystackSource
from ..haystacks.files import FilesHaystack
from ..haystacks.text import RepeatingTextHaystack
from ..providers.base import ModelProvider
from ..providers.registry import build_provider as _build_provider_from_config
from ..stores.base import ResultStore
from ..stores.jsonl import JsonlResultStore
from ..tasks.base import TASK_REGISTRY, Task
from .schema import (
    HaystackSpec,
    ModelConfig,
    RunConfig,
    StoreSpec,
    SweepSpec,
    TaskSpec,
)


class ConfigError(Exception):
    """User-facing config problem. Always carries an actionable message."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_run(path: str | Path) -> RunConfig:
    """Read + validate a run YAML."""
    raw = _read_yaml(path)
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"invalid run config at {path}:\n{_format_validation(e)}") from e


def load_model(path_or_id: str | Path, search_dirs: list[Path] | None = None) -> ModelConfig:
    """Resolve `path_or_id` to a model YAML and parse it.

    Resolution order:

    1. If `path_or_id` is an existing file, parse it.
    2. Else, for each `search_dirs` entry, try `<dir>/<id>.yaml`.

    Raises `ConfigError` if nothing resolves.
    """
    resolved = _resolve_model_path(path_or_id, search_dirs or [])
    raw = _read_yaml(resolved)
    try:
        return ModelConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"invalid model config at {resolved}:\n{_format_validation(e)}") from e


def _resolve_model_path(path_or_id: str | Path, search_dirs: list[Path]) -> Path:
    p = Path(path_or_id)
    if p.is_file():
        return p
    for d in search_dirs:
        candidate = d / f"{path_or_id}.yaml"
        if candidate.is_file():
            return candidate
    raise ConfigError(
        f"could not resolve model {str(path_or_id)!r}; "
        f"tried as a path and as <dir>/<id>.yaml in: {[str(d) for d in search_dirs]}"
    )


def _read_yaml(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_validation(e: ValidationError) -> str:
    lines: list[str] = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_task(spec: TaskSpec) -> Task:
    """Look up `spec.type` in the registry and instantiate with kwargs."""
    if spec.type not in TASK_REGISTRY:
        raise ConfigError(f"unknown task type {spec.type!r}; registered: {sorted(TASK_REGISTRY)}")
    cls = TASK_REGISTRY[spec.type]
    try:
        return cls(**spec.kwargs())
    except TypeError as e:
        raise ConfigError(
            f"task {spec.type!r} rejected its kwargs: {e}. Got: {sorted(spec.kwargs())}"
        ) from e


def build_haystack(spec: HaystackSpec) -> HaystackSource:
    """Build a `HaystackSource` from `{type, ...kwargs}`."""
    if spec.type == "files":
        return FilesHaystack(**spec.kwargs())
    if spec.type == "text":
        return RepeatingTextHaystack(**spec.kwargs())
    # pydantic's Literal already rules this out, but defend explicitly.
    raise ConfigError(f"unknown haystack type {spec.type!r}")


def build_store(spec: StoreSpec) -> ResultStore:
    """Build a `ResultStore`. Only `jsonl` is supported in v2.0."""
    if spec.type == "jsonl":
        return JsonlResultStore(spec.path)
    raise ConfigError(f"unknown store type {spec.type!r}")


def build_sweep(spec: SweepSpec) -> Sweep:
    """Resolve sweep dims to concrete lists and return a `Sweep`."""
    lengths = from_spec(_sweep_dim_to_any(spec.context_lengths), kind="length")
    depths = from_spec(_sweep_dim_to_any(spec.depth_percents), kind="depth")
    return Sweep(
        lengths=[int(v) for v in lengths],
        depths=[float(v) for v in depths],
        seeds=list(spec.seeds),
    )


def _sweep_dim_to_any(dim: Any) -> Any:
    """Translate the schema-style SweepRange back to the dict shape from_spec wants."""
    from .schema import SweepRange

    if isinstance(dim, SweepRange):
        return {"min": dim.min, "max": dim.max, "num": dim.num, "scale": dim.scale}
    return dim


def build_provider(model: ModelConfig) -> ModelProvider:
    """Look up the provider plugin for `(runtime.sdk, runtime.api)`."""
    return _build_provider_from_config(model)
