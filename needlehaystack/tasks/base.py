"""Protocol for tasks plus a tiny registry.

A **task** is the user-extension point: it composes a needle generator,
an inserter, a scorer, and a prompt question into one named bundle that
the runner can drive. Third parties add a new test type by writing a
class that satisfies the `Task` protocol and calling `register_task`.

The protocol is deliberately narrow:

- The task owns the needle (`generate_needle`) and the prompt
  (`question`).
- The task owns insertion via `insert(...)`, which returns the
  post-insertion token stream **and** the per-needle `NeedlePlacement`s.
  The runner uses those placements to build a `Recipe` for the result
  row — tasks don't need to know reconstruction exists.
- The task owns scoring (`score`).

`inserter_name` is exposed so the runner / store can record which
inserter strategy a row used without poking at the task's internals.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from ..core.types import Needle, NeedlePlacement, Score


@runtime_checkable
class Task(Protocol):
    """Compose generator + inserter + scorer + prompt into one test type."""

    name: ClassVar[str]
    """Stable identifier used in configs and result rows (e.g. `"single"`)."""

    inserter_name: str
    """Name of the inserter strategy the task uses. Recorded in `Recipe`."""

    def generate_needle(self, seed: int | None) -> Needle:
        """Produce the needle for one sweep cell.

        Deterministic when `seed` is supplied (where applicable).
        """
        ...

    def insert(
        self,
        context_tokens: list[int],
        needle: Needle,
        depth_percent: float,
    ) -> tuple[list[int], list[NeedlePlacement]]:
        """Splice `needle` into `context_tokens` at `depth_percent`.

        Returns `(new_tokens, placements)`. `placements` has one entry
        per needle text, in needle order.
        """
        ...

    def question(self, needle: Needle) -> str:
        """The user-facing prompt question for this needle."""
        ...

    def score(self, response: str, needle: Needle) -> Score:
        """Score the model's response against `needle`."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, type[Task]] = {}


def register_task(name: str, cls: type[Task]) -> None:
    """Register a task class under `name`.

    Raises `ValueError` if `name` is already registered. We deliberately
    don't allow silent overwrites — duplicate names usually mean a copy-
    paste bug in a contributor's plugin.
    """
    if name in TASK_REGISTRY:
        raise ValueError(
            f"task name {name!r} is already registered to {TASK_REGISTRY[name].__name__}"
        )
    TASK_REGISTRY[name] = cls


def get_task(name: str) -> type[Task]:
    """Look up a task class by name. Raises `KeyError` if unknown."""
    if name not in TASK_REGISTRY:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]


def unregister_task(name: str) -> None:
    """Remove a task from the registry. Used in tests to keep state clean."""
    TASK_REGISTRY.pop(name, None)
