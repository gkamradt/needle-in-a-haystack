"""Tests for the task registry — the user-extension entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from needlehaystack.core.types import Needle, NeedlePlacement, Score
from needlehaystack.tasks import (
    TASK_REGISTRY,
    MultiNeedleTask,
    SingleNeedleTask,
    UuidChainTask,
    UuidNeedleTask,
    get_task,
    register_task,
    unregister_task,
)


def test_builtins_are_registered() -> None:
    assert TASK_REGISTRY["single"] is SingleNeedleTask
    assert TASK_REGISTRY["multi"] is MultiNeedleTask
    assert TASK_REGISTRY["uuid"] is UuidNeedleTask
    assert TASK_REGISTRY["uuid_chain"] is UuidChainTask


def test_get_task_returns_class() -> None:
    assert get_task("single") is SingleNeedleTask


def test_get_unknown_task_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_task("not_a_real_task")


def test_register_then_resolve_custom_task() -> None:
    @dataclass(slots=True)
    class MyTask:
        name: ClassVar[str] = "my_test_task"
        inserter_name: str = "single_depth"

        def generate_needle(self, seed: int | None) -> Needle:
            return Needle(texts=["hi"], expected_answer="hi")

        def insert(
            self,
            context_tokens: list[int],
            needle: Needle,
            depth_percent: float,
        ) -> tuple[list[int], list[NeedlePlacement]]:
            return context_tokens, [
                NeedlePlacement(
                    text=needle.texts[0], insertion_token_index=0, actual_depth_percent=0.0
                )
            ]

        def question(self, needle: Needle) -> str:
            return "?"

        def score(self, response: str, needle: Needle) -> Score:
            return Score(value=1.0 if response == "hi" else 0.0)

    try:
        register_task("my_test_task", MyTask)
        assert get_task("my_test_task") is MyTask
        # An instance still satisfies the registry's contract.
        inst = MyTask()
        assert inst.name == "my_test_task"
    finally:
        unregister_task("my_test_task")


def test_duplicate_registration_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_task("single", SingleNeedleTask)


def test_adding_a_task_does_not_require_touching_core_or_cli() -> None:
    """Smoke check: importing `needlehaystack.tasks` in a fresh Python
    process does not pull in the CLI. Adding a new task must not force a
    contributor to touch `cli/`.

    We use a subprocess because pytest itself may have already imported
    `needlehaystack.cli.*` for other tests.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, needlehaystack.tasks; "
            "leaked = sorted(m for m in sys.modules if m.startswith('needlehaystack.cli')); "
            "print(','.join(leaked))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = [m for m in proc.stdout.strip().split(",") if m]
    assert not leaked, f"task package pulled in CLI modules: {leaked}"
