"""Tests for `niah reconstruct` — the recipe-replay payoff."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from needlehaystack.cli.reconstruct import reconstruct_from_jsonl, reconstruct_row
from needlehaystack.core import tokens
from needlehaystack.core.runner import Runner
from needlehaystack.core.sweep import Sweep
from needlehaystack.haystacks.text import RepeatingTextHaystack
from needlehaystack.providers.fake import FakeProvider
from needlehaystack.stores.jsonl import JsonlResultStore
from needlehaystack.tasks.single_needle import SingleNeedleTask


def _haystack() -> RepeatingTextHaystack:
    return RepeatingTextHaystack(
        text=" ".join(f"Background line number {i}." for i in range(50)),
    )


def _task() -> SingleNeedleTask:
    answer = "Dolores Park sandwich Q-12345"
    return SingleNeedleTask(
        needle_text=f"The very special fact is: {answer}.",
        expected_answer=answer,
        question_template=f"What is the very special fact? <<ECHO:{answer}:ECHO>>",
    )


def _make_sweep_and_run(tmp_path: Path) -> tuple[Path, dict[tuple[int, float], str]]:
    """Run a 2×2 sweep and recreate each rendered context for later comparison.

    SingleNeedleTask is fully deterministic (static needle, single-depth
    inserter), so we can compute the expected rendered context outside
    the runner by replaying the same `load → encode → truncate → insert`
    sequence. That avoids monkeypatching a slots dataclass.
    """
    haystack = _haystack()
    task = _task()
    lengths = [600, 900]
    depths = [25.0, 75.0]

    captured: dict[tuple[int, float], str] = {}
    for L in lengths:
        ctx_tokens = tokens.encode(haystack.load(min_tokens=L))[:L]
        needle = task.generate_needle(seed=None)
        for d in depths:
            new_tokens, _ = task.insert(ctx_tokens, needle, d)
            captured[(L, d)] = tokens.decode(new_tokens)

    out_path = tmp_path / "results.jsonl"
    store = JsonlResultStore(out_path)
    runner = Runner(
        task=task,
        provider=FakeProvider(mode="echo_answer"),
        haystack=haystack,
        sweep=Sweep(lengths=lengths, depths=depths),
        store=store,
        run_name="reconstruct-test",
    )
    asyncio.run(runner.run())
    return out_path, captured


def test_reconstruct_matches_captured_context(tmp_path: Path) -> None:
    out_path, captured = _make_sweep_and_run(tmp_path)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4

    for i, line in enumerate(lines):
        row = json.loads(line)
        rebuilt = reconstruct_row(row)
        key = (row["context_length"], row["target_depth_percent"])
        assert rebuilt == captured[key], f"row {i} mismatch for key {key}"


def test_reconstruct_is_deterministic(tmp_path: Path) -> None:
    out_path, _ = _make_sweep_and_run(tmp_path)
    a = reconstruct_from_jsonl(out_path, row_index=0)
    b = reconstruct_from_jsonl(out_path, row_index=0)
    assert a == b


def test_reconstruct_bad_row_raises(tmp_path: Path) -> None:
    out_path, _ = _make_sweep_and_run(tmp_path)
    with pytest.raises(IndexError):
        reconstruct_from_jsonl(out_path, row_index=99)


def test_reconstruction_contains_needle(tmp_path: Path) -> None:
    """Sanity check: every rebuilt context contains the needle text."""
    out_path, _ = _make_sweep_and_run(tmp_path)
    for line in out_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rebuilt = reconstruct_row(row)
        for placement in row["recipe"]["needle_placements"]:
            assert placement["text"] in rebuilt
