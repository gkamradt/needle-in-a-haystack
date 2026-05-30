"""Tests for the JSONL result store (append, resume, concurrency)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from needlehaystack.core.types import (
    NeedlePlacement,
    Recipe,
    Score,
    TestResult,
    Usage,
)
from needlehaystack.stores.base import ResultStore, key_for
from needlehaystack.stores.jsonl import JsonlResultStore


def _make_result(
    *,
    context_length: int = 1000,
    depth: float = 50.0,
    run_name: str = "run-1",
    model_id: str = "fake-model",
    seed: int | None = None,
) -> TestResult:
    return TestResult(
        schema_version=2,
        run_name=run_name,
        model_id=model_id,
        model_request_name="fake",
        task_type="single",
        context_length=context_length,
        target_depth_percent=depth,
        recipe=Recipe(
            haystack={"type": "text", "text_preview": "abc"},
            inserter="single_depth",
            needle_placements=[
                NeedlePlacement(text="x", insertion_token_index=10, actual_depth_percent=depth)
            ],
            final_context_token_count=context_length + 5,
        ),
        needle_metadata={"k": "v"},
        expected_answer="answer",
        prompt_question="?",
        response="ok",
        score=Score(value=1.0, details={"matches": 1, "total": 1}),
        usage=Usage(input_tokens=100, output_tokens=10),
        cost_usd=0.0,
        duration_seconds=0.01,
        status="ok",
        error=None,
        timestamp_utc="2026-01-01T00:00:00+00:00",
        seed=seed,
    )


def test_satisfies_protocol(tmp_path: Path) -> None:
    store = JsonlResultStore(tmp_path / "out.jsonl")
    assert isinstance(store, ResultStore)


def test_append_writes_one_line_per_row(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    store = JsonlResultStore(path)
    asyncio.run(store.append(_make_result(depth=0.0)))
    asyncio.run(store.append(_make_result(depth=50.0)))
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") == 2
    for line in text.splitlines():
        json.loads(line)  # each line is independently parseable


def test_append_then_reopen_resume_index_populated(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    store_a = JsonlResultStore(path)
    asyncio.run(store_a.append(_make_result(depth=0.0)))
    asyncio.run(store_a.append(_make_result(depth=100.0)))

    store_b = JsonlResultStore(path)  # fresh process simulation
    assert len(store_b) == 2
    assert store_b.already_has(key_for(_make_result(depth=0.0)))
    assert store_b.already_has(key_for(_make_result(depth=100.0)))
    assert not store_b.already_has(key_for(_make_result(depth=42.0)))


def test_concurrent_appends_serialize_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    store = JsonlResultStore(path)

    async def go() -> None:
        await asyncio.gather(
            store.append(_make_result(depth=0.0)),
            store.append(_make_result(depth=25.0)),
            store.append(_make_result(depth=50.0)),
            store.append(_make_result(depth=75.0)),
            store.append(_make_result(depth=100.0)),
        )

    asyncio.run(go())

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        json.loads(line)  # well-formed; no interleaving


def test_malformed_trailing_line_is_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "out.jsonl"
    # Write one good line + one corrupted trailing line.
    store = JsonlResultStore(path)
    asyncio.run(store.append(_make_result(depth=0.0)))
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"this is not valid json\n')

    with caplog.at_level("WARNING"):
        reopened = JsonlResultStore(path)
    assert len(reopened) == 1  # only the good line counted
    assert any("malformed" in m for m in caplog.messages)


def test_directory_is_created_if_missing(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "subdir" / "out.jsonl"
    JsonlResultStore(path)
    assert path.parent.is_dir()
