"""End-to-end runner integration tests using FakeProvider (no network)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from needlehaystack.core.runner import Pricing, Runner
from needlehaystack.core.sweep import Sweep
from needlehaystack.core.types import Usage
from needlehaystack.haystacks.text import RepeatingTextHaystack
from needlehaystack.providers.fake import FakeProvider
from needlehaystack.stores.jsonl import JsonlResultStore
from needlehaystack.tasks.single_needle import SingleNeedleTask

# ---------- shared fixtures --------------------------------------------------


def _haystack() -> RepeatingTextHaystack:
    return RepeatingTextHaystack(
        text=" ".join(f"Line number {i} of the background text." for i in range(20)),
    )


def _sweep() -> Sweep:
    return Sweep(
        lengths=[1000, 2000, 4000],
        depths=[0.0, 50.0, 100.0],
        seeds=[None],
    )


def _echo_task() -> SingleNeedleTask:
    # Customize so the FakeProvider's echo-mode finds an answer in the prompt:
    # we put the ECHO sentinels inside the question_template, around the
    # expected_answer the task already knows about.
    answer = "Dolores Park sandwich answer-XYZ"
    return SingleNeedleTask(
        needle_text=f"The very special fact is: {answer}.",
        expected_answer=answer,
        question_template=(f"What is the very special fact? <<ECHO:{answer}:ECHO>>"),
    )


# ---------- tests ------------------------------------------------------------


def test_3x3_sweep_writes_nine_ok_rows(tmp_path: Path) -> None:
    store = JsonlResultStore(tmp_path / "out.jsonl")
    runner = Runner(
        task=_echo_task(),
        provider=FakeProvider(mode="echo_answer"),
        haystack=_haystack(),
        sweep=_sweep(),
        store=store,
        run_name="rt",
    )

    results = asyncio.run(runner.run())

    assert len(results) == 9
    assert all(r.status == "ok" for r in results)
    assert all(r.score is not None and r.score.value == 1.0 for r in results)


def test_each_row_has_recipe_and_no_rendered_context(tmp_path: Path) -> None:
    store = JsonlResultStore(tmp_path / "out.jsonl")
    runner = Runner(
        task=_echo_task(),
        provider=FakeProvider(mode="echo_answer"),
        haystack=_haystack(),
        sweep=_sweep(),
        store=store,
    )
    asyncio.run(runner.run())

    lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 9
    for raw in lines:
        row = json.loads(raw)
        # recipe present and minimal
        recipe = row["recipe"]
        assert recipe["inserter"] == "single_depth"
        assert recipe["haystack"]["type"] == "text"
        assert len(recipe["needle_placements"]) == 1
        assert recipe["final_context_token_count"] >= row["context_length"]
        # No rendered context fields on the row (the haystack descriptor
        # may carry a 60-char `text_preview`; that's fine — what we want
        # to guarantee is that nothing the size of an actual rendered
        # context tagged onto the row).
        assert "rendered_context" not in row
        assert "context" not in row  # only `context_length`, no rendered text
        # Sentinel: a string that's in the haystack text but well past
        # the 60-char `text_preview` window. Confirms we're not stashing
        # the full background somewhere.
        assert "Line number 19 of the background text" not in json.dumps(row)
        # And the row is tiny — well under 10 KB.
        assert len(raw.encode("utf-8")) < 10_000


def test_resume_skips_existing_rows(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    store = JsonlResultStore(path)
    asyncio.run(
        Runner(
            task=_echo_task(),
            provider=FakeProvider(mode="echo_answer"),
            haystack=_haystack(),
            sweep=_sweep(),
            store=store,
        ).run()
    )
    first_line_count = len(path.read_text(encoding="utf-8").splitlines())
    assert first_line_count == 9

    # Re-run with the *same* sweep against a re-opened store: nothing new.
    store_b = JsonlResultStore(path)
    new_results = asyncio.run(
        Runner(
            task=_echo_task(),
            provider=FakeProvider(mode="echo_answer"),
            haystack=_haystack(),
            sweep=_sweep(),
            store=store_b,
        ).run()
    )
    assert new_results == []  # all skipped
    assert len(path.read_text(encoding="utf-8").splitlines()) == first_line_count


def test_pricing_computes_cost_usd(tmp_path: Path) -> None:
    store = JsonlResultStore(tmp_path / "out.jsonl")
    runner = Runner(
        task=_echo_task(),
        provider=FakeProvider(
            mode="echo_answer",
            usage_per_call=Usage(input_tokens=1_000_000, output_tokens=500_000),
        ),
        haystack=_haystack(),
        sweep=Sweep(lengths=[1000], depths=[50.0]),
        store=store,
        pricing=Pricing(input=5.00, output=25.00),
    )
    results = asyncio.run(runner.run())
    assert len(results) == 1
    # 1M input × $5 + 0.5M output × $25 = 5 + 12.5 = $17.50
    assert results[0].cost_usd == pytest.approx(17.50)


def test_retry_succeeds_after_failures(tmp_path: Path) -> None:
    store = JsonlResultStore(tmp_path / "out.jsonl")
    answer = "Dolores Park sandwich answer-XYZ"
    task = SingleNeedleTask(
        needle_text=f"The very special fact is: {answer}.",
        expected_answer=answer,
        question_template=f"<<ECHO:{answer}:ECHO>>",
    )
    provider = FakeProvider(
        mode="fail_n_times_then_canned",
        fail_until=2,
        response_text=answer,
    )
    runner = Runner(
        task=task,
        provider=provider,
        haystack=_haystack(),
        sweep=Sweep(lengths=[1000], depths=[50.0]),
        store=store,
        retries=3,
    )
    results = asyncio.run(runner.run())
    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].score is not None
    assert results[0].score.value == 1.0


def test_always_failing_provider_writes_error_row(tmp_path: Path) -> None:
    store = JsonlResultStore(tmp_path / "out.jsonl")
    runner = Runner(
        task=_echo_task(),
        provider=FakeProvider(mode="always_fail", fail_message="boom"),
        haystack=_haystack(),
        sweep=Sweep(lengths=[1000], depths=[50.0]),
        store=store,
        retries=1,
    )
    results = asyncio.run(runner.run())
    assert len(results) == 1
    r = results[0]
    assert r.status == "error"
    assert r.error is not None and "boom" in r.error
    assert r.response is None
    assert r.score is None


def test_on_result_callback_fires_per_cell(tmp_path: Path) -> None:
    store = JsonlResultStore(tmp_path / "out.jsonl")
    seen: list[str] = []
    runner = Runner(
        task=_echo_task(),
        provider=FakeProvider(mode="echo_answer"),
        haystack=_haystack(),
        sweep=Sweep(lengths=[1000, 2000], depths=[50.0]),
        store=store,
        on_result=lambda r: seen.append(r.status),
    )
    asyncio.run(runner.run())
    assert seen == ["ok", "ok"]
