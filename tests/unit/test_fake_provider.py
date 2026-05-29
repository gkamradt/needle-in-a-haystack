"""Unit tests for the deterministic in-process FakeProvider."""

from __future__ import annotations

import pytest

from needlehaystack.core.types import Usage
from needlehaystack.providers.base import Completion, ModelProvider
from needlehaystack.providers.fake import FakeProvider, make_echo_prompt


def test_satisfies_model_provider_protocol() -> None:
    fp = FakeProvider()
    # `runtime_checkable` makes this a real type assertion at runtime.
    assert isinstance(fp, ModelProvider)


async def test_canned_mode_returns_configured_text() -> None:
    fp = FakeProvider(response_text="all good")
    result = await fp.complete("sys", "user")
    assert isinstance(result, Completion)
    assert result.text == "all good"
    assert result.usage is None


async def test_records_every_call() -> None:
    fp = FakeProvider()
    await fp.complete("sys a", "user a")
    await fp.complete("sys b", "user b")
    assert len(fp.calls) == 2
    assert fp.calls[0].system == "sys a"
    assert fp.calls[0].user == "user a"
    assert fp.calls[1].user == "user b"


async def test_synthesizes_usage_when_configured() -> None:
    fp = FakeProvider(usage_per_call=Usage(input_tokens=123, output_tokens=4))
    result = await fp.complete("sys", "user")
    assert result.usage == Usage(input_tokens=123, output_tokens=4)


async def test_echo_answer_round_trips_through_user_prompt() -> None:
    fp = FakeProvider(mode="echo_answer")
    answer = "uuid-12345"
    user = f"Here is some context. {make_echo_prompt(answer)} Please answer."
    result = await fp.complete("sys", user)
    assert result.text == answer


async def test_echo_answer_round_trips_through_system_prompt() -> None:
    fp = FakeProvider(mode="echo_answer")
    answer = "secret"
    system = f"You know that {make_echo_prompt(answer)} is the answer."
    result = await fp.complete(system, "user with no echo block")
    assert result.text == answer


async def test_echo_answer_raises_when_no_sentinel_present() -> None:
    fp = FakeProvider(mode="echo_answer")
    with pytest.raises(ValueError, match="ECHO"):
        await fp.complete("sys", "user without echo block")


async def test_fail_n_times_then_canned() -> None:
    fp = FakeProvider(
        mode="fail_n_times_then_canned",
        fail_until=2,
        response_text="finally",
        fail_message="boom",
    )
    with pytest.raises(RuntimeError, match="boom"):
        await fp.complete("s", "u")
    with pytest.raises(RuntimeError, match="boom"):
        await fp.complete("s", "u")
    result = await fp.complete("s", "u")
    assert result.text == "finally"
    # All three calls were recorded, including the failures.
    assert len(fp.calls) == 3


async def test_always_fail_raises_every_time() -> None:
    fp = FakeProvider(mode="always_fail", fail_message="nope")
    for _ in range(3):
        with pytest.raises(RuntimeError, match="nope"):
            await fp.complete("s", "u")
    assert len(fp.calls) == 3
