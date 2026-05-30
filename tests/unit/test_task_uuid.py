"""Integration test for UuidNeedleTask end-to-end pipeline."""

from __future__ import annotations

import asyncio
import uuid

from needlehaystack.core import tokens
from needlehaystack.providers.fake import FakeProvider, make_echo_prompt
from needlehaystack.tasks.base import Task
from needlehaystack.tasks.uuid_needle import UuidNeedleTask


def _context(n_sentences: int = 200) -> list[int]:
    text = " ".join(f"Filler sentence {i}." for i in range(n_sentences))
    return tokens.encode(text)


def test_satisfies_protocol() -> None:
    assert isinstance(UuidNeedleTask(), Task)


def test_name_and_inserter_name() -> None:
    task = UuidNeedleTask()
    assert task.name == "uuid"
    assert task.inserter_name == "single_depth"


def test_generated_needle_carries_uuid_in_metadata() -> None:
    task = UuidNeedleTask()
    needle = task.generate_needle(seed=42)
    # Should be a parseable UUID and match the expected answer.
    parsed = uuid.UUID(needle.expected_answer)
    assert str(parsed) == needle.metadata["uuid"]


def test_seed_makes_needle_deterministic() -> None:
    task = UuidNeedleTask()
    a = task.generate_needle(seed=7)
    b = task.generate_needle(seed=7)
    assert a.expected_answer == b.expected_answer


def test_echo_provider_scores_full_marks() -> None:
    task = UuidNeedleTask()
    needle = task.generate_needle(seed=123)

    new_tokens, placements = task.insert(_context(), needle, depth_percent=25.0)
    assert needle.expected_answer in tokens.decode(new_tokens)
    assert len(placements) == 1

    prompt = task.question(needle) + " " + make_echo_prompt(needle.expected_answer)
    provider = FakeProvider(mode="echo_answer")
    completion = asyncio.run(provider.complete(system="", user=prompt))

    score = task.score(completion.text, needle)
    assert score.value == 1.0
