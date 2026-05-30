"""Integration test for SingleNeedleTask end-to-end pipeline."""

from __future__ import annotations

import asyncio

from needlehaystack.core import tokens
from needlehaystack.providers.fake import FakeProvider, make_echo_prompt
from needlehaystack.tasks.base import Task
from needlehaystack.tasks.single_needle import SingleNeedleTask


def _context(n_sentences: int = 200) -> list[int]:
    text = " ".join(f"This is sentence number {i}." for i in range(n_sentences))
    return tokens.encode(text)


def test_satisfies_protocol() -> None:
    assert isinstance(SingleNeedleTask(), Task)


def test_name_and_inserter_name() -> None:
    task = SingleNeedleTask()
    assert task.name == "single"
    assert task.inserter_name == "single_depth"


def test_pipeline_with_echo_provider_scores_full_marks() -> None:
    task = SingleNeedleTask()
    needle = task.generate_needle(seed=None)

    new_tokens, placements = task.insert(_context(), needle, depth_percent=50.0)
    assert len(placements) == 1
    assert needle.texts[0] in tokens.decode(new_tokens)

    # FakeProvider echoes whatever's between the ECHO sentinels in the
    # user prompt. That lets us simulate "model produced the expected
    # answer" without any heuristics.
    prompt = task.question(needle) + " " + make_echo_prompt(needle.expected_answer)
    provider = FakeProvider(mode="echo_answer")
    completion = asyncio.run(provider.complete(system="", user=prompt))

    score = task.score(completion.text, needle)
    assert score.value == 1.0


def test_pipeline_with_wrong_response_scores_zero() -> None:
    task = SingleNeedleTask()
    needle = task.generate_needle(seed=None)
    score = task.score("totally unrelated answer", needle)
    assert score.value == 0.0


def test_question_is_the_configured_template() -> None:
    task = SingleNeedleTask(question_template="What is the answer?")
    needle = task.generate_needle(seed=None)
    assert task.question(needle) == "What is the answer?"
