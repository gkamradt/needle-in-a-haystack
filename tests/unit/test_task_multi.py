"""Integration test for MultiNeedleTask end-to-end pipeline."""

from __future__ import annotations

import pytest

from needlehaystack.core import tokens
from needlehaystack.tasks.base import Task
from needlehaystack.tasks.multi_needle import MultiNeedleTask


def _context(n_sentences: int = 400) -> list[int]:
    text = " ".join(f"Line number {i} in the haystack." for i in range(n_sentences))
    return tokens.encode(text)


def test_satisfies_protocol() -> None:
    assert isinstance(MultiNeedleTask(), Task)


def test_name_and_inserter_name() -> None:
    task = MultiNeedleTask()
    assert task.name == "multi"
    assert task.inserter_name == "even_spread"


def test_default_inserts_three_needles_and_full_match_scores_one() -> None:
    task = MultiNeedleTask()
    needle = task.generate_needle(seed=None)
    assert len(needle.texts) == 3

    new_tokens, placements = task.insert(_context(), needle, depth_percent=0.0)
    assert len(placements) == 3
    decoded = tokens.decode(new_tokens)
    for text in needle.texts:
        assert text in decoded

    # Response mentions every expected fragment → exact 1.0.
    response = "The ingredients are figs, prosciutto, and goat cheese."
    score = task.score(response, needle)
    assert score.value == 1.0
    assert score.details == {"matches": 3, "total": 3}


def test_partial_response_scores_fraction() -> None:
    task = MultiNeedleTask()
    needle = task.generate_needle(seed=None)

    response = "I think you need figs and goat cheese."
    score = task.score(response, needle)
    assert score.value == pytest.approx(2 / 3)
    assert score.details == {"matches": 2, "total": 3}


def test_empty_needle_texts_rejected() -> None:
    with pytest.raises(ValueError):
        MultiNeedleTask(needle_texts=[], expected_fragments=["x"])


def test_empty_fragments_rejected() -> None:
    with pytest.raises(ValueError):
        MultiNeedleTask(needle_texts=["x"], expected_fragments=[])
