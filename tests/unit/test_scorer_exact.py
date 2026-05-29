"""Unit tests for ExactMatchScorer."""

from __future__ import annotations

from needlehaystack.core.types import Needle
from needlehaystack.scorers import ExactMatchScorer
from needlehaystack.scorers.base import Scorer


def test_satisfies_protocol() -> None:
    assert isinstance(ExactMatchScorer(), Scorer)


def test_hits_on_substring_match() -> None:
    needle = Needle(texts=["fact"], expected_answer="rosebud")
    s = ExactMatchScorer().score(
        response="The answer to the puzzle is rosebud, no doubt.", needle=needle
    )
    assert s.value == 1.0
    assert s.details == {"matches": 1, "total": 1}


def test_case_insensitive() -> None:
    needle = Needle(texts=["x"], expected_answer="RoseBud")
    s = ExactMatchScorer().score(response="rosebud", needle=needle)
    assert s.value == 1.0


def test_strips_whitespace() -> None:
    needle = Needle(texts=["x"], expected_answer="  rosebud  ")
    s = ExactMatchScorer().score(response="\n  rosebud  \t", needle=needle)
    assert s.value == 1.0


def test_misses_cleanly_when_expected_absent() -> None:
    needle = Needle(texts=["x"], expected_answer="rosebud")
    s = ExactMatchScorer().score(response="no such thing in this response", needle=needle)
    assert s.value == 0.0
    assert s.details == {"matches": 0, "total": 1}


def test_multi_fragment_via_metadata_partial_match() -> None:
    needle = Needle(
        texts=["x"],
        expected_answer="unused",
        metadata={"expected_fragments": ["figs", "prosciutto", "goat cheese"]},
    )
    # Only two of three present.
    s = ExactMatchScorer().score(
        response="A pizza needs figs and prosciutto, but I forgot the third.",
        needle=needle,
    )
    assert s.value == 2 / 3
    assert s.details == {"matches": 2, "total": 3}


def test_multi_fragment_full_match_scores_one() -> None:
    needle = Needle(
        texts=["x"],
        expected_answer="unused",
        metadata={"expected_fragments": ["a", "b", "c"]},
    )
    s = ExactMatchScorer().score(response="abc", needle=needle)
    assert s.value == 1.0
