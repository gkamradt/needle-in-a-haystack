"""Unit tests for ChainMatchScorer.

Test fixtures use realistic UUID-shaped strings rather than single
letters: single letters substring-match all over normal English text
(e.g. "a" is in "and", "e" is in "see"), which would make the scoring
checks meaningless. The production usage is always real UUIDs.
"""

from __future__ import annotations

import pytest

from needlehaystack.core.types import Needle
from needlehaystack.scorers import ChainMatchScorer
from needlehaystack.scorers.base import Scorer

# Five distinctive, easy-to-eyeball "UUIDs" — long enough that no two
# substring-match into each other or into casual prose.
UUID_A = "uuid-aaaa1111"
UUID_B = "uuid-bbbb2222"
UUID_C = "uuid-cccc3333"
UUID_D = "uuid-dddd4444"
UUID_E = "uuid-eeee5555"
FIVE_CHAIN = [UUID_A, UUID_B, UUID_C, UUID_D, UUID_E]


def _needle(chain: list[str]) -> Needle:
    return Needle(
        texts=[f"{a} maps to {b}." for a, b in zip(chain, chain[1:], strict=False)],
        expected_answer=chain[-1],
        metadata={"chain": chain, "start": chain[0]},
    )


def test_satisfies_protocol() -> None:
    assert isinstance(ChainMatchScorer(), Scorer)


def test_full_chain_in_response_scores_one() -> None:
    needle = _needle(FIVE_CHAIN)
    response = f"I followed {UUID_A} → {UUID_B} → {UUID_C} → {UUID_D} → {UUID_E}."
    s = ChainMatchScorer().score(response=response, needle=needle)
    assert s.value == 1.0
    assert s.details == {"hops_correct": 5, "chain_length": 5}


def test_final_answer_only_still_scores_one() -> None:
    """The default UUID-chain prompt asks the model for the final value;
    if the model just outputs that value, we award full credit."""
    needle = _needle(FIVE_CHAIN)
    s = ChainMatchScorer().score(response=UUID_E, needle=needle)
    assert s.value == 1.0
    assert s.details["hops_correct"] == 5


def test_first_two_of_five_scores_0_4() -> None:
    needle = _needle(FIVE_CHAIN)
    s = ChainMatchScorer().score(
        response=f"I see {UUID_A} and then {UUID_B}, but I'm stuck.",
        needle=needle,
    )
    assert s.value == pytest.approx(0.4)
    assert s.details == {"hops_correct": 2, "chain_length": 5}


def test_no_chain_node_present_scores_zero() -> None:
    needle = _needle(FIVE_CHAIN)
    s = ChainMatchScorer().score(response="totally unrelated answer", needle=needle)
    assert s.value == 0.0
    assert s.details == {"hops_correct": 0, "chain_length": 5}


def test_case_insensitive_match_on_uuids() -> None:
    chain = ["AAAA-BBBB-1111", "CCCC-DDDD-2222"]
    needle = _needle(chain)
    s = ChainMatchScorer().score(response="the value is cccc-dddd-2222", needle=needle)
    assert s.value == 1.0


def test_furthest_index_wins_not_order_of_appearance() -> None:
    """If the response mentions a later UUID, the scorer credits the
    model for getting that far even if intermediate nodes are absent."""
    needle = _needle(FIVE_CHAIN)
    s = ChainMatchScorer().score(
        response=f"skipping ahead, the answer involves {UUID_D}.", needle=needle
    )
    assert s.details["hops_correct"] == 4
    assert s.value == pytest.approx(0.8)


def test_missing_chain_metadata_raises() -> None:
    bad_needle = Needle(texts=["x"], expected_answer="x")
    with pytest.raises(ValueError, match="chain"):
        ChainMatchScorer().score(response="anything", needle=bad_needle)
