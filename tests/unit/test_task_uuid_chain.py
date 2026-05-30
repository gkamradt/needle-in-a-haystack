"""Integration test for UuidChainTask end-to-end pipeline."""

from __future__ import annotations

import pytest

from needlehaystack.core import tokens
from needlehaystack.tasks.base import Task
from needlehaystack.tasks.uuid_chain import UuidChainTask


def _context(n_sentences: int = 500) -> list[int]:
    text = " ".join(f"Filler sentence number {i}." for i in range(n_sentences))
    return tokens.encode(text)


def test_satisfies_protocol() -> None:
    assert isinstance(UuidChainTask(), Task)


def test_name_and_inserter_name() -> None:
    task = UuidChainTask()
    assert task.name == "uuid_chain"
    assert task.inserter_name == "even_spread"


def test_chain_length_drives_needle_text_count() -> None:
    task = UuidChainTask(chain_length=5)
    needle = task.generate_needle(seed=1)
    assert len(needle.texts) == 4  # N-1 link statements
    assert len(needle.metadata["chain"]) == 5
    assert needle.expected_answer == needle.metadata["chain"][-1]


def test_question_does_not_reveal_chain_structure() -> None:
    """Default question must not mention chains, hops, links, or follow."""
    task = UuidChainTask()
    needle = task.generate_needle(seed=2)
    q = task.question(needle).lower()
    for banned in ("chain", "hop", "link", "follow", "next", "maps to"):
        assert banned not in q, f"chain-revealing word {banned!r} appears in default question"
    # But it must reference the starting UUID.
    assert needle.metadata["start"] in task.question(needle)


def test_final_uuid_in_response_scores_one() -> None:
    task = UuidChainTask(chain_length=5)
    needle = task.generate_needle(seed=3)

    new_tokens, placements = task.insert(_context(), needle, depth_percent=10.0)
    assert len(placements) == 4
    decoded = tokens.decode(new_tokens)
    for link in needle.texts:
        assert link in decoded

    # Response with just the final UUID: full marks.
    score = task.score(f"The answer is {needle.expected_answer}.", needle)
    assert score.value == 1.0


def test_first_hop_only_scores_one_fifth() -> None:
    task = UuidChainTask(chain_length=5)
    needle = task.generate_needle(seed=4)
    chain = needle.metadata["chain"]

    # Response that only mentions the starting UUID (chain[0]) → 1/5.
    score = task.score(f"I see {chain[0]} but I'm not sure what comes next.", needle)
    assert score.value == pytest.approx(1 / 5)
    assert score.details == {"hops_correct": 1, "chain_length": 5}


def test_two_hops_scores_two_fifths() -> None:
    task = UuidChainTask(chain_length=5)
    needle = task.generate_needle(seed=5)
    chain = needle.metadata["chain"]

    score = task.score(f"I followed {chain[0]} to {chain[1]}.", needle)
    assert score.value == pytest.approx(2 / 5)


def test_seed_makes_chain_deterministic() -> None:
    a = UuidChainTask(chain_length=4).generate_needle(seed=9)
    b = UuidChainTask(chain_length=4).generate_needle(seed=9)
    assert a.metadata["chain"] == b.metadata["chain"]
