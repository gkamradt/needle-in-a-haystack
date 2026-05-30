"""Unit tests for ExplicitDepthsInserter."""

from __future__ import annotations

import pytest

from needlehaystack.core import tokens
from needlehaystack.inserters import ExplicitDepthsInserter
from needlehaystack.inserters.base import Inserter


def _ctx(n_sentences: int = 200) -> list[int]:
    text = " ".join(f"This is sentence number {i}." for i in range(n_sentences))
    return tokens.encode(text)


def test_satisfies_protocol() -> None:
    inserter = ExplicitDepthsInserter(depth_percents=[50.0])
    assert isinstance(inserter, Inserter)


def test_inserts_at_supplied_depths() -> None:
    ctx = _ctx()
    targets = [10.0, 40.0, 75.0]
    result = ExplicitDepthsInserter(depth_percents=targets).insert(
        ctx,
        ["A.", "B.", "C."],
        depth_percent=0.0,  # ignored
    )
    for placement, target in zip(result.placements, targets, strict=True):
        assert abs(placement.actual_depth_percent - target) <= 2.0


def test_placements_returned_in_input_order_even_when_depths_unsorted() -> None:
    ctx = _ctx()
    # Provide depths out of order; placements should still come back in
    # the order the caller passed needle_texts.
    inserter = ExplicitDepthsInserter(depth_percents=[80.0, 20.0, 50.0])
    result = inserter.insert(ctx, ["LAST.", "FIRST.", "MID."], depth_percent=0.0)
    texts = [p.text for p in result.placements]
    assert texts == ["LAST.", "FIRST.", "MID."]
    # And each landed near its requested depth.
    assert abs(result.placements[0].actual_depth_percent - 80.0) <= 2.0
    assert abs(result.placements[1].actual_depth_percent - 20.0) <= 2.0
    assert abs(result.placements[2].actual_depth_percent - 50.0) <= 2.0


def test_all_needle_texts_present_in_decoded_output() -> None:
    ctx = _ctx()
    inserter = ExplicitDepthsInserter(depth_percents=[15.0, 55.0, 95.0])
    result = inserter.insert(ctx, ["EARLY.", "MIDDLE.", "LATE."], depth_percent=0.0)
    decoded = tokens.decode(result.new_tokens)
    for needle in ["EARLY.", "MIDDLE.", "LATE."]:
        assert needle in decoded


def test_mismatched_lengths_raise() -> None:
    ctx = _ctx()
    inserter = ExplicitDepthsInserter(depth_percents=[10.0, 50.0])
    with pytest.raises(ValueError, match="must match"):
        inserter.insert(ctx, ["one"], depth_percent=0.0)


def test_construct_rejects_out_of_range_depth() -> None:
    with pytest.raises(ValueError, match="\\[0, 100\\]"):
        ExplicitDepthsInserter(depth_percents=[50.0, 150.0])
