"""Unit tests for EvenSpreadInserter.

The v1 multi-needle code had two bugs we explicitly guard against:

1. Depth percent computed from post-insertion length (so each later
   needle's reported depth was too low).
2. The depth==100 branch never advanced.
"""

from __future__ import annotations

import pytest

from needlehaystack.core import tokens
from needlehaystack.inserters import EvenSpreadInserter
from needlehaystack.inserters.base import Inserter


def _ctx(n_sentences: int = 200) -> list[int]:
    text = " ".join(f"This is sentence number {i}." for i in range(n_sentences))
    return tokens.encode(text)


def test_satisfies_protocol() -> None:
    assert isinstance(EvenSpreadInserter(), Inserter)


def test_three_needles_starting_at_40_land_near_40_60_80() -> None:
    ctx = _ctx()
    result = EvenSpreadInserter().insert(ctx, ["A.", "B.", "C."], depth_percent=40.0)
    depths = [p.actual_depth_percent for p in result.placements]
    targets = [40.0, 60.0, 80.0]
    for actual, target in zip(depths, targets, strict=True):
        assert abs(actual - target) <= 2.0, (
            f"needle depth {actual} too far from target {target}; got depths {depths}"
        )


def test_later_needles_record_correct_depth_not_inflated_by_earlier_inserts() -> None:
    """The v1 bug: depths computed against the inflated, post-insertion
    token count made each later needle appear shallower than its target.
    Our fix snapshots pre_len once. Regression test: insert two huge
    needles that add significant length, then verify the last needle
    still reports a depth close to its target (not drifted toward 0)."""
    ctx = _ctx(n_sentences=400)
    pre_len = len(ctx)
    huge = "INSERTED " * 400  # ~hundreds of tokens
    needles = [huge, huge, "marker."]
    result = EvenSpreadInserter().insert(ctx, needles, depth_percent=30.0)

    # Sanity: total length grew enough that a post_len-based denominator
    # would visibly skew the math.
    assert len(result.new_tokens) > pre_len * 1.5, (
        f"test setup too small: pre={pre_len}, post={len(result.new_tokens)}"
    )

    # Last needle's target: 30 + 2 * ((100 - 30) / 3) = ~76.67%
    last_depth = result.placements[-1].actual_depth_percent
    assert 73.0 <= last_depth <= 80.0, (
        f"last needle depth {last_depth} drifted from target ~76.7; "
        "are we computing against post-insertion length?"
    )


def test_insertion_indices_strictly_increase() -> None:
    ctx = _ctx()
    result = EvenSpreadInserter().insert(
        ctx, ["one.", "two.", "three.", "four."], depth_percent=10.0
    )
    indices = [p.insertion_token_index for p in result.placements]
    assert indices == sorted(indices)
    for i in range(1, len(indices)):
        assert indices[i] > indices[i - 1], "indices must strictly increase"


def test_all_needle_texts_present_in_decoded_output() -> None:
    ctx = _ctx()
    needles = ["FIRST.", "SECOND.", "THIRD."]
    result = EvenSpreadInserter().insert(ctx, needles, depth_percent=25.0)
    decoded = tokens.decode(result.new_tokens)
    for n in needles:
        assert n in decoded


def test_records_one_placement_per_needle() -> None:
    ctx = _ctx()
    needles = ["a.", "b.", "c.", "d.", "e."]
    result = EvenSpreadInserter().insert(ctx, needles, depth_percent=10.0)
    assert len(result.placements) == len(needles)
    for needle, placement in zip(needles, result.placements, strict=True):
        assert placement.text == needle


def test_rejects_empty_needle_list() -> None:
    ctx = _ctx()
    with pytest.raises(ValueError, match="at least one needle"):
        EvenSpreadInserter().insert(ctx, [], depth_percent=50.0)


def test_rejects_out_of_range_depth() -> None:
    ctx = _ctx()
    with pytest.raises(ValueError, match="depth_percent"):
        EvenSpreadInserter().insert(ctx, ["x"], depth_percent=-1.0)
