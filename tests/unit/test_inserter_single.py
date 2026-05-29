"""Unit tests for SingleDepthInserter."""

from __future__ import annotations

import pytest

from needlehaystack.core import tokens
from needlehaystack.inserters import SingleDepthInserter
from needlehaystack.inserters.base import Inserter


def _context_with_periods(n_sentences: int = 100) -> list[int]:
    text = " ".join(f"This is sentence number {i}." for i in range(n_sentences))
    return tokens.encode(text)


def test_satisfies_protocol() -> None:
    assert isinstance(SingleDepthInserter(), Inserter)


def test_insert_at_zero_puts_needle_at_start() -> None:
    ctx = _context_with_periods()
    result = SingleDepthInserter().insert(ctx, ["NEEDLE TEXT."], depth_percent=0.0)
    decoded = tokens.decode(result.new_tokens)
    assert decoded.startswith("NEEDLE TEXT.")
    assert result.placements[0].insertion_token_index == 0
    assert result.placements[0].actual_depth_percent == 0.0


def test_insert_at_hundred_appends_needle() -> None:
    ctx = _context_with_periods()
    result = SingleDepthInserter().insert(ctx, ["END NEEDLE."], depth_percent=100.0)
    decoded = tokens.decode(result.new_tokens)
    assert decoded.endswith("END NEEDLE.")
    assert result.placements[0].insertion_token_index == len(ctx)


def test_insert_at_fifty_lands_around_middle() -> None:
    ctx = _context_with_periods()
    result = SingleDepthInserter().insert(ctx, ["MID NEEDLE."], depth_percent=50.0)
    decoded = tokens.decode(result.new_tokens)
    assert "MID NEEDLE." in decoded
    actual = result.placements[0].actual_depth_percent
    assert 40.0 <= actual <= 60.0, f"actual depth {actual} too far from 50%"


def test_placement_insertion_index_is_within_new_tokens() -> None:
    ctx = _context_with_periods()
    result = SingleDepthInserter().insert(ctx, ["X."], depth_percent=42.0)
    p = result.placements[0]
    assert 0 <= p.insertion_token_index < len(result.new_tokens)


def test_falls_back_to_raw_point_when_no_period_in_window() -> None:
    # Context with no periods at all → period-snapping has nothing to find.
    # Bounded loop should give up and use the raw insertion point rather
    # than walking back to 0 or hanging.
    no_periods = tokens.encode("alpha bravo charlie delta echo foxtrot " * 200)
    result = SingleDepthInserter().insert(no_periods, ["N."], depth_percent=50.0)
    actual = result.placements[0].actual_depth_percent
    assert 40.0 <= actual <= 50.0, (
        f"with no periods we should still land near the target, got {actual}"
    )


def test_rejects_more_than_one_needle() -> None:
    ctx = _context_with_periods()
    with pytest.raises(ValueError, match="exactly 1 needle"):
        SingleDepthInserter().insert(ctx, ["a", "b"], depth_percent=50.0)


def test_rejects_out_of_range_depth() -> None:
    ctx = _context_with_periods()
    with pytest.raises(ValueError, match="depth_percent"):
        SingleDepthInserter().insert(ctx, ["x"], depth_percent=150.0)


def test_period_snapping_can_be_disabled() -> None:
    ctx = _context_with_periods()
    snapped = SingleDepthInserter(snap_to_periods=True).insert(ctx, ["X."], depth_percent=33.0)
    raw = SingleDepthInserter(snap_to_periods=False).insert(ctx, ["X."], depth_percent=33.0)
    # Raw insertion lands exactly at int(len * 33/100); snapped may have
    # walked back to a sentence boundary, so its index is ≤ the raw one.
    raw_idx = int(len(ctx) * 0.33)
    assert raw.placements[0].insertion_token_index == raw_idx
    assert snapped.placements[0].insertion_token_index <= raw_idx
