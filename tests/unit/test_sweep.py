"""Tests for sweep grid builders."""

from __future__ import annotations

import pytest

from needlehaystack.core.sweep import (
    Sweep,
    build_depths,
    build_lengths,
    from_spec,
)

# --- build_lengths -----------------------------------------------------------


def test_build_lengths_linear() -> None:
    lengths = build_lengths(min=1000, max=10_000, num=10)
    assert len(lengths) == 10
    assert lengths[0] == 1000
    assert lengths[-1] == 10_000
    assert lengths == sorted(lengths)


def test_build_lengths_single_value() -> None:
    assert build_lengths(min=4000, max=4000, num=1) == [4000]


def test_build_lengths_rejects_bad_num() -> None:
    with pytest.raises(ValueError):
        build_lengths(min=0, max=10, num=0)


def test_build_lengths_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        build_lengths(min=100, max=10, num=5)


# --- build_depths ------------------------------------------------------------


def test_build_depths_linear_endpoints() -> None:
    depths = build_depths(min=0, max=100, num=11)
    assert depths[0] == 0.0
    assert depths[-1] == 100.0
    assert len(depths) == 11


def test_sigmoid_endpoints_pin_to_min_and_max() -> None:
    depths = build_depths(min=0, max=100, num=11, scale="sigmoid")
    assert depths[0] == pytest.approx(0.0, abs=1e-9)
    assert depths[-1] == pytest.approx(100.0, abs=1e-9)


def test_sigmoid_is_monotonic_non_decreasing() -> None:
    depths = build_depths(min=0, max=100, num=15, scale="sigmoid")
    for a, b in zip(depths, depths[1:], strict=False):
        assert b >= a


def test_sigmoid_bunches_near_endpoints() -> None:
    """First gap and last gap should be smaller than the middle gap."""
    d = build_depths(min=0, max=100, num=9, scale="sigmoid")
    first_gap = d[1] - d[0]
    middle_gap = d[len(d) // 2 + 1] - d[len(d) // 2]
    last_gap = d[-1] - d[-2]
    assert first_gap < middle_gap
    assert last_gap < middle_gap


def test_unknown_scale_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scale"):
        build_depths(min=0, max=100, num=5, scale="potato")  # type: ignore[arg-type]


# --- from_spec ---------------------------------------------------------------


def test_from_spec_list_passes_through_as_lengths() -> None:
    assert from_spec([1000, 2000, 4000], kind="length") == [1000, 2000, 4000]


def test_from_spec_list_passes_through_as_depths() -> None:
    assert from_spec([0, 25, 50, 75, 100], kind="depth") == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_from_spec_dict_dispatches_to_builder() -> None:
    spec = {"min": 0, "max": 100, "num": 5, "scale": "linear"}
    out = from_spec(spec, kind="depth")
    assert out == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_from_spec_invalid_type_rejected() -> None:
    with pytest.raises(TypeError):
        from_spec("not a spec", kind="length")  # type: ignore[arg-type]


# --- Sweep dataclass ---------------------------------------------------------


def test_sweep_cells_is_cartesian_product() -> None:
    sw = Sweep(lengths=[1000, 2000], depths=[0.0, 50.0, 100.0], seeds=[None])
    cells = sw.cells()
    assert len(cells) == 6
    assert (1000, 0.0, None) in cells
    assert (2000, 100.0, None) in cells


def test_sweep_with_multiple_seeds_multiplies_cells() -> None:
    sw = Sweep(lengths=[1000], depths=[50.0], seeds=[1, 2, 3])
    assert len(sw.cells()) == 3


def test_sweep_rejects_empty_lengths() -> None:
    with pytest.raises(ValueError):
        Sweep(lengths=[], depths=[50.0])


def test_sweep_rejects_empty_depths() -> None:
    with pytest.raises(ValueError):
        Sweep(lengths=[1000], depths=[])
