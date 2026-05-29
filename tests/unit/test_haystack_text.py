"""Unit tests for RepeatingTextHaystack."""

from __future__ import annotations

import pytest

from needlehaystack.core import tokens
from needlehaystack.haystacks import RepeatingTextHaystack
from needlehaystack.haystacks.base import HaystackSource
from needlehaystack.haystacks.text import LARGE_TEXT_THRESHOLD_BYTES


def test_satisfies_protocol() -> None:
    assert isinstance(RepeatingTextHaystack(text="abc"), HaystackSource)


def test_repeats_until_min_tokens_satisfied() -> None:
    h = RepeatingTextHaystack(text="the needle never lies")
    out = h.load(min_tokens=1000)
    assert tokens.count(out) >= 1000


def test_handles_very_large_request() -> None:
    h = RepeatingTextHaystack(text="lorem ipsum dolor sit amet")
    out = h.load(min_tokens=50_000)
    assert tokens.count(out) >= 50_000


def test_descriptor_round_trips_source_text() -> None:
    """Descriptor must hold the full source text so reconstruction can rebuild
    the exact context the model saw."""
    text = "Some background prose. More of it."
    h = RepeatingTextHaystack(text=text, separator="\n---\n")
    d = h.descriptor()
    assert d["type"] == "text"
    assert d["text"] == text
    assert d["separator"] == "\n---\n"


def test_empty_text_raises() -> None:
    h = RepeatingTextHaystack(text="")
    with pytest.raises(ValueError, match="non-empty"):
        h.load(min_tokens=100)


def test_small_text_emits_no_warning(recwarn: pytest.WarningsRecorder) -> None:
    RepeatingTextHaystack(text="small unit")
    assert not [w for w in recwarn.list if "FilesHaystack" in str(w.message)]


def test_large_text_warns_and_points_at_files_haystack() -> None:
    big = "x" * (LARGE_TEXT_THRESHOLD_BYTES + 1)
    with pytest.warns(UserWarning, match="FilesHaystack"):
        RepeatingTextHaystack(text=big)
