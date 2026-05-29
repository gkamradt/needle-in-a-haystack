"""Unit tests for RepeatingTextHaystack."""

from __future__ import annotations

import pytest

from needlehaystack.core import tokens
from needlehaystack.haystacks import RepeatingTextHaystack
from needlehaystack.haystacks.base import HaystackSource


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


def test_descriptor_truncates_preview() -> None:
    long_text = "a" * 500
    h = RepeatingTextHaystack(text=long_text)
    d = h.descriptor()
    assert d["type"] == "text"
    assert len(d["text_preview"]) <= 60


def test_empty_text_raises() -> None:
    h = RepeatingTextHaystack(text="")
    with pytest.raises(ValueError, match="non-empty"):
        h.load(min_tokens=100)
