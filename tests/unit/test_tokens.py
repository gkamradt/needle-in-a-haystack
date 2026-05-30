"""Unit tests for the shared tokenizer module."""

from __future__ import annotations

from needlehaystack.core import tokens


def test_encoding_name_is_cl100k_base() -> None:
    # The whole project is wired to one encoding; if this changes by
    # accident, sweep results across runs become incomparable.
    assert tokens.encoding_name() == "cl100k_base"


def test_encode_decode_round_trips_ascii() -> None:
    s = "Hello, Needle In A Haystack!"
    assert tokens.decode(tokens.encode(s)) == s


def test_encode_decode_round_trips_unicode() -> None:
    s = "café — résumé — 你好 — 🦙"
    assert tokens.decode(tokens.encode(s)) == s


def test_count_matches_len_of_encode() -> None:
    s = "the quick brown fox jumps over the lazy dog"
    assert tokens.count(s) == len(tokens.encode(s))


def test_count_is_deterministic() -> None:
    s = "needles never lie"
    assert tokens.count(s) == tokens.count(s)


def test_empty_string_round_trips() -> None:
    assert tokens.encode("") == []
    assert tokens.decode([]) == ""
    assert tokens.count("") == 0
