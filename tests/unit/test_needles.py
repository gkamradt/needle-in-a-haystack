"""Unit tests for needle generators."""

from __future__ import annotations

import re
import uuid

import pytest

from needlehaystack.needles import StaticNeedle, UuidChain, UuidSingle
from needlehaystack.needles.base import NeedleGenerator

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


# --- StaticNeedle --------------------------------------------------------


def test_static_satisfies_protocol() -> None:
    assert isinstance(StaticNeedle(texts=["x"], expected_answer="x"), NeedleGenerator)


def test_static_returns_configured_values() -> None:
    n = StaticNeedle(
        texts=["secret fact"],
        expected_answer="secret",
        metadata={"category": "fact"},
    ).generate()
    assert n.texts == ["secret fact"]
    assert n.expected_answer == "secret"
    assert n.metadata == {"category": "fact"}


def test_static_returns_fresh_copies_each_call() -> None:
    s = StaticNeedle(texts=["a"], expected_answer="a", metadata={"k": 1})
    a = s.generate()
    b = s.generate()
    a.metadata["mutated"] = True
    a.texts.append("extra")
    assert "mutated" not in b.metadata
    assert b.texts == ["a"]


# --- UuidSingle ----------------------------------------------------------


def test_uuid_single_satisfies_protocol() -> None:
    assert isinstance(UuidSingle(), NeedleGenerator)


def test_uuid_single_emits_valid_uuid_in_text() -> None:
    n = UuidSingle().generate()
    assert UUID_RE.match(n.expected_answer)
    # Default template embeds the UUID directly into the inserted text.
    assert n.expected_answer in n.texts[0]


def test_uuid_single_is_deterministic_with_seed() -> None:
    a = UuidSingle().generate(seed=42)
    b = UuidSingle().generate(seed=42)
    c = UuidSingle().generate(seed=43)
    assert a.expected_answer == b.expected_answer
    assert a.expected_answer != c.expected_answer


def test_uuid_single_respects_template() -> None:
    n = UuidSingle(template="The code: {uuid}!").generate(seed=1)
    assert n.texts[0].startswith("The code: ")
    assert n.texts[0].endswith("!")
    # Make sure str(uuid.UUID(...)) is a valid UUID v4-shaped string.
    uuid.UUID(n.expected_answer)


# --- UuidChain -----------------------------------------------------------


def test_uuid_chain_satisfies_protocol() -> None:
    assert isinstance(UuidChain(), NeedleGenerator)


def test_uuid_chain_emits_n_minus_1_links_and_full_chain_metadata() -> None:
    n = UuidChain(chain_length=5).generate(seed=7)
    assert len(n.texts) == 4
    chain = n.metadata["chain"]
    assert len(chain) == 5
    assert all(UUID_RE.match(u) for u in chain)
    assert n.expected_answer == chain[-1]
    assert n.metadata["start"] == chain[0]


def test_uuid_chain_links_match_chain_pairs() -> None:
    n = UuidChain(chain_length=4).generate(seed=11)
    chain = n.metadata["chain"]
    for i, link in enumerate(n.texts):
        # Default template embeds both prev and next.
        assert chain[i] in link
        assert chain[i + 1] in link


def test_uuid_chain_is_deterministic_with_seed() -> None:
    a = UuidChain(chain_length=5).generate(seed=99)
    b = UuidChain(chain_length=5).generate(seed=99)
    assert a.metadata["chain"] == b.metadata["chain"]


def test_uuid_chain_rejects_length_below_2() -> None:
    with pytest.raises(ValueError, match="chain_length"):
        UuidChain(chain_length=1)
