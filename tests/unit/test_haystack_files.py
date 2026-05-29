"""Unit tests for FilesHaystack."""

from __future__ import annotations

from needlehaystack.core import tokens
from needlehaystack.haystacks import FilesHaystack
from needlehaystack.haystacks.base import HaystackSource


def test_satisfies_protocol() -> None:
    assert isinstance(FilesHaystack(), HaystackSource)


def test_loads_bundled_paul_graham_essays_to_target_length() -> None:
    h = FilesHaystack()
    text = h.load(min_tokens=2000)
    assert tokens.count(text) >= 2000


def test_loads_enough_to_satisfy_large_request() -> None:
    # Larger than any single file but well within one cycle of the dir.
    h = FilesHaystack()
    text = h.load(min_tokens=30_000)
    assert tokens.count(text) >= 30_000


def test_descriptor_includes_type_and_path() -> None:
    h = FilesHaystack(path="PaulGrahamEssays")
    d = h.descriptor()
    assert d["type"] == "files"
    assert d["path"] == "PaulGrahamEssays"


def test_cycles_short_directory_to_reach_min_tokens(tmp_path) -> None:
    # Tiny directory with one small file; should cycle until min_tokens met.
    one_file = tmp_path / "tiny.txt"
    one_file.write_text("The needle waits patiently in the haystack. " * 5)
    h = FilesHaystack(path=str(tmp_path))
    text = h.load(min_tokens=1000)
    assert tokens.count(text) >= 1000


def test_raises_when_no_txt_files_present(tmp_path) -> None:
    import pytest

    h = FilesHaystack(path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no .txt files"):
        h.load(min_tokens=100)
