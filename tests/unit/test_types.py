"""Unit tests for the core dataclasses."""

from __future__ import annotations

from dataclasses import asdict

from needlehaystack.core.types import (
    Needle,
    NeedlePlacement,
    Recipe,
    Score,
    TestCase,
    TestResult,
    Usage,
)


def test_needle_defaults_metadata_to_empty_dict() -> None:
    n = Needle(texts=["hello"], expected_answer="hello")
    assert n.metadata == {}


def test_needle_metadata_instances_are_not_shared() -> None:
    # Regression guard: a naive `metadata: dict = {}` default would share
    # the same dict between instances.
    a = Needle(texts=["a"], expected_answer="a")
    b = Needle(texts=["b"], expected_answer="b")
    a.metadata["k"] = 1
    assert b.metadata == {}


def test_needle_placement_round_trips() -> None:
    p = NeedlePlacement(text="x", insertion_token_index=10, actual_depth_percent=50.0)
    d = asdict(p)
    assert d == {"text": "x", "insertion_token_index": 10, "actual_depth_percent": 50.0}
    assert NeedlePlacement(**d) == p


def test_recipe_serializes_to_schema_shape() -> None:
    placements = [
        NeedlePlacement(text="a→b", insertion_token_index=100, actual_depth_percent=25.0),
        NeedlePlacement(text="b→c", insertion_token_index=200, actual_depth_percent=50.0),
    ]
    r = Recipe(
        haystack={"type": "files", "path": "PaulGrahamEssays"},
        inserter="even_spread",
        needle_placements=placements,
        final_context_token_count=4096,
    )
    d = asdict(r)

    # Keys match the schema documented in docs/refactor/overview.md.
    assert set(d.keys()) == {
        "haystack",
        "inserter",
        "needle_placements",
        "final_context_token_count",
    }
    assert d["haystack"] == {"type": "files", "path": "PaulGrahamEssays"}
    assert d["inserter"] == "even_spread"
    assert d["final_context_token_count"] == 4096
    assert len(d["needle_placements"]) == 2
    assert d["needle_placements"][0]["insertion_token_index"] == 100


def test_usage_is_simple_pair() -> None:
    u = Usage(input_tokens=100, output_tokens=20)
    assert asdict(u) == {"input_tokens": 100, "output_tokens": 20}


def test_test_case_seed_is_optional() -> None:
    a = TestCase(context_length=1000, depth_percent=50.0)
    b = TestCase(context_length=1000, depth_percent=50.0, seed=42)
    assert a.seed is None
    assert b.seed == 42


def test_score_defaults_details_to_empty_dict() -> None:
    s = Score(value=1.0)
    assert s.details == {}


def test_test_result_has_no_rendered_context_field() -> None:
    """A regression check that we never accidentally re-add a `context`
    field. The whole point of the v2 schema is that the rendered context
    is reconstructable, not stored."""
    placeholder_recipe = Recipe(
        haystack={"type": "files", "path": "x"},
        inserter="single_depth",
        needle_placements=[],
        final_context_token_count=0,
    )
    r = TestResult(
        schema_version=2,
        run_name="t",
        model_id="fake",
        model_request_name="fake-model",
        task_type="single",
        context_length=1000,
        target_depth_percent=50.0,
        recipe=placeholder_recipe,
        needle_metadata={},
        expected_answer="x",
        prompt_question="?",
        response="x",
        score=Score(value=1.0),
        usage=None,
        cost_usd=None,
        duration_seconds=0.1,
        status="ok",
        error=None,
        timestamp_utc="2026-05-29T10:00:00+00:00",
        seed=None,
    )
    d = asdict(r)
    assert "context" not in d
    assert "rendered_context" not in d
