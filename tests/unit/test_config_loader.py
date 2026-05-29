"""Unit tests for the config loader + factories."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from needlehaystack.config.loader import (
    ConfigError,
    build_haystack,
    build_provider,
    build_store,
    build_sweep,
    build_task,
    load_model,
    load_run,
)
from needlehaystack.config.schema import HaystackSpec, ModelConfig, StoreSpec, TaskSpec
from needlehaystack.haystacks.files import FilesHaystack
from needlehaystack.haystacks.text import RepeatingTextHaystack
from needlehaystack.providers.fake import FakeProvider
from needlehaystack.stores.jsonl import JsonlResultStore
from needlehaystack.tasks.uuid_chain import UuidChainTask
from needlehaystack.tasks.uuid_needle import UuidNeedleTask

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
RUNS_DIR = CONFIGS_DIR / "runs"
MODELS_DIR = CONFIGS_DIR / "models"
TEST_CONFIGS = REPO_ROOT / "tests" / "configs"


# ---------- Example YAMLs parse --------------------------------------------------


@pytest.mark.parametrize("path", sorted(RUNS_DIR.glob("*.yaml")))
def test_every_example_run_yaml_parses(path: Path) -> None:
    cfg = load_run(path)
    assert cfg.run_name
    assert cfg.task.type
    assert cfg.haystack.type


@pytest.mark.parametrize("path", sorted(MODELS_DIR.glob("*.yaml")))
def test_every_example_model_yaml_parses(path: Path) -> None:
    cfg = load_model(path)
    assert cfg.id
    assert cfg.runtime.sdk
    assert cfg.runtime.api
    assert cfg.request.model


# ---------- Model id resolution -------------------------------------------------


def test_bare_model_id_resolves_against_search_dirs() -> None:
    cfg = load_model("fake", search_dirs=[MODELS_DIR])
    assert cfg.id == "fake"


def test_unknown_model_id_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="could not resolve"):
        load_model("not-a-real-model", search_dirs=[MODELS_DIR])


def test_model_path_takes_precedence_over_id_resolution(tmp_path: Path) -> None:
    direct = tmp_path / "direct.yaml"
    direct.write_text(
        yaml.safe_dump(
            {
                "id": "direct",
                "runtime": {"sdk": "fake", "api": "fake"},
                "request": {"model": "fake"},
            }
        )
    )
    cfg = load_model(direct, search_dirs=[MODELS_DIR])
    assert cfg.id == "direct"


# ---------- Error reporting -----------------------------------------------------


def test_missing_required_field_raises_friendly_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"task": {"type": "single"}}))  # missing run_name etc.
    with pytest.raises(ConfigError) as e:
        load_run(bad)
    msg = str(e.value)
    assert "invalid run config" in msg
    assert "run_name" in msg


def test_unknown_task_type_raises_with_registered_names() -> None:
    with pytest.raises(ConfigError) as e:
        build_task(TaskSpec(type="not_a_task"))
    assert "registered" in str(e.value)
    assert "single" in str(e.value)


def test_unknown_haystack_type_rejected_at_parse_time() -> None:
    # `type` is `Literal["files","text"]` → pydantic rejects.
    with pytest.raises(ValidationError):
        HaystackSpec.model_validate({"type": "rss"})


def test_unknown_provider_runtime_raises() -> None:
    cfg = ModelConfig.model_validate(
        {
            "id": "x",
            "runtime": {"sdk": "nonsense-sdk", "api": "nope"},
            "request": {"model": "y"},
        }
    )
    with pytest.raises(KeyError, match="no provider registered"):
        build_provider(cfg)


# ---------- request: extras round-trip -----------------------------------------


def test_request_extras_pass_through() -> None:
    cfg = ModelConfig.model_validate(
        {
            "id": "opus",
            "runtime": {"sdk": "anthropic-python", "api": "messages"},
            "client": {"api_key_env": "ANTHROPIC_API_KEY"},
            "request": {
                "model": "claude-opus-4",
                "max_tokens": 100,
                "stream": True,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "medium"},
            },
            "pricing": {"input": 5.0, "output": 25.0},
        }
    )
    dumped = cfg.request.model_dump()
    assert dumped["thinking"] == {"type": "adaptive"}
    assert dumped["output_config"] == {"effort": "medium"}
    assert dumped["stream"] is True


def test_pricing_optional() -> None:
    cfg = load_model("fake", search_dirs=[MODELS_DIR])
    assert cfg.pricing is None


# ---------- Factories build the right objects ----------------------------------


def test_build_task_returns_registered_class_instance() -> None:
    spec = TaskSpec(type="uuid_chain", chain_length=4)
    task = build_task(spec)
    assert isinstance(task, UuidChainTask)
    assert task.chain_length == 4


def test_build_task_with_no_kwargs_uses_defaults() -> None:
    task = build_task(TaskSpec(type="uuid"))
    assert isinstance(task, UuidNeedleTask)


def test_build_haystack_files_and_text() -> None:
    f = build_haystack(HaystackSpec.model_validate({"type": "files", "path": "PaulGrahamEssays"}))
    assert isinstance(f, FilesHaystack)
    t = build_haystack(HaystackSpec.model_validate({"type": "text", "text": "hi"}))
    assert isinstance(t, RepeatingTextHaystack)


def test_build_store_jsonl(tmp_path: Path) -> None:
    store = build_store(StoreSpec(type="jsonl", path=str(tmp_path / "out.jsonl")))
    assert isinstance(store, JsonlResultStore)


def test_build_provider_fake_via_full_pipeline() -> None:
    cfg = load_model("fake", search_dirs=[MODELS_DIR])
    provider = build_provider(cfg)
    assert isinstance(provider, FakeProvider)
    assert provider.id == "fake"


def test_build_sweep_from_range_and_list() -> None:
    cfg = load_run(TEST_CONFIGS / "uuid_tiny.yaml")
    sw = build_sweep(cfg.sweep)
    assert sw.lengths == [500]
    assert sw.depths == [50.0]
    assert sw.seeds == [42]


def test_build_sweep_range_dispatches_to_scale() -> None:
    cfg = load_run(RUNS_DIR / "single_needle.example.yaml")
    sw = build_sweep(cfg.sweep)
    # context_lengths is a 15-step linear ramp from 1000 to 32000.
    assert len(sw.lengths) == 15
    assert sw.lengths[0] == 1000
    assert sw.lengths[-1] == 32000
    # depth_percents is a 11-step sigmoid ramp 0 → 100.
    assert len(sw.depths) == 11
    assert sw.depths[0] == pytest.approx(0.0)
    assert sw.depths[-1] == pytest.approx(100.0)
