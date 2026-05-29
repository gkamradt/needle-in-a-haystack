"""Integration tests for the `niah` CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_CONFIGS = REPO_ROOT / "tests" / "configs"


def _run_niah(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI as a subprocess so we exercise the real entry point."""
    return subprocess.run(
        [sys.executable, "-m", "needlehaystack.cli.main", *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_command_exits_zero_for_good_config() -> None:
    p = _run_niah("validate", str(TEST_CONFIGS / "uuid_tiny.yaml"))
    assert p.returncode == 0, p.stderr
    assert "ok:" in p.stdout


def test_validate_command_exits_nonzero_for_missing_file() -> None:
    p = _run_niah("validate", str(TEST_CONFIGS / "does_not_exist.yaml"))
    assert p.returncode != 0
    assert "invalid" in p.stderr or "not found" in p.stderr


def test_dry_run_does_not_call_provider_or_write_results(tmp_path: Path) -> None:
    # Rewrite store path so we can prove no file is created.
    out_path = tmp_path / "would-be-results.jsonl"
    cfg_text = (TEST_CONFIGS / "uuid_tiny.yaml").read_text()
    cfg = yaml.safe_load(cfg_text)
    cfg["store"]["path"] = str(out_path)
    cfg["model"] = str(REPO_ROOT / "tests/configs/fake.yaml")
    tmp_cfg = tmp_path / "uuid_tiny.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg))

    p = _run_niah("run", str(tmp_cfg), "--dry-run")
    assert p.returncode == 0, p.stderr
    payload = json.loads(p.stdout)
    assert payload["run_name"] == "uuid-tiny"
    assert payload["sweep_cells"] == 1
    assert not out_path.exists()


def test_run_writes_one_result_row(tmp_path: Path) -> None:
    out_path = tmp_path / "results.jsonl"
    cfg_text = (TEST_CONFIGS / "uuid_tiny.yaml").read_text()
    cfg = yaml.safe_load(cfg_text)
    cfg["store"]["path"] = str(out_path)
    cfg["model"] = str(REPO_ROOT / "tests/configs/fake.yaml")
    tmp_cfg = tmp_path / "uuid_tiny.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg))

    p = _run_niah("run", str(tmp_cfg))
    assert p.returncode == 0, p.stderr

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    # New schema fields all present.
    assert row["schema_version"] == 2
    assert row["model_id"] == "fake-canned"
    assert row["task_type"] == "uuid"
    assert row["expected_answer"]
    assert row["recipe"]["inserter"] == "single_depth"
    assert row["recipe"]["haystack"]["type"] == "text"
    assert len(row["recipe"]["needle_placements"]) == 1
    assert row["score"] is not None
    # FakeProvider returned a canned response that doesn't contain the UUID
    # → score 0.0 (the run itself still succeeded; this is a scorer outcome).
    assert row["score"]["value"] == 0.0
    assert row["status"] == "ok"


def test_help_subcommands_listed() -> None:
    p = _run_niah("--help")
    assert p.returncode == 0
    for cmd in ("run", "validate", "reconstruct"):
        assert cmd in p.stdout
