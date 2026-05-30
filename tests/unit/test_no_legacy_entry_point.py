"""Regression test: the v1 `needlehaystack.run_test` entry point must
not be re-introduced.

v2 is a clean break — the only console script is `niah`. If anyone adds
`needlehaystack.run_test` back to `[project.scripts]` this test fails
and points them at the migration story.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_pyproject_loadable() -> None:
    assert PYPROJECT.is_file()
    with open(PYPROJECT, "rb") as f:
        tomllib.load(f)


def test_only_niah_is_declared_as_console_script() -> None:
    with open(PYPROJECT, "rb") as f:
        pyproject = tomllib.load(f)
    scripts = pyproject.get("project", {}).get("scripts", {})
    assert "niah" in scripts, "`niah` console script missing from [project.scripts]"


def test_legacy_run_test_entry_point_is_absent() -> None:
    with open(PYPROJECT, "rb") as f:
        pyproject = tomllib.load(f)
    scripts = pyproject.get("project", {}).get("scripts", {})
    assert "needlehaystack.run_test" not in scripts, (
        "v1 `needlehaystack.run_test` entry point reintroduced; "
        "v2.0.0 is a clean break — see README migration section."
    )
