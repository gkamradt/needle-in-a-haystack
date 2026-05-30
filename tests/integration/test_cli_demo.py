"""Integration tests for the `niah demo` zero-config command.

`niah demo --fake` is the absolute easiest path for a new user: no API
keys, no config files, just verifies the install end-to-end. These
tests pin that contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_niah(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "needlehaystack.cli.main", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_demo_fake_writes_jsonl_with_one_row_per_cell(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    p = _run_niah("demo", "--fake", "--out", str(out), cwd=tmp_path)

    assert p.returncode == 0, p.stderr
    assert out.exists(), "demo did not write the results file"

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    # 2 lengths × 3 depths = 6 cells; that's the contract the README leads with.
    assert len(rows) == 6, f"expected 6 rows, got {len(rows)}: {rows}"

    # Canned fake response contains the SingleNeedleTask expected answer,
    # so every cell should be a perfect match.
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["score"]["value"] == 1.0 for row in rows)


def test_demo_real_provider_fails_friendly_without_api_key(tmp_path: Path, monkeypatch) -> None:
    """Without a key, `niah demo` should tell the user what to do — not
    bubble up a 401 from the SDK halfway through the sweep."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Also ensure no `.env` in tmp_path accidentally loads one.

    p = _run_niah("demo", "--out", str(tmp_path / "r.jsonl"), cwd=tmp_path)

    assert p.returncode != 0
    assert "OPENAI_API_KEY" in p.stderr
    assert "--fake" in p.stderr
