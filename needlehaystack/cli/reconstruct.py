"""Rebuild the exact context a model saw, from a stored result row.

The runner stores only a `Recipe` (haystack descriptor + ordered
placements + final token count) on each row. This module replays it:
load the haystack, truncate to `context_length` tokens, then splice
each needle text back at its stored `insertion_token_index`. Because
the placements were captured in the post-insertion token stream and
we splice in ascending index order, the result is byte-identical to
what the model originally received.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config.loader import build_haystack
from ..config.schema import HaystackSpec
from ..core import tokens


def reconstruct_row(row: dict[str, Any]) -> str:
    """Return the rendered context for one result-row dict."""
    recipe = row["recipe"]
    context_length = int(row["context_length"])

    haystack = build_haystack(HaystackSpec.model_validate(recipe["haystack"]))
    raw = haystack.load(min_tokens=context_length)
    ctx_tokens = tokens.encode(raw)[:context_length]

    # Splice in ascending insertion_token_index. The stored index is
    # already in the post-insertion stream, so this exactly reverses
    # what the inserter did.
    placements = sorted(
        recipe["needle_placements"],
        key=lambda p: int(p["insertion_token_index"]),
    )
    new_tokens = list(ctx_tokens)
    for p in placements:
        idx = int(p["insertion_token_index"])
        needle_tokens = tokens.encode(p["text"])
        new_tokens = new_tokens[:idx] + needle_tokens + new_tokens[idx:]

    expected = int(recipe["final_context_token_count"])
    if len(new_tokens) != expected:
        # Surface mismatch loudly — usually means the haystack source
        # changed (e.g. file added/removed) since the run.
        raise RuntimeError(
            f"reconstruction produced {len(new_tokens)} tokens; "
            f"recipe says {expected}. The haystack source may have changed."
        )

    return tokens.decode(new_tokens)


def reconstruct_from_jsonl(path: str | Path, row_index: int) -> str:
    """Read `path`, pick row `row_index` (0-based), and reconstruct it."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == row_index:
                return reconstruct_row(json.loads(line))
    raise IndexError(f"row {row_index} not found in {p}")
