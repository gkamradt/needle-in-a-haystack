"""Append-only JSONL result store.

One row per line, UTF-8, newline-terminated. On construction the store
scans the existing file (if any) and builds an in-memory index of
already-written keys so `already_has` is O(1) per check. Subsequent
appends are serialized through an `asyncio.Lock` so concurrent runner
coroutines can't interleave a partial write.

Corrupted trailing lines (e.g. an interrupted previous run) are skipped
with a warning during index build — we never truncate or rewrite the
file ourselves.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
from pathlib import Path

from ..core.types import TestResult
from .base import ResultKey, key_for

logger = logging.getLogger(__name__)


class JsonlResultStore:
    """JSONL result store with O(1) resume lookups."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()
        self._seen: set[ResultKey] = set()
        if self.path.exists():
            self._build_index()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- ResultStore protocol -----------------------------------------------

    async def append(self, result: TestResult) -> None:
        line = json.dumps(dataclasses.asdict(result), ensure_ascii=False)
        async with self._lock:
            # Append in a thread to avoid blocking the event loop on
            # large files / slow disks.
            await asyncio.to_thread(self._append_sync, line)
            self._seen.add(key_for(result))

    def already_has(self, key: ResultKey) -> bool:
        return key in self._seen

    # --- introspection ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._seen)

    # --- internal -----------------------------------------------------------

    def _append_sync(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

    def _build_index(self) -> None:
        """Read every line of the existing file and index its key.

        Tolerant of a corrupted trailing line: we log a warning and
        stop. Anything *before* the bad line is still indexed.
        """
        with open(self.path, encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning(
                        "JsonlResultStore: skipping malformed line %d in %s",
                        line_no,
                        self.path,
                    )
                    continue
                try:
                    key: ResultKey = (
                        obj["run_name"],
                        obj["model_id"],
                        int(obj["context_length"]),
                        float(obj["target_depth_percent"]),
                        obj.get("seed"),
                    )
                except (KeyError, TypeError, ValueError):
                    logger.warning(
                        "JsonlResultStore: line %d in %s missing key fields; skipping",
                        line_no,
                        self.path,
                    )
                    continue
                self._seen.add(key)
