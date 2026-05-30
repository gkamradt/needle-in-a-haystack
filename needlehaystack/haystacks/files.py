"""Haystack made from a directory of plain-text files.

Cycles through every `.txt` file in the directory, concatenating contents
until the required token count is reached. The Paul Graham essays
directory bundled with the package is the canonical default.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import ClassVar

from ..core import tokens


@dataclass(slots=True)
class FilesHaystack:
    """Reads `.txt` files from `path`, cycling until `min_tokens` reached.

    `path` may be:

    - absolute (e.g. `/Users/x/my-essays`)
    - relative to CWD
    - the name of a directory bundled inside the package (e.g.
      `"PaulGrahamEssays"`) — resolved against the package root.
    """

    name: ClassVar[str] = "files"

    path: str = "PaulGrahamEssays"

    def load(self, min_tokens: int) -> str:
        files = self._discover_files()
        if not files:
            raise FileNotFoundError(
                f"FilesHaystack: no .txt files found under {self._resolved_path()!r}"
            )

        buf: list[str] = []
        running_tokens = 0
        # Cycle through the file list as many times as needed. Most sweeps
        # ask for sizes well within one pass through PaulGrahamEssays, but
        # very long contexts may require multiple cycles.
        while running_tokens < min_tokens:
            for file in files:
                with open(file, encoding="utf-8") as f:
                    chunk = f.read()
                buf.append(chunk)
                running_tokens += tokens.count(chunk)
                if running_tokens >= min_tokens:
                    break

        return "".join(buf)

    def descriptor(self) -> dict[str, str]:
        return {"type": self.name, "path": self.path}

    # --- internal -----------------------------------------------------------

    def _resolved_path(self) -> str:
        if os.path.isabs(self.path) or os.path.isdir(self.path):
            return self.path
        # Treat as a directory bundled inside the package.
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(package_root, self.path)

    def _discover_files(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self._resolved_path(), "*.txt")))
