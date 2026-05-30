"""Result stores — append-only sinks for `TestResult` rows."""

from .base import ResultKey, ResultStore, key_for
from .jsonl import JsonlResultStore

__all__ = ["JsonlResultStore", "ResultKey", "ResultStore", "key_for"]
