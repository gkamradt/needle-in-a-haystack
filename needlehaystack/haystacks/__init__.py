"""Haystack sources — the background text needles get inserted into."""

from .base import HaystackSource
from .files import FilesHaystack
from .text import RepeatingTextHaystack

__all__ = ["FilesHaystack", "HaystackSource", "RepeatingTextHaystack"]
