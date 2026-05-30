"""Needle generators — produce the snippets a task inserts into a haystack."""

from .base import NeedleGenerator
from .static import StaticNeedle
from .uuid_chain import UuidChain
from .uuid_single import UuidSingle

__all__ = ["NeedleGenerator", "StaticNeedle", "UuidChain", "UuidSingle"]
