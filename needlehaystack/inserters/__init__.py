"""Inserters — splice needle texts into the haystack at the right depths."""

from .base import Inserter, InsertResult
from .even_spread import EvenSpreadInserter
from .explicit import ExplicitDepthsInserter
from .single_depth import SingleDepthInserter

__all__ = [
    "EvenSpreadInserter",
    "ExplicitDepthsInserter",
    "InsertResult",
    "Inserter",
    "SingleDepthInserter",
]
