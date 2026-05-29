"""Single shared tokenizer for all length math across the codebase.

Per the refactor plan we use one generic `tiktoken` encoding
(`cl100k_base`) for every provider. Token accounting is approximate; the
goal is sweep reproducibility, not exact-to-the-byte matching of a given
model's vocabulary. Providers that disagree on token counts will still see
the same context bytes — only the *labelled* `context_length` may differ
slightly from what their internal tokenizer would report.
"""

from __future__ import annotations

import tiktoken

_ENCODING_NAME = "cl100k_base"
_encoding = tiktoken.get_encoding(_ENCODING_NAME)


def encoding_name() -> str:
    """Return the name of the encoding everything in the project uses."""
    return _ENCODING_NAME


def encode(text: str) -> list[int]:
    """Encode `text` to token ids using the shared encoding."""
    return _encoding.encode(text)


def decode(tokens: list[int]) -> str:
    """Decode `tokens` back to a string using the shared encoding."""
    return _encoding.decode(tokens)


def count(text: str) -> int:
    """Return the number of tokens in `text` under the shared encoding."""
    return len(_encoding.encode(text))
