"""Deterministic in-process provider used by every test.

`FakeProvider` is the v2 testing workhorse. It never touches the network,
returns whatever you configure it to return, and records every call so
tests can assert on call counts, args, and ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..core.types import Usage
from .base import Completion


@dataclass(slots=True)
class FakeCall:
    """One captured `complete` invocation."""

    system: str
    user: str


@dataclass
class FakeProvider:
    """Configurable, deterministic ModelProvider for tests.

    Modes:

    - `mode="canned"` (default): always returns `response_text`.
    - `mode="echo_answer"`: returns the substring sandwiched between the
      delimiters configured on the test prompt — see `make_echo_prompt`
      below. The fake doesn't know the expected answer otherwise, so the
      test must encode it into the prompt itself.
    - `mode="fail_n_times_then_canned"`: the first `fail_until` calls
      raise `RuntimeError(fail_message)`; subsequent calls return
      `response_text`. Used to exercise retry logic in the runner.
    - `mode="always_fail"`: every call raises `RuntimeError(fail_message)`.

    `usage_per_call`, when set, is returned verbatim in the `Completion`.
    """

    id: str = "fake"
    request_model: str = "fake-model"

    mode: Literal[
        "canned",
        "echo_answer",
        "fail_n_times_then_canned",
        "always_fail",
    ] = "canned"

    response_text: str = "ok"
    fail_until: int = 0
    fail_message: str = "synthetic failure"
    usage_per_call: Usage | None = None

    calls: list[FakeCall] = field(default_factory=list)
    _failures_emitted: int = 0

    # Sentinel used by `make_echo_prompt` and detected here so tests can
    # round-trip an expected answer through a real-looking prompt without
    # the fake needing out-of-band knowledge.
    ECHO_OPEN = "<<ECHO:"
    ECHO_CLOSE = ":ECHO>>"

    async def complete(self, system: str, user: str) -> Completion:
        self.calls.append(FakeCall(system=system, user=user))

        if self.mode == "always_fail":
            raise RuntimeError(self.fail_message)

        if self.mode == "fail_n_times_then_canned":
            if self._failures_emitted < self.fail_until:
                self._failures_emitted += 1
                raise RuntimeError(self.fail_message)
            return Completion(text=self.response_text, usage=self.usage_per_call)

        if self.mode == "echo_answer":
            answer = _extract_echo(user) or _extract_echo(system)
            if answer is None:
                raise ValueError(
                    "FakeProvider(mode='echo_answer') needs the prompt to contain an "
                    f"{self.ECHO_OPEN}...{self.ECHO_CLOSE} block; got none."
                )
            return Completion(text=answer, usage=self.usage_per_call)

        # mode == "canned"
        return Completion(text=self.response_text, usage=self.usage_per_call)


def make_echo_prompt(answer: str) -> str:
    """Wrap `answer` in the sentinel block `FakeProvider` echoes back.

    Tests use this to construct a prompt that — when fed to
    `FakeProvider(mode='echo_answer')` — round-trips the expected answer.
    """
    return f"{FakeProvider.ECHO_OPEN}{answer}{FakeProvider.ECHO_CLOSE}"


def _extract_echo(text: str) -> str | None:
    open_at = text.find(FakeProvider.ECHO_OPEN)
    if open_at == -1:
        return None
    start = open_at + len(FakeProvider.ECHO_OPEN)
    close_at = text.find(FakeProvider.ECHO_CLOSE, start)
    if close_at == -1:
        return None
    return text[start:close_at]
