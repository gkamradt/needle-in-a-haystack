"""The async sweep runner.

This is the only "orchestration" piece in the project. All the actual
work (loading a haystack, generating a needle, inserting it, calling a
model, scoring) is delegated to the building-block layers; the runner
just walks the sweep grid and stitches them together.

Responsibilities:

- Walk `sweep.cells()` with bounded concurrency.
- Skip cells already in the store when `resume=True`.
- Retry provider errors up to `retries` extra attempts.
- Build the `Recipe` for each row from the task's returned placements +
  the haystack descriptor. Tasks themselves don't know recipes exist.
- Compute `cost_usd` from the provider's `Usage` and the optional
  pricing block (USD per million tokens).
- Hand each completed `TestResult` to the store and (optionally) a
  caller-supplied pretty-printer callback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..haystacks.base import HaystackSource
from ..providers.base import ModelProvider
from ..stores.base import ResultStore, key_for
from ..tasks.base import Task
from . import tokens
from .sweep import Sweep
from .types import Recipe, TestResult, Usage

logger = logging.getLogger(__name__)


# Pricing is USD per million tokens (industry standard for LLM APIs).
TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass(slots=True)
class Pricing:
    """Per-million-token prices used to compute `cost_usd` on result rows."""

    input: float
    output: float


@dataclass(slots=True)
class Runner:
    """Drives a sweep of `(length × depth × seed)` cells against one provider."""

    task: Task
    provider: ModelProvider
    haystack: HaystackSource
    sweep: Sweep
    store: ResultStore

    run_name: str = "run"
    pricing: Pricing | None = None
    concurrency: int = 1
    retries: int = 0
    sleep_between_seconds: float = 0.0
    resume: bool = True
    # Optional pretty-printer fired once per completed row (ok or error).
    on_result: Callable[[TestResult], None] | None = None

    # Per-length context cache so we don't re-read the haystack for every
    # depth in the same length tier. Populated lazily inside `run()`.
    _ctx_cache: dict[int, list[int]] = field(default_factory=dict, init=False, repr=False)

    async def run(self) -> list[TestResult]:
        sem = asyncio.Semaphore(self.concurrency)

        async def run_cell(length: int, depth: float, seed: int | None) -> TestResult | None:
            async with sem:
                return await self._run_one_cell(length, depth, seed)

        coros = [run_cell(L, d, s) for L, d, s in self.sweep.cells()]
        all_results = await asyncio.gather(*coros)
        return [r for r in all_results if r is not None]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _run_one_cell(
        self,
        length: int,
        depth: float,
        seed: int | None,
    ) -> TestResult | None:
        key = (self.run_name, self.provider.id, length, depth, seed)
        if self.resume and self.store.already_has(key):
            return None

        ctx = self._load_context(length)
        needle = self.task.generate_needle(seed=seed)
        new_tokens, placements = self.task.insert(ctx, needle, depth)
        rendered = tokens.decode(new_tokens)
        question = self.task.question(needle)
        user_prompt = f"{rendered}\n\n{question}"

        started = time.monotonic()
        completion, error = await self._call_with_retries(user_prompt)
        duration = time.monotonic() - started

        recipe = Recipe(
            haystack=self.haystack.descriptor(),
            inserter=self.task.inserter_name,
            needle_placements=list(placements),
            final_context_token_count=len(new_tokens),
        )

        common: dict[str, Any] = {
            "schema_version": 2,
            "run_name": self.run_name,
            "model_id": self.provider.id,
            "model_request_name": self.provider.request_model,
            "task_type": self.task.name,
            "context_length": length,
            "target_depth_percent": depth,
            "recipe": recipe,
            "needle_metadata": dict(needle.metadata),
            "expected_answer": needle.expected_answer,
            "prompt_question": question,
            "duration_seconds": duration,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "seed": seed,
        }

        if completion is None:
            result = TestResult(
                response=None,
                score=None,
                usage=None,
                cost_usd=None,
                status="error",
                error=str(error) if error else "unknown error",
                **common,
            )
        else:
            score = self.task.score(completion.text, needle)
            cost = _compute_cost(completion.usage, self.pricing)
            result = TestResult(
                response=completion.text,
                score=score,
                usage=completion.usage,
                cost_usd=cost,
                status="ok",
                error=None,
                **common,
            )

        await self.store.append(result)
        # sanity: key match between what we'd skip and what we persisted
        assert key_for(result) == key
        if self.on_result is not None:
            self.on_result(result)
        return result

    async def _call_with_retries(self, user_prompt: str) -> tuple[Any | None, BaseException | None]:
        last_err: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                completion = await self.provider.complete(system="", user=user_prompt)
                return completion, None
            except Exception as e:
                last_err = e
                logger.warning(
                    "provider call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.retries + 1,
                    e,
                )
                if attempt < self.retries and self.sleep_between_seconds > 0:
                    await asyncio.sleep(self.sleep_between_seconds)
        return None, last_err

    def _load_context(self, length: int) -> list[int]:
        cached = self._ctx_cache.get(length)
        if cached is not None:
            return cached
        raw = self.haystack.load(min_tokens=length)
        toks = tokens.encode(raw)[:length]
        self._ctx_cache[length] = toks
        return toks


def _compute_cost(usage: Usage | None, pricing: Pricing | None) -> float | None:
    if usage is None or pricing is None:
        return None
    return (
        usage.input_tokens * pricing.input + usage.output_tokens * pricing.output
    ) / TOKENS_PER_PRICE_UNIT
