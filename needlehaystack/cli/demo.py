"""`niah demo` — zero-config end-to-end sweep.

Lets a new user verify the install with one command:

    pip install needlehaystack
    echo "OPENAI_API_KEY=sk-..." > .env
    niah demo

Runs a small sweep (2 context lengths × 3 depths = 6 cells) against the
default model (`gpt-4o-mini`), using the bundled Paul Graham essays as
the haystack and the default single-fact needle. Writes
`./results.jsonl`. The whole thing costs about a penny.

`--provider {openai,anthropic,cohere,fake}` switches model. `--fake`
uses the in-process FakeProvider so you can verify the install with
no API key at all.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated, Literal

import typer

from ..config.schema import ModelConfig
from ..core.runner import Pricing, Runner
from ..core.sweep import Sweep
from ..haystacks.files import FilesHaystack
from ..providers.registry import build_provider
from ..stores.jsonl import JsonlResultStore
from ..tasks.single_needle import SingleNeedleTask

# 2 × 3 = 6 cells. Small enough that a real-provider run is ~$0.01,
# big enough that the output table is interesting to look at.
DEMO_LENGTHS = [2000, 8000]
DEMO_DEPTHS = [10.0, 50.0, 90.0]
DEMO_OUTPUT_PATH = "results.jsonl"

ProviderName = Literal["openai", "anthropic", "cohere", "fake"]


def demo(
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Which provider to demo against. One of: openai, anthropic, cohere, fake.",
        ),
    ] = "openai",
    fake: Annotated[
        bool,
        typer.Option(
            "--fake",
            help="Shortcut for --provider fake. Runs end-to-end with no API key.",
        ),
    ] = False,
    out: Annotated[
        Path,
        typer.Option("--out", help="Where to write the JSONL results."),
    ] = Path(DEMO_OUTPUT_PATH),
) -> None:
    """Run a small zero-config sweep to verify the install end-to-end."""
    chosen = "fake" if fake else provider
    if chosen not in ("openai", "anthropic", "cohere", "fake"):
        typer.echo(
            f"error: --provider must be one of openai, anthropic, cohere, fake; got {chosen!r}",
            err=True,
        )
        raise typer.Exit(code=2)

    model_cfg = _demo_model_config(chosen)  # type: ignore[arg-type]

    # Friendly pre-flight: if the user picked a real provider but hasn't
    # set the key, tell them what to do instead of letting the SDK error
    # bubble up mid-run.
    if chosen != "fake":
        key_env = model_cfg.client.api_key_env
        if key_env and not os.environ.get(key_env):
            typer.echo(
                f"error: ${key_env} is not set. Either:\n"
                f"  echo '{key_env}=...' > .env       (then re-run)\n"
                f"  niah demo --fake                  (no API key needed)",
                err=True,
            )
            raise typer.Exit(code=2)

    task = SingleNeedleTask()
    haystack = FilesHaystack(path="PaulGrahamEssays")
    sweep = Sweep(lengths=list(DEMO_LENGTHS), depths=list(DEMO_DEPTHS))
    store = JsonlResultStore(out)

    pricing = None
    if model_cfg.pricing is not None:
        pricing = Pricing(input=model_cfg.pricing.input, output=model_cfg.pricing.output)

    runner = Runner(
        task=task,
        provider=build_provider(model_cfg),
        haystack=haystack,
        sweep=sweep,
        store=store,
        run_name="demo",
        pricing=pricing,
        concurrency=3,
        retries=1,
    )

    typer.echo(
        f"running demo: model={model_cfg.id}  "
        f"task=single-fact  haystack=PaulGrahamEssays  "
        f"cells={len(sweep.cells())}  out={out}"
    )

    def _line(r: object) -> None:
        score = getattr(r, "score", None)
        score_s = f"{score.value:.2f}" if score is not None else "—"
        typer.echo(
            f"[{getattr(r, 'status', '?')}] "
            f"ctx={getattr(r, 'context_length', '?')} "
            f"depth={getattr(r, 'target_depth_percent', 0.0):.0f}% "
            f"score={score_s} "
            f"dur={getattr(r, 'duration_seconds', 0.0):.2f}s"
        )

    runner.on_result = _line
    results = asyncio.run(runner.run())

    total_cost = sum((r.cost_usd or 0.0) for r in results)
    typer.echo(
        f"done: {len(results)} cells written to {out}"
        + (f" (cost: ${total_cost:.4f})" if total_cost else "")
    )
    typer.echo(f"next: niah reconstruct {out} --row 0   # see the exact context model saw")


def _demo_model_config(provider: ProviderName) -> ModelConfig:
    """Built-in model configs for the demo, one per supported provider."""
    if provider == "openai":
        return ModelConfig.model_validate(
            {
                "id": "openai-gpt-4o-mini",
                "runtime": {"sdk": "openai-python", "api": "chat_completions"},
                "client": {"api_key_env": "OPENAI_API_KEY"},
                "request": {
                    "model": "gpt-4o-mini",
                    "max_tokens": 256,
                    "temperature": 0.0,
                },
                "pricing": {"input": 0.15, "output": 0.60},
            }
        )
    if provider == "anthropic":
        return ModelConfig.model_validate(
            {
                "id": "anthropic-haiku",
                "runtime": {"sdk": "anthropic-python", "api": "messages"},
                "client": {"api_key_env": "ANTHROPIC_API_KEY"},
                "request": {"model": "claude-3-5-haiku-latest", "max_tokens": 256},
                "pricing": {"input": 0.80, "output": 4.00},
            }
        )
    if provider == "cohere":
        return ModelConfig.model_validate(
            {
                "id": "cohere-command-r",
                "runtime": {"sdk": "cohere-python", "api": "chat"},
                "client": {"api_key_env": "COHERE_API_KEY"},
                "request": {"model": "command-r", "temperature": 0.0},
                "pricing": {"input": 0.50, "output": 1.50},
            }
        )
    # fake
    # The canned response contains the SingleNeedleTask default
    # `expected_answer` substring so the demo prints 1.00 scores instead
    # of confusing 0.00s.
    return ModelConfig.model_validate(
        {
            "id": "fake-demo",
            "runtime": {"sdk": "fake", "api": "fake"},
            "client": {},
            "request": {
                "model": "fake-model",
                "mode": "canned",
                "response_text": (
                    "The best thing to do in San Francisco is "
                    "eat a sandwich and sit in Dolores Park on a sunny day."
                ),
            },
        }
    )
