"""`niah` — the v2 command-line interface.

Subcommands:

- `niah run <run.yaml>` — load configs, build all the pieces, run the
  sweep, write JSONL.
- `niah validate <run.yaml>` — parse + resolve the model config + sanity
  check without running anything.
- `niah reconstruct <results.jsonl> --row N [--out file]` — replay the
  recipe on row N and print (or write) the exact context the model saw.

The CLI is deliberately thin. All real work lives in `config/loader.py`,
`core/runner.py`, and `cli/reconstruct.py` so the same code paths drive
both `niah` and library users who skip the CLI entirely.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from ..config.loader import (
    ConfigError,
    build_haystack,
    build_provider,
    build_store,
    build_sweep,
    build_task,
    load_model,
    load_run,
)
from ..config.schema import ModelConfig, RunConfig
from ..core.runner import Pricing, Runner
from .reconstruct import reconstruct_from_jsonl

# Default place to look up bare model ids in a run config.
DEFAULT_MODEL_DIRS = [Path("configs/models")]

app = typer.Typer(
    name="niah",
    help="Needle In A Haystack — pressure-test LLM long-context retrieval.",
    add_completion=False,
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# `niah run`
# ---------------------------------------------------------------------------


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="Path to a run YAML.")],
    model_dir: Annotated[
        list[Path] | None,
        typer.Option(
            "--model-dir",
            help="Extra directory to search for model YAMLs (repeatable).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate, resolve model, and print the merged plan — don't call the model.",
        ),
    ] = False,
) -> None:
    """Run a sweep described by `config` against the referenced model."""
    try:
        run_cfg, model_cfg = _load_and_resolve(config, model_dir)
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e

    if dry_run:
        _print_plan(run_cfg, model_cfg)
        return

    runner = _build_runner(run_cfg, model_cfg)

    def _printer(result: object) -> None:
        # Conservative pretty-printer: one line per cell.
        r = result  # noqa: F841 — for clarity; we access attrs below
        typer.echo(
            f"[{getattr(r, 'status', '?')}] "
            f"ctx={getattr(r, 'context_length', '?')} "
            f"depth={getattr(r, 'target_depth_percent', '?'):.1f}% "
            f"score={_score_str(r)} "
            f"dur={getattr(r, 'duration_seconds', 0.0):.2f}s"
        )

    runner.on_result = _printer
    results = asyncio.run(runner.run())
    typer.echo(f"done: {len(results)} new cells written to {run_cfg.store.path}")


# ---------------------------------------------------------------------------
# `niah validate`
# ---------------------------------------------------------------------------


@app.command()
def validate(
    config: Annotated[Path, typer.Argument(help="Path to a run YAML.")],
    model_dir: Annotated[
        list[Path] | None,
        typer.Option("--model-dir", help="Extra directory to search for model YAMLs."),
    ] = None,
) -> None:
    """Parse + validate `config` and its referenced model. No model calls."""
    try:
        run_cfg, model_cfg = _load_and_resolve(config, model_dir)
        # Touch all the builders so registry-related errors surface here.
        build_task(run_cfg.task)
        build_haystack(run_cfg.haystack)
        build_provider(model_cfg)
        build_sweep(run_cfg.sweep)
    except ConfigError as e:
        typer.echo(f"invalid: {e}", err=True)
        raise typer.Exit(code=2) from e

    typer.echo(f"ok: {config} (model: {model_cfg.id})")


# ---------------------------------------------------------------------------
# `niah reconstruct`
# ---------------------------------------------------------------------------


@app.command()
def reconstruct(
    results: Annotated[Path, typer.Argument(help="Path to a results JSONL.")],
    row: Annotated[int, typer.Option("--row", help="Row index (0-based).")] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write to file instead of stdout."),
    ] = None,
) -> None:
    """Replay the recipe on row N and emit the exact context the model saw."""
    try:
        text = reconstruct_from_jsonl(results, row)
    except (IndexError, FileNotFoundError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e

    if out is None:
        # Direct stdout write to skip typer's line-wrap heuristics for
        # potentially-very-large bodies.
        sys.stdout.write(text)
    else:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {len(text):,} chars to {out}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_and_resolve(config: Path, model_dir: list[Path] | None) -> tuple[RunConfig, ModelConfig]:
    run_cfg = load_run(config)
    search_dirs = list(DEFAULT_MODEL_DIRS)
    if model_dir:
        search_dirs.extend(model_dir)
    model_cfg = load_model(run_cfg.model, search_dirs=search_dirs)
    return run_cfg, model_cfg


def _build_runner(run_cfg: RunConfig, model_cfg: ModelConfig) -> Runner:
    task = build_task(run_cfg.task)
    haystack = build_haystack(run_cfg.haystack)
    provider = build_provider(model_cfg)
    store = build_store(run_cfg.store)
    sweep = build_sweep(run_cfg.sweep)
    pricing = None
    if model_cfg.pricing is not None:
        pricing = Pricing(input=model_cfg.pricing.input, output=model_cfg.pricing.output)
    return Runner(
        task=task,
        provider=provider,
        haystack=haystack,
        sweep=sweep,
        store=store,
        run_name=run_cfg.run_name,
        pricing=pricing,
        concurrency=run_cfg.runner.concurrency,
        retries=run_cfg.runner.retries,
        sleep_between_seconds=run_cfg.runner.sleep_between_seconds,
        resume=run_cfg.runner.resume,
    )


def _print_plan(run_cfg: RunConfig, model_cfg: ModelConfig) -> None:
    plan = {
        "run_name": run_cfg.run_name,
        "model_id": model_cfg.id,
        "runtime": model_cfg.runtime.model_dump(),
        "task": run_cfg.task.model_dump(),
        "haystack": run_cfg.haystack.model_dump(),
        "sweep_cells": len(build_sweep(run_cfg.sweep).cells()),
        "store_path": run_cfg.store.path,
        "concurrency": run_cfg.runner.concurrency,
    }
    typer.echo(json.dumps(plan, indent=2, default=str))


def _score_str(result: object) -> str:
    score = getattr(result, "score", None)
    if score is None:
        return "—"
    return f"{score.value:.2f}"


if __name__ == "__main__":  # pragma: no cover
    app()
