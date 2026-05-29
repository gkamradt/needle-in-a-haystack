# Contributing

Thanks for considering a contribution to Needle In A Haystack.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and
package management. You don't need to install Python yourself — uv will
fetch the right version.

```bash
# one-time install of uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone & sync (creates .venv automatically with Python 3.12 + all deps)
git clone https://github.com/gkamradt/needle-in-a-haystack.git
cd needle-in-a-haystack
uv sync --all-extras
```

## Day-to-day commands

```bash
uv run pytest                  # run the test suite
uv run ruff check .            # lint
uv run ruff format .           # format
uv run ruff format --check .   # format check (CI)
uv run mypy needlehaystack     # type check
```

All of these must pass before a PR can be merged — CI runs the same
commands. Tests must not require network access; provider calls go
through `FakeProvider` (see `needlehaystack/providers/fake.py`).

## Project layout

The package is being refactored from v1 to v2. The current structure is
documented in [docs/](./docs/) once Phase 8 lands; until then, see the
phased plan you're working under for the current target shape.

## Branch & PR conventions

- Branch off `main`.
- Keep PRs scoped to a single phase / module — easier to review.
- Don't bundle drive-by refactors with feature work.

## Reporting issues

Open a [GitHub issue](https://github.com/gkamradt/needle-in-a-haystack/issues)
with a reproducible example. For bugs in a sweep, include the YAML config
plus the JSONL row(s) that demonstrate the problem.
