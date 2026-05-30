# Needle In A Haystack

> Pressure-test LLM long-context retrieval. Now in v2.

`niah` runs a sweep of `(context length × needle depth)` cells against any
configured model, scores each response, and writes one result row per cell to
a JSONL file. Built-in tasks include single-fact lookup, multi-fact recall,
single-UUID retrieval, and **UUID-chain hops** for testing multi-step
reasoning over long contexts.

Supported providers out of the box: **OpenAI**, **Anthropic**, **Cohere**.
Adding more is a small plugin.

---

## Quick start

```bash
pip install needlehaystack
niah demo --fake          # no API key, ~1s, proves the install works
```

That runs a 2 × 3 sweep (2 context lengths × 3 depths = 6 cells) against an
in-process fake model and writes `results.jsonl`. Inspect the exact context any
cell saw:

```bash
niah reconstruct results.jsonl --row 0
```

### Demo against a real model

```bash
echo "OPENAI_API_KEY=sk-..." > .env       # niah auto-loads .env
niah demo                                 # default: gpt-4o-mini, ~$0.01
```

Or pick another provider:

```bash
niah demo --provider anthropic            # needs ANTHROPIC_API_KEY
niah demo --provider cohere               # needs COHERE_API_KEY
```

That's it. The demo uses sensible defaults (gpt-4o-mini, the bundled Paul
Graham essays haystack, a single-fact needle, 6 cells) so you can see real
output before learning anything about config files.

---

## Custom runs

Once the demo works, drop the `demo` command and drive your own sweep with two
small YAML files. You point `niah` at **one run config** that references **one
model config**. Full examples live in [`configs/`](./configs).

```bash
niah validate my-run.yaml         # parse + resolve, no model calls
niah run      my-run.yaml         # actually run the sweep, append to JSONL
```

### Run config (`configs/runs/uuid_chain.example.yaml`)

```yaml
run_name: "uuid-chain-opus"
model: "anthropic-opus-4-medium"   # resolved against configs/models/

task:
  type: "uuid_chain"
  chain_length: 5

haystack:
  type: "files"
  path: "PaulGrahamEssays"

sweep:
  context_lengths: {min: 2000, max: 32000, num: 8,  scale: "linear"}
  depth_percents:  {min: 0,    max: 100,   num: 11, scale: "sigmoid"}
  seeds: [1, 2, 3]

runner:
  concurrency: 2
  retries: 2
  resume: true

store:
  type: "jsonl"
  path: "results/uuid-chain-opus.jsonl"
```

### Model config (`configs/models/anthropic-opus-4-medium.yaml`)

```yaml
id: "anthropic-opus-4-medium"
runtime:
  sdk: "anthropic-python"
  api: "messages"
client:
  api_key_env: "ANTHROPIC_API_KEY"
request:
  model: "claude-opus-4"
  max_tokens: 120000
  thinking:
    type: "adaptive"
  output_config:
    effort: "medium"
pricing:
  input: 5.00      # USD per 1M input tokens
  output: 25.00    # USD per 1M output tokens
```

Anything under `request:` is forwarded verbatim to the SDK, so adding new
provider-specific knobs (`thinking`, `reasoning_effort`, `top_p`, …) doesn't
require a code change.

---

## Built-in tasks

| `task.type`    | What it does |
|----------------|-------------|
| `single`       | One fact placed at one depth; exact-match scored. |
| `multi`        | N facts spread evenly through the context; fractional score. |
| `uuid`         | One fresh UUID at one depth; model must repeat it. |
| `uuid_chain`   | Chain of `A → B → C → …` links spread through the context. The question asks "what is the value associated with A?" **without** revealing the chain structure — the model has to discover the hops on its own. |

Tasks are a small Protocol — see [`needlehaystack/tasks/`](./needlehaystack/tasks).
Adding your own is one file and a registry call; nothing in the runner needs to
change.

```python
from needlehaystack.tasks import register_task

class MyCustomTask:
    name = "my_task"
    inserter_name = "single_depth"
    def generate_needle(self, seed): ...
    def insert(self, ctx, needle, depth): ...
    def question(self, needle): ...
    def score(self, response, needle): ...

register_task("my_task", MyCustomTask)
```

Reference it from a run config with `task.type: "my_task"`.

---

## CLI

```text
niah run        <run.yaml>            run a sweep, append to JSONL
niah run        <run.yaml> --dry-run  validate, resolve model, print plan, exit
niah validate   <run.yaml>            parse + resolve without running
niah reconstruct <results.jsonl> --row N [--out file]
                                      rebuild the exact context shown to the model
```

`--model-dir DIR` (repeatable) adds extra search paths for bare model ids.

---

## Result rows & reconstruction

Each row in the JSONL is small (a few KB) regardless of context size. We don't
store the rendered 200k-token context per row — that would balloon a single
sweep into gigabytes. Instead each row carries a **recipe**:

```json
{
  "schema_version": 2,
  "run_name": "uuid-chain-opus",
  "model_id": "anthropic-opus-4-medium",
  "task_type": "uuid_chain",
  "context_length": 32000,
  "target_depth_percent": 50.0,
  "recipe": {
    "haystack": {"type": "files", "path": "PaulGrahamEssays"},
    "inserter": "even_spread",
    "needle_placements": [
      {"text": "abc... maps to def...", "insertion_token_index": 15876, "actual_depth_percent": 49.6},
      ...
    ],
    "final_context_token_count": 32082
  },
  "expected_answer": "the-final-uuid",
  "prompt_question": "What is the value associated with abc-...?",
  "response": "...",
  "score": {"value": 0.6, "details": {"hops_correct": 3, "chain_length": 5}},
  "usage": {"input_tokens": 32100, "output_tokens": 412},
  "cost_usd": 0.171,
  "duration_seconds": 12.4,
  "status": "ok",
  "seed": 1,
  "timestamp_utc": "2026-..."
}
```

`niah reconstruct` walks the recipe and produces a byte-identical string of
what the model actually saw, which is what you want when a result is
surprising and you want to read the prompt.

---

## Extending

- **New provider**: write a class satisfying `ModelProvider` and call
  `register_provider(sdk, api, factory)`. See
  [`needlehaystack/providers/openai.py`](./needlehaystack/providers/openai.py)
  as a reference.
- **New task**: as above; see
  [`needlehaystack/tasks/uuid_chain.py`](./needlehaystack/tasks/uuid_chain.py).
- **New haystack source**: implement `HaystackSource` (`load(min_tokens)` +
  `descriptor()`).
- **New scorer**: implement `Scorer` (`score(response, needle)`).

The system is intentionally a set of small Protocols connected by registries
so contributors never need to edit the runner.

---

## Develop / contribute

The repo ships an `examples/` worth of configs under [`configs/`](./configs)
and a FakeProvider so the entire pipeline runs end-to-end with no API keys.

```bash
git clone https://github.com/gkamradt/needle-in-a-haystack.git
cd needle-in-a-haystack
uv sync --extra dev

# Full end-to-end run against the FakeProvider — no API keys needed
uv run niah run configs/runs/smoke.fake.yaml

# Run the example configs against real providers (needs .env keys)
uv run niah run configs/runs/single_needle.example.yaml

# Lint / format / type-check / test (same as CI)
uv run ruff check .
uv run ruff format --check .
uv run mypy needlehaystack
uv run pytest
```

In this contributor environment, prefix CLI calls with `uv run` (above) or
activate the venv first (`source .venv/bin/activate`) and drop the prefix.
End users who installed via `pip install` use bare `niah`.

---

## Original story & historical results

The original 2023 runs that started all this:

- **Behind-the-scenes video**: [youtu.be/KwRRuiCCdmc](https://youtu.be/KwRRuiCCdmc)
- **OpenAI GPT-4 analysis**: [tweet thread](https://twitter.com/GregKamradt/status/1722386725635580292)
- **Anthropic Claude 2.1 analysis**: [tweet thread](https://twitter.com/GregKamradt/status/1727018183608193393)
- **How the viz was built**: [tweet](https://twitter.com/GregKamradt/status/1729573848893579488) ·
  [Google Slides version](https://docs.google.com/presentation/d/15JEdEBjm32qBbqeYM6DK6G-3mUJd7FAJu-qEzj8IYLQ/edit?usp=sharing)

![Needle In A Haystack code snippet](img/NeedleHaystackCodeSnippet.png)

### OpenAI's GPT-4-128K (Run 11/8/2023)

<img src="img/GPT_4_testing.png" alt="GPT-4-128 Context Testing" width="800"/>

### Anthropic's Claude 2.1 (Run 11/21/2023)

<img src="img/Claude_2_1_testing.png" alt="Claude 2.1 Context Testing" width="800"/>

The raw result files from those original runs are preserved in
[`original_results/`](./original_results) for posterity — the schema does not
match v2, so they don't load with the new tooling.

### How multi-needle spacing works (still accurate in v2)

Given N needles and a starting `depth_percent`, the `EvenSpreadInserter`
places the first needle at `depth_percent`, then distributes the rest evenly
through the remaining context up to 100%. The interval is:

```
depth_percent_interval = (100 - depth_percent) / N
```

So for N=10 needles starting at depth_percent=40:

```
depth_percent_interval = (100 - 40) / 10 = 6

Needle 1: 40
Needle 2: 46
Needle 3: 52
Needle 4: 58
Needle 5: 64
Needle 6: 70
Needle 7: 76
Needle 8: 82
Needle 9: 88
Needle 10: 94
```

v2 fixes a bug in the v1 multi-needle code where each needle's reported depth
was off by however much the earlier needles had inflated the token count. The
new inserter computes target depths against the pre-insertion length and
reports the true depth each needle landed at.

---

## License

MIT — see [LICENSE.txt](LICENSE.txt). Use of this software requires
attribution to the original author and project.
