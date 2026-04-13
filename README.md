# conv_gen — Multi-Agent Tool-Use Conversation Generator

An offline synthetic data generation system that produces multi-turn
conversations with multi-step tool-use traces, grounded in the ToolBench
API registry. 

The system:

1. Ingests ToolBench tools and APIs into a cleaned registry.
2. Builds a **tool graph** capturing data-flow and semantic relationships between endpoints.
3. Samples realistic tool chains from the graph using constrained sampling.
4. Runs a **multi-agent generator** (user, assistant, tool executor, judge) that produces turn-by-turn conversations where the assistant makes real tool calls and the mock executor returns chain-consistent responses.
5. Scores each conversation with an **LLM-as-judge** on naturalness, tool correctness, and task completion.
6. **Retries and repairs** low-scoring conversations using specific issues from the judge as targeted hints.
7. Writes the output as JSONL conforming to the deliverable schema.

Full design documentation including architectural tradeoffs and the diversity–quality analysis lives in **[DESIGN.md](DESIGN.md)**.

---

## Table of contents

- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [The three commands](#the-three-commands)
  - [`build` — one-time artifact construction](#build--one-time-artifact-construction)
  - [`generate` — produce conversations](#generate--produce-conversations)
  - [`evaluate` — score and optionally gate on thresholds](#evaluate--score-and-optionally-gate-on-thresholds)
  - [`compare` — side-by-side A/B analysis](#compare--side-by-side-ab-analysis)
- [End-to-end example (the diversity experiment)](#end-to-end-example-the-diversity-experiment)
- [Output format](#output-format)
- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

---

## Quick start

Two ways to run the pipeline. Pick whichever fits how you obtained the code.

### Option A — Installed as a package (cloned repo or source tarball)

```bash
# 1. Install as an editable package (recommended for development)
pip install -e .

# 2. Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# 3. Build the tool registry + graph (one-time, ~5-10 min)
conv-gen build

# 4. Generate 150 conversations
conv-gen generate --seed 42 --count 150 --output output/run.jsonl

# 5. Evaluate and verify quality thresholds
conv-gen evaluate --input output/run.jsonl \
    --threshold-overall 3.5 --threshold-ms-mt 0.50
```

### Option B — No install (downloaded zip or just want to run `cli.py` directly)

If you only downloaded a zip of the project and don't want to install anything system-wide, you can run the pipeline straight from source. All you need is the runtime dependencies:

```bash
# 1. Install only the runtime dependencies (no package install)
pip install -r requirements.txt

# 2. Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# 3. Build the tool registry + graph (one-time, ~5-10 min)
python3 cli.py build

# 4. Generate 150 conversations
python3 cli.py generate --seed 42 --count 150 --output output/run.jsonl

# 5. Evaluate and verify quality thresholds
python3 cli.py evaluate --input output/run.jsonl \
    --threshold-overall 3.5 --threshold-ms-mt 0.50
```

Both paths produce identical output. The only difference: Option A registers a `conv-gen` command on your PATH so you don't have to type `python3 cli.py`. The rest of this README mixes both forms — `conv-gen <cmd>` and `python3 cli.py <cmd>` are interchangeable.

If `evaluate` exits with code 0, every threshold passed. If it exits with code 1, the printed report shows which dimension fell below threshold.

---

## Prerequisites

- **Python 3.11+** (tested on 3.13)
- **API keys** for both Anthropic (Claude) and OpenAI (GPT-4o + GPT-4.1-nano) set as environment variables:
  - `ANTHROPIC_API_KEY`
  - `OPENAI_API_KEY`
- **Network access** for the first `build` run (downloads ToolBench). After that, everything is cached in `.cache/` and no network is needed except for the LLM API calls.

---

## Installation

The project ships as a proper Python package via `pyproject.toml`, but it's also fully runnable straight from source if you just downloaded a zip. You don't need to install it as a package to use the pipeline.

### Step 1 — Get the code

Either clone the repo:

```bash
git clone <repo-url>
cd multiagents
```

Or download the project zip, extract it, and `cd` into the extracted folder.

### Step 2 — Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

Pick one of the following, depending on how you want to use the project.

**(a) Run the pipeline straight from source, no package install** — use this if you downloaded a zip, or if you just want to run the code without installing anything:

```bash
pip install -r requirements.txt
```

Then invoke the pipeline with `python3 cli.py <command>` from the project root.

**(b) Editable package install** — recommended if you plan to modify the code. Registers a `conv-gen` command on your PATH:

```bash
pip install -e .
```

**(c) Editable install with test dependencies** — (b) plus pytest + pytest-xdist:

```bash
pip install -e ".[dev]"
```

**(d) Regular package install** — for shipping to another machine:

```bash
pip install .
```

**(e) Build a distributable wheel + tarball** — for PyPI or offline distribution:

```bash
pip install build
python3 -m build
# → dist/conv_gen-0.1.0-py3-none-any.whl and dist/conv_gen-0.1.0.tar.gz
```

### Invoking the pipeline

After any of the above, both invocation styles work:

| Install mode | Command style |
|---|---|
| (a) `requirements.txt` only | `python3 cli.py <command>` |
| (b)(c)(d) `pip install` | `conv-gen <command>` OR `python3 cli.py <command>` |

They do the same thing. The rest of this README uses `python3 cli.py` because it works in all install modes.

Key dependencies (all pinned in `pyproject.toml` and `requirements.txt`):
- `anthropic` — Claude API client
- `openai` — GPT-4o + GPT-4.1-nano API client
- `pydantic` — data models
- `networkx` — tool graph
- `sentence-transformers` — semantic edges in the graph
- `mem0ai` — cross-conversation steering memory (optional; pipeline falls back gracefully if mem0 fails to initialize)
- `gdown` — ToolBench dataset download from Google Drive
- `click` — CLI framework
- `pytest` — test suite (dev dependency)

---

## The three commands

All commands are invoked via `python3 cli.py <command>`. Common flags:

- `--data-dir PATH` (group-level) — where cached artifacts live. Default: `.cache`
- `-v` / `--verbose` (group-level) — enable debug logging

### `build` — one-time artifact construction

Downloads the full ToolBench dataset (~10,600 tools), parses it, selects 500 high-quality tools for chain potential, LLM-enriches missing response schemas, and builds the knowledge graph. Saves everything to `.cache/`.

```bash
python3 cli.py build
```

Produces:
- `.cache/registry.json` — serialized `ToolRegistry` with 500 selected tools, ~12,000 endpoints
- `.cache/tool_graph.pkl` — the NetworkX `DiGraph` with all nodes and edges
- `.cache/registry_full.json` — the full parsed registry (pre-selection) for reference

This step only needs to run once. Subsequent `generate` commands read from the cached artifacts.

### `generate` — produce conversations

The main command. Runs the full pipeline and writes conversations to a JSONL file.

```bash
python3 cli.py generate --seed 42 --count 150 --output output/run.jsonl
```

Options:
- `--seed INT` — random seed for reproducibility. Required for reviewer runs.
- `--count INT` — number of conversations to generate. Default: 100.
- `--output / -o PATH` — output JSONL file path. Default: `output/conversations.jsonl`
- `--no-cross-conversation-steering` — disable the diversity steering mechanism. This is the flag used for **Run A** in the diversity experiment (see spec section 5.3). When omitted, steering is on and the run is **Run B**.
- `--no-llm-mocks` — disable LLM-based mock tool responses, use schema-derived mocks instead. Useful for fully-offline runs but produces less realistic tool outputs.

Runtime: **~45–75 minutes** for 150 conversations at typical API speeds. Cost: **~$0.40** in API credits per 150-conversation run.

As conversations are generated, progress is printed per conversation:
```
[42/150] conv_9e3b0327: 19 turns, 3 tool calls, 2 tools | scores: nat=5.0 tool=5.0 task=5.0
```

### `evaluate` — score and optionally gate on thresholds

Reads a JSONL file, scores any unscored conversations with the LLM-as-judge, computes diversity and quality metrics, and optionally gates on quality thresholds.

```bash
python3 cli.py evaluate --input output/run.jsonl
```

Options:
- `--input PATH` — input JSONL file (required)
- `--output / -o PATH` — where to write the scored output. Defaults to overwriting the input file.
- `--registry PATH` — path to the registry.json. Defaults to `.cache/registry.json`. Used for computing `unique_tool_coverage_ratio`.
- `--json PATH` — also write a machine-readable metrics summary to this JSON file
- **Threshold flags** (optional pass/fail gates):
  - `--threshold-naturalness FLOAT` — fail if mean naturalness < value
  - `--threshold-tool FLOAT` — fail if mean tool_correctness < value
  - `--threshold-task FLOAT` — fail if mean task_completion < value
  - `--threshold-overall FLOAT` — fail if overall mean < value
  - `--threshold-ms-mt FLOAT` — fail if MS+MT rate < value (0.0–1.0 scale)

When no `--threshold-*` flags are set, `evaluate` prints the report and exits 0. When any are set, it checks each and prints PASS/FAIL per threshold, exiting with code 1 if any fail and 0 if all pass. This makes `evaluate` usable as the pass/fail gate of an end-to-end validation workflow.

**Example report output:**

```
====================================================================
  EVALUATION RESULTS
====================================================================

  Conversations scored: 150

  QUALITY (LLM judge means, 1-5)
    Naturalness:       4.51
    Tool correctness:  3.87
    Task completion:   4.15
    Overall mean:      4.18

  SPEC COMPLIANCE
    MS+MT rate:        64.7%
    Real chaining:     63.3%

  DIVERSITY — PRIMARY (steering target metrics)
    tool_usage_entropy:             0.9932
    unique_tools_used:              369
    unique_tool_coverage_ratio:     0.0326

  DIVERSITY — SECONDARY
    top_5_tool_concentration:       0.0394
    tool_combination_entropy:       7.089
    domain_coverage_uniformity:     0.8043
    unique_chain_ratio:             0.9722
    unique_chain_combinations:      146/150

====================================================================
  THRESHOLD CHECKS
====================================================================
  [PASS] naturalness          4.510  (threshold >= 4.0)
  [PASS] overall_mean         4.180  (threshold >= 3.5)
  [PASS] ms_mt_rate           0.647  (threshold >= 0.50)

  ✓ All 3 threshold check(s) passed.
  Overall result: PASS
```

### `compare` — side-by-side A/B analysis

Compares two generated runs and prints quality/diversity metrics with deltas. Designed for the diversity experiment (Run A vs Run B).

```bash
python3 cli.py compare output/run_a.jsonl output/run_b.jsonl \
    --label-a "Run A (no steering)" --label-b "Run B (steering)" \
    --json output/comparison.json
```

Options:
- `RUN_A`, `RUN_B` — two JSONL files to compare (positional)
- `--registry PATH` — `.cache/registry.json` for coverage ratio
- `--label-a STR`, `--label-b STR` — labels for the report columns
- `--json PATH` — write full comparison to a JSON file

Produces a table with quality metrics, spec compliance (MS+MT, chaining), primary diversity metrics (`tool_usage_entropy`, `unique_tools_used`, `unique_tool_coverage_ratio`), secondary diversity metrics, per-type MS+MT breakdown, and tool-overlap stats.

---

## End-to-end example (the diversity experiment)

This is the full reviewer workflow matching the spec's section 5.3:

```bash
# Set API keys in this shell
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# 1. Build artifacts (one-time, ~10 min)
python3 cli.py build

# 2. Run A — steering DISABLED
python3 cli.py generate \
    --seed 42 --count 150 \
    --no-cross-conversation-steering \
    --output output/run_a.jsonl

# 3. Run B — steering ENABLED (default)
python3 cli.py generate \
    --seed 42 --count 150 \
    --output output/run_b.jsonl

# 4. Compare the two runs side-by-side
python3 cli.py compare \
    output/run_a.jsonl output/run_b.jsonl \
    --label-a "Run A (no steering)" \
    --label-b "Run B (steering)" \
    --json output/comparison.json

# 5. Check quality thresholds on each run
python3 cli.py evaluate \
    --input output/run_a.jsonl \
    --threshold-overall 3.5 --threshold-ms-mt 0.50

python3 cli.py evaluate \
    --input output/run_b.jsonl \
    --threshold-overall 3.5 --threshold-ms-mt 0.50
```

The output of `compare` is what's referenced in DESIGN.md section 4.2 (Diversity & Quality Analysis). `output/comparison.json` contains the full machine-readable report for further analysis.


## Output format

Every conversation is one JSONL record. Abbreviated example:

```json
{
  "conversation_id": "conv_9e3b0327",
  "messages": [
    {"role": "user", "content": "Find me a hotel in Paris for next weekend"},
    {"role": "assistant", "content": "What's your budget range?"},
    {"role": "user", "content": "Under 200 euros per night"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"endpoint": "hotels/search",
       "arguments": {"city": "Paris", "max_price": 200, "currency": "EUR"}}
    ]},
    {"role": "tool", "content": {"results": [
      {"id": "htl_881", "name": "Hôtel du Marais", "price": 175}
    ]}},
    {"role": "assistant", "content": null, "tool_calls": [
      {"endpoint": "hotels/book",
       "arguments": {"hotel_id": "htl_881", "check_in": "2026-04-11"}}
    ]},
    {"role": "tool", "content": {"booking_id": "bk_3391", "status": "confirmed"}},
    {"role": "assistant", "content": "I've booked Hôtel du Marais for Apr 11. Confirmation: bk_3391."}
  ],
  "judge_scores": {
    "naturalness": 4.2,
    "tool_correctness": 4.8,
    "task_completion": 5.0,
    "reasoning": "..."
  },
  "metadata": {
    "seed": 42,
    "conversation_type": "simple_sequential",
    "pattern": "sequential",
    "steering_enabled": true,
    "scenario": "User wants to book a hotel in Paris for next weekend...",
    "chain": ["hotels/search", "hotels/book"],
    "tools_used": ["hotels/search", "hotels/book"],
    "num_turns": 7,
    "tools_planned": 2,
    "tools_executed": 2,
    "num_retries": 0,
    "categories": {"hotels/search": "Travel", "hotels/book": "Travel"},
    "categories_list": ["Travel"],
    "plan_summary": {...}
  }
}
```

**Key shape rules:**

- **Tool calls** use `{"endpoint": "<tool>/<api>", "arguments": {...}}` — a single endpoint string plus a typed argument dict.
- **Tool messages** hold the response dict directly in `content` (no separate `tool_outputs` list).
- **Parallel tool calls** expand to multiple consecutive `{"role": "tool", ...}` messages, one per call, in the same order as the preceding assistant message's `tool_calls` array.

See **DESIGN.md section 5** for the full metadata schema field-by-field.

---

## Reproducibility

**Run A (`--no-cross-conversation-steering`) is fully deterministic** at the Python seed layer. Running `generate --seed 42 --count 150 --no-cross-conversation-steering` twice produces the same conversations modulo inherent LLM-side non-determinism (which comes from Anthropic/OpenAI serving, outside our control).

**Run B (steering enabled) has small non-determinism** from mem0's vector search. The primary steering signal (counter-based `get_exclude_tools`) is deterministic, but mem0's "similar past conversations" hint feeds into a textual guidance string on the scenario generator's prompt. ANN searches can return different neighbors across runs, producing slightly different scenario text which cascades through the user simulator and assistant. The net effect on metrics is small (~±1 percentage point on MS+MT, ~±0.005 on entropy).

For bit-exact reproducibility of Run B, use `--no-cross-conversation-steering`. See **DESIGN.md section 6** for the full reproducibility story.

---

## Testing

```bash
# Fast test suite (~60 seconds, no API calls)
python3 -m pytest tests/

# Verbose output
python3 -m pytest tests/ -v

# Just unit tests for a specific module
python3 -m pytest tests/test_sampler.py -v

# Opt-in real end-to-end test (requires API keys, ~40 min, ~$0.50)
python3 -m pytest tests/test_e2e.py -m e2e_real
```

The test suite contains **141 tests** covering:

| File | Tests | Coverage |
|---|---|---|
| `test_agents.py` | 10 | User simulator, assistant (tool_use, sanitize, credential strip), tool executor |
| `test_e2e.py` | 3 | Fast mocked wiring test + opt-in real end-to-end test |
| `test_graph.py` | 14 | Knowledge graph builder, edge construction |
| `test_ingestor.py` | 18 | Parser, registry, edge cases for malformed ToolBench data |
| `test_judge.py` | 25 | Judge LLM, field normalization, chain-break detection |
| `test_memory.py` | 31 | Context, diversity steering, DiversityMetrics, QualityMetrics |
| `test_orchestrator.py` | 5 | Full orchestrator, **retry/repair loop integration** |
| `test_output_format.py` | 24 | Wire format serialization, backwards-compatible loading |
| `test_sampler.py` | 18 | Chain sampling, constraints, **must_include_categories**, **exact_steps** |
| `test_simulator.py` | 12 | Session state, schema fallback, LLM mock |

The `e2e_real` test runs the full pipeline end-to-end with real LLMs against cached build artifacts, generates 100 conversations, and asserts LLM-judge scores meet justified thresholds. It is skipped automatically when API keys or cached artifacts are missing.


## Project layout

```
multiagents/
├── cli.py                      # Main CLI entry point (build/generate/evaluate/compare)
├── conv_gen/                   # The package
│   ├── models.py               # Pydantic data models
│   ├── output_format.py        # Wire-format JSONL serialization
│   │
│   ├── ingestor/               # ToolBench → registry
│   │   ├── downloader.py       # HuggingFace fetch + cache
│   │   ├── parser.py           # Defensive JSON → model parser
│   │   ├── selector.py         # 5-factor tool selection
│   │   ├── schema_enricher.py  # LLM-based response schema filling
│   │   └── registry.py         # Indexed tool registry
│   │
│   ├── graph/
│   │   └── builder.py          # Knowledge graph construction
│   │
│   ├── sampler/
│   │   ├── sampler.py          # ToolChainSampler + SamplingConstraints
│   │   └── scenario.py         # Scenario generator (Claude)
│   │
│   ├── agents/                 # Multi-agent pipeline
│   │   ├── base.py             # BaseAgent interface
│   │   ├── user_simulator.py   # User simulator (Claude)
│   │   ├── assistant.py        # Assistant agent (Claude tool_use)
│   │   ├── tool_executor.py    # Tool executor wrapper
│   │   ├── director.py         # Conversation type director
│   │   ├── plan.py             # ConversationPlan state machine
│   │   └── orchestrator.py     # Turn loop + retry/repair
│   │
│   ├── simulator/
│   │   └── executor.py         # Mock tool executor with SessionState
│   │
│   ├── memory/
│   │   ├── context.py          # ConversationContext (within-conv grounding)
│   │   └── steering.py         # DiversitySteering + Diversity/Quality metrics
│   │
│   └── judgellm/
│       └── judge.py            # LLM-as-judge with structural checks
│
├── tests/                      # 141 unit + integration + e2e tests
│   ├── test_agents.py
│   ├── test_e2e.py
│   ├── test_graph.py
│   ├── test_ingestor.py
│   ├── test_judge.py
│   ├── test_memory.py
│   ├── test_orchestrator.py
│   ├── test_output_format.py
│   ├── test_sampler.py
│   ├── test_simulator.py
│   └── conftest.py
│
├── pytest.ini                  # Pytest config (markers registered)
├── requirements.txt
├── README.md                   # This file
└── DESIGN.md                   # Architecture, decisions, tradeoffs, analysis
```
