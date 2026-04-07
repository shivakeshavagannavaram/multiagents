# Multi-Agent Tool-Use Conversation Generator

An offline synthetic data generation system that produces multi-turn conversations with multi-step tool-use traces, grounded in ToolBench API schemas.

## Quick Start

### Prerequisites

- Python 3.11+
- API keys for Anthropic (Claude) and OpenAI

### Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### Run the Pipeline

```bash
# Step 1: Download ToolBench data and build graph
python cli.py build

# Step 2: Generate conversations
python cli.py generate --seed 42 --count 100

# Step 3: Evaluate quality
python cli.py evaluate --input output/conversations.jsonl
```

## CLI Commands

### `build`

Downloads ToolBench data from HuggingFace, parses tools into a registry, and builds the tool graph with semantic similarity edges.

```bash
python cli.py build [--data-dir .cache]
```

**Artifacts produced:**
- `.cache/toolbench/tools.json` — raw parsed tool definitions
- `.cache/registry.json` — indexed tool registry
- `.cache/tool_graph.pkl` — NetworkX directed graph

### `generate`

Generates synthetic conversations using the multi-agent system.

```bash
python cli.py generate \
  --seed 42 \
  --count 100 \
  --output output/conversations.jsonl \
  [--no-cross-conversation-steering]
```

| Flag | Description |
|------|-------------|
| `--seed` | Random seed for reproducibility |
| `--count` | Number of conversations to generate (default: 100) |
| `--output` | Output JSONL file path |
| `--no-cross-conversation-steering` | Disable diversity steering (for Run A in the diversity experiment) |

### `evaluate`

Scores conversations using the LLM-as-judge and computes diversity metrics.

```bash
python cli.py evaluate --input output/conversations.jsonl [--output output/scored.jsonl]
```

## Output Format

Each line in the JSONL output contains a conversation record:

```json
{
  "conversation_id": "conv_0042",
  "messages": [
    {"role": "user", "content": "Find me a hotel in Paris for next weekend"},
    {"role": "assistant", "content": "What's your budget range?"},
    {"role": "user", "content": "Under 200 euros per night"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"tool_name": "HotelFinder", "api_name": "search_hotels",
       "arguments": {"city": "Paris", "max_price": 200}}
    ]},
    {"role": "tool", "content": null, "tool_outputs": [
      {"tool_call": {"tool_name": "HotelFinder", "api_name": "search_hotels", "arguments": {"city": "Paris", "max_price": 200}},
       "response": {"results": [{"id": "htl_881", "name": "Hotel du Marais", "price": 175}]}}
    ]},
    {"role": "assistant", "content": "I found Hotel du Marais at 175 EUR/night. Shall I book it?"}
  ],
  "tool_calls": [...],
  "tool_outputs": [...],
  "judge_scores": {
    "naturalness": 4.2,
    "tool_correctness": 4.8,
    "task_completion": 5.0,
    "reasoning": "Natural conversation flow with correct tool usage..."
  },
  "metadata": {
    "seed": 42,
    "chain": ["HotelFinder/search_hotels", "HotelFinder/book_hotel"],
    "tools_used": ["HotelFinder/search_hotels", "HotelFinder/book_hotel"],
    "categories_list": ["Travel"],
    "steering_enabled": true,
    "num_retries": 0
  }
}
```

## Running the Diversity Experiment

```bash
# Run A: steering disabled
python cli.py generate --seed 42 --count 100 --no-cross-conversation-steering --output output/run_a.jsonl

# Run B: steering enabled
python cli.py generate --seed 42 --count 100 --output output/run_b.jsonl

# Evaluate both
python cli.py evaluate --input output/run_a.jsonl
python cli.py evaluate --input output/run_b.jsonl
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests (fast)
pytest tests/ -v --ignore=tests/test_e2e.py

# Run E2E test
pytest tests/test_e2e.py -v
```

## Project Structure

```
saplvl/
  models.py             # Pydantic data models (Tool, Conversation, etc.)
  ingestor/             # ToolBench data download, parsing, registry
    downloader.py       # HuggingFace dataset download
    parser.py           # Raw JSON to clean models
    registry.py         # Indexed tool lookup
  graph/                # Tool graph construction
    builder.py          # NetworkX graph with 3 edge types
  sampler/              # Tool chain sampling
    sampler.py          # Graph walk with constraints
    scenario.py         # LLM scenario generation from chains
  simulator/            # Offline tool execution
    executor.py         # Mock response generation + session state
  memory/               # Context management
    context.py          # Within-conversation grounding
    steering.py         # Cross-conversation diversity (mem0)
  agents/               # Multi-agent system
    base.py             # Abstract agent interface
    user_simulator.py   # User message generation (Claude)
    assistant.py        # Tool-calling assistant (Claude tool_use)
    tool_executor.py    # Tool execution wrapper
    orchestrator.py     # Generation loop + retry/repair
  judgellm/             # Quality evaluation
    judge.py            # LLM-as-judge (OpenAI)
cli.py                  # CLI entry point
tests/                  # Unit, integration, and E2E tests
```
