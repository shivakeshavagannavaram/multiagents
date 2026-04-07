O# DESIGN.md — Multi-Agent Tool-Use Conversation Generator

## Architecture & Decisions

### System Overview

The system follows a pipeline architecture with six core components connected through a shared data model layer:

```
ToolBench Data → [Ingestor] → [Tool Graph] → [Sampler] → [Multi-Agent Generator] → [Judge] → JSONL
                                                 ↑                    ↑
                                                 └── [Diversity Steering (mem0)] ──┘
```

### Component Architecture

**1. Data Layer (`saplvl/models.py`)**

All components share Pydantic v2 models: `Tool`, `APIEndpoint`, `ToolParameter`, `ToolCall`, `ToolOutput`, `Message`, `JudgeScore`, `Conversation`. This provides type safety, validation, and serialization throughout the pipeline.

Key decision: Using Pydantic over dataclasses because we need JSON serialization for JSONL output, validation for score clamping, and `model_validate_json()` for deserialization. The overhead is negligible for our use case.

**2. Ingestor (`saplvl/ingestor/`)**

- `ToolBenchDownloader`: Fetches data from HuggingFace using the `datasets` library, with caching to avoid re-downloads.
- `ToolBenchParser`: Defensive parsing with fallbacks for every field. ToolBench has known quality issues — ~50% of original data had parsing errors, parameters frequently missing types or descriptions. Our parser defaults to `type="string"` and `description=""` rather than failing.
- `ToolRegistry`: In-memory indexed collection with O(1) lookups by name, category, and (tool_name, api_name) pair. Serializable to JSON for reuse between `build` and `generate` commands.

Data model decisions:
- Parameters are flattened to `{name, type, description, default, example_value}` rather than preserving nested JSON schemas, because ToolBench rarely uses nested structures and a flat model simplifies both graph construction and Claude tool definition generation.
- Categories are stored as strings on each Tool, not in a separate hierarchy. ToolBench categories are flat (49 top-level categories), so a hierarchy would add complexity without benefit.

**3. Tool Graph (`saplvl/graph/`)**

A NetworkX DiGraph where nodes are `(tool_name, api_name)` tuples — API-level granularity rather than tool-level — because chaining happens between specific endpoints (e.g., `search_hotels` → `book_hotel`), not between entire tools.

Three edge types, each serving a different role in chain sampling:

| Edge Type | Direction | Purpose | Construction |
|-----------|-----------|---------|-------------|
| `same_category` | Bidirectional | Enable cross-tool chains within a domain | All pairs in same ToolBench category |
| `parameter_compatibility` | Directed A→B | Enable realistic data flow chains | A's inferred outputs match B's input parameter names |
| `semantic_similarity` | Bidirectional | Enable discovery of related tools across categories | Cosine similarity of description embeddings > 0.65 |

Key decision: Parameter compatibility uses heuristic output inference (e.g., `search_hotels` likely returns `hotel_id`) because ToolBench does not define response schemas. The heuristic matches common REST patterns (`get_X` returns `X_id`). This is imperfect but sufficient — the sampler proposes candidates, and the LLM agents make the actual coherent connections.

Semantic similarity uses `all-MiniLM-L6-v2` (fast, 384-dim embeddings). For ~16K API descriptions, computing all pairwise similarities is feasible via batched matrix multiplication on normalized vectors. Threshold of 0.65 balances recall (finding useful connections) against noise (spurious edges).

**4. Sampler (`saplvl/sampler/`)**

`ToolChainSampler` walks the graph to produce tool chains. The walk uses weighted edge selection:
- `parameter_compatibility`: weight 3.0 (strongest — creates realistic data flow)
- `same_category`: weight 2.0 (good for multi-tool scenarios)
- `semantic_similarity`: weight 1.0 (enables cross-domain discovery)

`SamplingConstraints` supports: `min_tools`, `max_tools`, `min_steps`, `max_steps`, `categories` filter, `required_tools`, `exclude_tools`, `allow_parallel`.

The sampler enforces the 50-60% multi-step/multi-tool requirement by tracking the running ratio during generation and adjusting constraints dynamically — setting `min_tools=2, min_steps=3` when the ratio is below target.

`ScenarioGenerator` uses Claude to convert a tool chain into a natural-language user scenario. This is a critical bridge: the sampler ensures graph-grounded tool selection, while the scenario generator ensures the resulting conversation has a plausible user motivation.

**5. Multi-Agent Generator (`saplvl/agents/`)**

Four agent roles:

| Agent | LLM | Role | Structured Output |
|-------|-----|------|-------------------|
| `UserSimulatorAgent` | Claude Sonnet | Generates user messages | No (free text) |
| `AssistantAgent` | Claude Sonnet | Selects tools, generates responses | **Yes** — Claude native `tool_use` |
| `ToolExecutorAgent` | OpenAI GPT-4o-mini | Generates mock tool responses | Yes — JSON mode |
| `JudgeLLM` | OpenAI GPT-4o | Scores conversations | Yes — JSON mode |

The `AssistantAgent` uses Claude's native `tool_use` feature, which produces validated JSON arguments against the tool's input schema. This satisfies the structured output requirement and provides reliable tool call extraction.

Communication protocol:
1. `Orchestrator` initializes `ConversationContext`
2. `UserSimulatorAgent.run(context, scenario)` → user `Message`
3. `AssistantAgent.run(context, available_tools)` → assistant `Message` (possibly with `tool_calls`)
4. If tool calls: `ToolExecutorAgent.run(context, tool_calls)` → tool `Message` with outputs
5. `AssistantAgent.run(context)` → assistant follows up with tool results
6. Loop until all chain tools used or max turns reached
7. `JudgeLLM.score(conversation)` → `JudgeScore`
8. If score < threshold: retry with repair hints

LLM assignment rationale:
- Claude for user/assistant: Superior at natural conversation and has native tool_use with schema validation.
- OpenAI for judge/mock generation: Reliable JSON mode, cost-effective for structured output tasks (GPT-4o-mini for mocks, GPT-4o for judge).

**6. Quality Evaluation (`saplvl/judgellm/`)**

Three scoring dimensions, each on a 1-5 scale:

1. **Naturalness**: Does the conversation feel like a real user-assistant interaction? Evaluates tone, flow, and appropriate use of clarifying questions.
2. **Tool Correctness**: Are the right tools selected with valid arguments? Are values from previous tool outputs correctly referenced (not hallucinated)?
3. **Task Completion**: Is the user's original goal fully addressed by the end of the conversation?

These dimensions were chosen because they capture the three failure modes most likely to degrade training data quality:
- Unnatural conversations teach models awkward interaction patterns
- Incorrect tool calls teach wrong tool selection or argument construction
- Incomplete tasks teach models to abandon requests prematurely

The judge uses a detailed rubric with explicit score-level definitions (5=excellent through 1=failure) to reduce scoring variance.

### Automatic Retry / Repair

When a conversation scores below the quality threshold (default 3.0), the orchestrator retries with dimension-specific repair hints injected into agent prompts:

- Low naturalness → "Make the conversation more natural and conversational. Use contractions, informal language."
- Low tool_correctness → "Ensure tool calls use correct parameter names and valid values. Double-check schema."
- Low task_completion → "Make sure the user's request is fully addressed. Complete all steps."

The orchestrator keeps the best-scoring attempt across retries (up to `max_retries=2`). This avoids discarding conversations entirely while bounding generation cost.

---

## Context Management Design

### Within-Conversation Grounding

`ConversationContext` (`saplvl/memory/context.py`) serves as the shared state within a single conversation:

1. **Message history**: All messages are accumulated and formatted into prompts via `build_prompt_context()`. The assistant sees the full conversation flow to maintain coherence.

2. **Value extraction**: When tool outputs arrive, `_extract_values()` recursively extracts scalar values (IDs, names, codes, etc.) and stores them in `_available_values`. The `format_available_values()` method injects these into the assistant's system prompt so it can reference real values from previous tool calls.

3. **Session state**: `SessionState` (in the simulator) independently tracks generated values for mock response consistency. If a search returns `hotel_id: "htl_881"`, a subsequent booking call should accept that exact ID.

**Tradeoff**: We inject the full conversation history plus extracted values into every prompt. This is simple and effective but grows linearly with conversation length. For conversations beyond ~10 turns, we truncate oldest messages. A more sophisticated approach would use selective retrieval (e.g., only inject values relevant to the current tool's parameters), but the added complexity isn't justified for our typical 5-7 turn conversations.

### Cross-Conversation Steering

`DiversitySteering` (`saplvl/memory/steering.py`) uses mem0 to track what has been generated and steer toward diversity:

1. After each conversation, records a summary in mem0: tools used, categories, pattern type (single_call, multi_tool, etc.).
2. Before generating a new conversation, searches mem0 for similar tool combinations.
3. If over-represented tools are detected (usage > 2x mean), provides guidance to the scenario generator.
4. Identifies under-represented categories and suggests incorporating them.
5. Maintains local counters as a fast fallback when mem0 is unavailable.

The `--no-cross-conversation-steering` CLI flag disables this entirely (sets `enabled=False`), producing Run A for the diversity experiment.

**Where this breaks down**: mem0's approximate nearest-neighbor search is non-deterministic, so exact reproducibility between runs with the same seed is not guaranteed when steering is enabled. We document this and focus diversity metrics on distributional properties rather than exact sequence matching. At scale (10K+ conversations), the semantic search would also need to be bounded (limit results, use more targeted queries) to avoid query latency becoming a bottleneck.

**What we would do differently at scale**: Replace mem0 with a lightweight database tracking tool/category/pattern histograms directly. The semantic search adds overhead without proportional benefit once we have enough data — simple frequency-based steering (sample tools inversely proportional to their usage count) would be faster and deterministic. We'd also add batch-level steering: plan the next N conversations' tool selections together to optimize for coverage.

---

## Prompt Design

### Key Prompts

**1. Scenario Generation Prompt** (`saplvl/sampler/scenario.py`)

```
You are a scenario designer for a synthetic conversation dataset.
Given a sequence of API tools that will be used in a conversation,
generate a realistic user scenario that would naturally require
these tools in this order.
...
Requirements:
- Write a 2-3 sentence scenario
- Make it natural to use these tools in roughly this order
- Include specific details (city names, dates, preferences)
- Sometimes make the request slightly ambiguous
- Do NOT mention tool names or API endpoints
```

**Why this structure**: The prompt bridges graph-sampled tool chains and natural user behavior. By instructing "sometimes make the request slightly ambiguous," we generate conversations that include disambiguation turns — a dataset requirement. Not mentioning tool names prevents the user simulator from producing unrealistically specific requests.

**2. Assistant System Prompt** (`saplvl/agents/assistant.py`)

```
You are a helpful AI assistant that can use tools to accomplish tasks.
When a user makes a request, you should:
1. If the request is ambiguous or missing required information, ask a clarifying question FIRST
2. Otherwise, use the available tools to fulfill the request
3. After receiving tool results, summarize them for the user
...
{available_values}
```

**Why this structure**: The numbered priority list ensures disambiguation happens before tool calls. Injecting `available_values` grounds tool arguments in real data from previous calls, preventing hallucinated IDs.

**3. Judge Rubric** (`saplvl/judgellm/judge.py`)

Each dimension has a 5-level rubric with explicit descriptions (e.g., naturalness 5 = "Indistinguishable from a real user-assistant interaction"). This reduces scoring variance compared to a simple "rate 1-5" instruction.

### Prompt Iteration That Did Not Work

**Failed approach: Single-prompt conversation generation**

Early iteration: A single Claude prompt that generated an entire conversation in one shot, given the tool chain and scenario.

Problems:
- Tool arguments were frequently hallucinated (no grounding in actual tool outputs)
- Conversations were formulaic — all followed the same ask→search→book pattern
- No natural disambiguation — the "user" always provided complete information
- Tool outputs were not internally consistent (IDs changed between references)

**What we learned**: Multi-turn generation with actual mock tool execution between turns is essential for grounding. The multi-agent approach, while more expensive (more API calls), produces coherent chains because each step sees real outputs from the previous step. The session state mechanism ensures IDs are consistent across the conversation.

---

## Diversity & Quality Analysis

### Diversity Metrics

**1. Tool Combination Entropy** (Shannon entropy over unique tool combinations)

Measures the variety of tool combinations used across the corpus. Computed as H = -Σ p(c) log₂ p(c) where c is a unique sorted tool combination. Higher values indicate more diverse tool usage. A corpus where every conversation uses the same tools has entropy 0; one where every conversation uses a unique combination has maximum entropy log₂(N).

**Why this metric**: It directly measures what our steering mechanism is designed to change — the distribution over tool combinations. Without steering, popular tools dominate and entropy is low. Steering should spread probability mass more evenly, increasing entropy.

**2. Domain Coverage Uniformity** (1 - coefficient of variation of category counts)

Measures how evenly the ToolBench categories are represented. Computed as 1 - (σ/μ) of per-category conversation counts. A value of 1.0 means perfect uniformity; 0.0 means a single category dominates.

**Why this metric**: Entropy captures variety but not balance. A corpus could have high entropy (many unique combinations) but all from the Travel category. Coverage uniformity ensures the dataset spans the full breadth of ToolBench's 49 categories.

### Experiment Setup

Both runs use the same seed (`--seed 42`) and generate 100 conversations:
- **Run A**: `python cli.py generate --seed 42 --count 100 --no-cross-conversation-steering`
- **Run B**: `python cli.py generate --seed 42 --count 100`

### Expected Results

| Metric | Run A (no steering) | Run B (with steering) |
|--------|--------------------|-----------------------|
| Tool Combination Entropy | Lower | Higher |
| Domain Coverage Uniformity | Lower | Higher |
| Mean Naturalness | Baseline | Similar or slightly lower |
| Mean Tool Correctness | Baseline | Similar |
| Mean Task Completion | Baseline | Similar |

### Diversity–Quality Tradeoff Analysis

**Expected tradeoff**: Steering may slightly reduce quality scores because:
1. It pushes generation toward less-common tools with sparser descriptions, making it harder for the LLM to generate realistic scenarios.
2. Cross-domain chains (e.g., Weather + Finance) may feel less natural than within-domain chains (e.g., hotel search + hotel booking).

**Mitigation**: The sampler still prioritizes `parameter_compatibility` edges (weight 3.0) even with steering, so chains remain structurally coherent. Steering operates primarily through scenario generation guidance and tool exclusion — it doesn't force incoherent chains.

**Honest uncertainty**: The magnitude of the quality cost is hard to predict without running the full experiment with real LLM calls. If quality drops significantly (mean scores below 3.0), we would:
1. Soften the steering — use guidance suggestions rather than hard exclusions
2. Only steer tool selection, not scenario framing
3. Set a floor: never exclude tools that have strong parameter_compatibility edges with the rest of the chain

*Note: Numeric results will be populated after running the full pipeline with real API calls.*

---

## What We Would Do Next

If time permitted, these are the extensions in priority order:

1. **Async generation**: Use `asyncio` to parallelize LLM calls within a batch. Each conversation is independent, so we could generate 10+ concurrently, reducing wall-clock time ~10x.

2. **Response schema inference**: Use the LLM to infer likely response schemas from API descriptions, then validate mock responses against inferred schemas. This would improve tool output quality beyond our current heuristic.

3. **Conversation branching**: Generate multiple assistant responses per turn and use the judge to select the best one (best-of-N sampling). More expensive but higher quality.

4. **Fine-grained diversity control**: Instead of binary steering on/off, expose a "diversity temperature" that interpolates between pure graph sampling and maximally steered sampling.

5. **Human evaluation calibration**: Compare LLM-as-judge scores against human ratings on a small calibration set to validate the rubric.
