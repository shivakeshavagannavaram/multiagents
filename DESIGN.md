# Multi-Agent Tool-Use Conversation Generator

This document describes the design, decisions, and tradeoffs behind `conv_gen`,
a pipeline that generates synthetic multi-turn conversations containing
multi-step tool-use traces over the ToolBench registry.

---

## Document Structure


**Required sections**

1. **Architecture & Decisions** — system design, components, agents, protocol
   - 1.1 System overview
   - 1.2 Component breakdown (Ingestor, Tool Graph, Sampler, Director, Multi-Agent Generator, Orchestrator, Judge)
   - 1.3 Agent communication protocol

2. **Context Management Design** — within-conversation grounding + cross-conversation steering
   - 2.1 Within-conversation grounding (three-layer model)
   - 2.2 Cross-conversation steering (counters + mem0)
   - 2.3 Where grounding breaks down

3. **Prompt Design** — how the prompts were shaped and what I learned
   - 3.1 Assistant system prompt
   - 3.2 User simulator prompts
   - 3.3 Scenario generator prompt
   - 3.4 Judge prompt
   - 3.5 LLM coherence check prompt
   - 3.6 The iteration that didn't work: few-shot hotel examples

4. **Diversity & Quality Analysis** — metrics, Run A vs Run B, and the tradeoff
   - 4.1 Metrics chosen
   - 4.2 Results for both runs
   - 4.3 Diversity–quality tradeoff analysis

**Supporting sections**

5. **Output Format & Metadata Schema** — what a generated JSONL record looks like
   - 5.1 Wire format
   - 5.2 Metadata schema

6. **Reproducibility & Determinism** — how I handle ANN non-determinism from mem0
   - 6.1 What's deterministic
   - 6.2 What's non-deterministic
   - 6.3 How I handle this in the metrics
   - 6.4 What I don't do (and honest caveats)

7. **Model Choices** — which LLMs do what, and why

8. **Known Limitations & What I'd Do Next** — where the design breaks down and what I would change at scale

**Appendix** — running the pipeline end-to-end

---

## 1. Architecture & Decisions

### 1.1 System overview

The pipeline is organized into three phases, each wired to its own CLI subcommand:

- **build** — one-time, cached. Ingests ToolBench, builds the tool graph, enriches schemas.
- **generate** — per-conversation. Directs types, samples chains, runs the agent loop, scores each conversation.
- **evaluate** — offline analysis. Computes diversity and quality metrics over a generated JSONL.

Each phase uses a subset of the components below.

<img src="img.png" alt="Architecture" width="50%">

### 1.2 Component breakdown

Each box in the diagram above corresponds to one or two modules in `conv_gen/`. Here's what each does and why it's shaped that way.

#### Ingestor (`conv_gen/ingestor/`)

Three layered responsibilities:

- `ToolBenchDownloader` — fetches the raw ToolBench JSON from HuggingFace, caches locally, supports both the paper's referenced subset (~3,400 tools) and the full dataset (~10,600 tools).
- `ToolBenchParser` — defensive parser. ToolBench is messy: ~50% of tools have malformed or missing fields, parameters frequently lack types or descriptions, some tools have zero endpoints. Every field defaults to a safe value rather than crashing. Defaults: `type="string"`, `description=""`, `method="GET"`, `api_list=[]`.
- `ToolRegistry` — indexed in-memory collection with O(1) lookups by tool name, category, and `(tool_name, api_name)` pair. Serializable to `.cache/registry.json` so `build` and `generate` can run independently.

Plus a **schema enricher** (`ingestor/schema_enricher.py`) that runs during `build` and uses GPT-4.1-nano to fill in missing response schemas.

Why it exists:

- ToolBench documents inputs but rarely outputs.
- Without response schemas, the tool graph can't know what field names flow from one endpoint to another, and chain construction breaks.
- The enricher generates a plausible schema with domain-appropriate field names for every endpoint that doesn't already have one.
- Runs once at build time (~5 minutes for 10K endpoints), not at generate time.

#### Tool Graph (`conv_gen/graph/builder.py`)

A typed knowledge graph built on top of a NetworkX `DiGraph`.

NetworkX gives me the storage and traversal primitives — nodes, edges, neighbor queries, BFS/DFS. What it doesn't give me is any notion of *what kind of thing* a node is or *what kind of relationship* an edge represents. A plain DiGraph of "endpoint → endpoint" would have been faster to build, but it would have collapsed too much of the structure I need:

- **No way to ask "what fields does this endpoint produce?"** — I'd have to fall back to the raw registry every time.
- **No way to tier edges.** In a flat DiGraph, every edge is the same kind of edge. The walker couldn't prefer `same_tool` over `feeds_into_hard` over `semantic_bridge` without a parallel lookup table.
- **No way to compute IDF specificity cleanly.** IDF needs to count how many endpoints produce a given field name. That's a natural query if `output_field` is its own node type; it's a manual scan otherwise.
- **No way to slice by category.** "Give me every endpoint in `Travel`" is a one-hop query on a typed graph and a full-table scan on a flat one.

So I layered a KG on top: five node types (`category`, `tool`, `endpoint`, `parameter`, `output_field`) and five edge types (`belongs_to`, `takes_input`/`produces`, `same_tool`, `feeds_into`, `semantic_bridge`). NetworkX still does the bookkeeping — I store the type as a node/edge attribute — but every component reads the graph through the typed lens.

**What improved by going typed:**

- **The walker has tiered priority for free.** `_select_next` filters candidate edges by type and confidence, picking `same_tool` → `feeds_into_hard` → `feeds_into_soft` → `semantic_bridge` → same-category fallback, in that order. A flat DiGraph would have needed a separate priority structure alongside it.
- **Data flow is explicit, not implicit.** Because `output_field` and `parameter` are their own nodes, the `feeds_into` edge lives between two *fields* — not between two endpoints. That means I can ask "which specific output field of endpoint A feeds which specific parameter of endpoint B?" and get a direct answer. The walker uses this to guarantee every hop has a named data channel.
- **IDF specificity is a graph query.** For a given field name, I count in-degree on the `output_field` node to get "how many endpoints produce this field," then apply `log(N / count)`. Generic fields like `id` have high in-degree → near-zero IDF → weak chain signal. Domain-specific fields like `fixture_id` have low in-degree → high IDF → strong chain signal. This is the 0.40 weight in the confidence formula below.
- **Multiple consumers share one structure.** The sampler walks it, the scenario generator reads it to build API capability descriptions, the chain coherence checker consults it, the metrics compute domain coverage from it. A flat endpoint-only graph would have forced each consumer to reconstruct its own view.
- **Debugging is simpler.** When a chain looks wrong, I can inspect the typed nodes and edges directly (`graph.nodes[endpoint]["type"]`, `graph.edges[a, b]["confidence"]`) and see exactly why the walker picked that hop.

The cost of typing is small — a dict comprehension at build time to tag each node, and a few `if data["type"] == "..."` filters in the walker — and the clarity it buys is worth it for a system where six different components all need different views of the same underlying structure.

**Node types**

| Node type      | Meaning                                                |
| -------------- | ------------------------------------------------------ |
| `category`     | A ToolBench category (e.g., `Travel`, `Finance`)       |
| `tool`         | A vendor product (e.g., `HotelFinder`)                 |
| `endpoint`     | A specific API endpoint within a tool                  |
| `parameter`    | An input parameter on an endpoint                      |
| `output_field` | A field name produced in the endpoint's response       |

**Edge types**

| Edge type                  | Direction                                     | What it captures                                                       | How it's built                                                                                              |
| -------------------------- | --------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `belongs_to`               | category → tool, tool → endpoint              | Structural containment                                                 | Trivial — from registry hierarchy                                                                           |
| `takes_input` / `produces` | endpoint → parameter / endpoint → output_field | I/O schema for each endpoint                                           | Parsed directly from registry schemas                                                                       |
| `same_tool`                | endpoint → endpoint (within one tool)         | Workflow within a single vendor (e.g., `search` → `book`)              | Field-name matching on same-tool endpoints; confidence fixed at 1.0                                         |
| `feeds_into`               | output_field → parameter                      | Data flow across tools                                                 | Field-name match with IDF-based confidence formula; split into **hard** (≥ 0.7) and **soft** (0.6 – 0.7)     |
| `semantic_bridge`          | endpoint → endpoint (across tools)            | Cross-category connections via embedding similarity                    | Sentence-transformer cosine ≥ 0.75, filtered by data-driven category affinity                               |

The `feeds_into` confidence formula is a weighted mix:

```
confidence = 0.20 × name_quality  (exact normalized match = 1.0, root match = 0.5)
           + 0.10 × type_score    (same type = 1.0, compatible = 0.7)
           + 0.30 × category_score (same category = 1.0, different = 0.3)
           + 0.40 × specificity   (IDF-based, generic fields ~0, unique fields ~1)
```

> **Why these specific weights?**
>
> I gave specificity the highest weight because a field called `hotel_id` is a much stronger chain signal than a generic `id` that appears in 2,000 endpoints. IDF (inverse document frequency) computed over the whole registry gives me specificity automatically — common fields get near-zero weight, rare domain-specific fields get near-one. I made category proximity the second-strongest signal because within-category chains are generally more coherent. Name quality and type matter but are secondary for me.

> **Why is cross-category penalized (0.3 vs 1.0) but not excluded entirely?**
>
> Because genuinely useful cross-category chains exist — Travel→Weather, eCommerce→Payment — but they're rarer than same-category chains. The penalty ensures they don't dominate the sampling; the non-zero weight lets them through when specificity compensates for the category gap.

**Same-tool edges get confidence 1.0 and the top walker priority.** Within-tool chains (e.g., `Gmail/deleteForwarding` → `Gmail/setNotification`) are almost always coherent by construction — the tool's author designed them to work together.

#### Sampler (`conv_gen/sampler/sampler.py`)

`ToolChainSampler` walks the graph to produce chains in one of three patterns:

| Pattern | Method | Traversal strategy |
|---|---|---|
| SEQUENTIAL | `sample_sequential` | Seed → `_select_next` repeatedly. Priority tiers: (1) `same_tool`, (2) hard `feeds_into`, (3) soft `feeds_into`, (4) `semantic_bridge`, (5) same-category fallback with field-name overlap. Each hop is tagged with its tier. |
| PARALLEL | `sample_parallel` | Seed → find independent endpoints (no data flow between them) from different tools in the same category. All tools in one step. |
| SINGLE | `sample_single` | One endpoint weighted by schema quality. Used for quick lookups. |

**Chain quality gate (hybrid hard/soft).** After walking, each chain is classified:

- **Hard** = every hop is `same_tool` or `feeds_into_hard`. These chains have real data flow backing every step; accepted immediately.
- **Soft** = any hop used soft `feeds_into`, `semantic_bridge`, same-category fallback, or diversify fallback. These are sent to a one-shot LLM coherence check (GPT-4.1-nano) that asks "Does this form a realistic single-session workflow?" If YES, accepted. If NO, the walker retries with a new seed.

The sampler retries up to 8 times. If nothing passes, it returns the best-seen chain as a safety net (so I always get output).

> **Why a hybrid gate instead of strict hard-only?**
>
> I tried strict hard-only first and it produced very narrow chains. The math works against it: cross-category `feeds_into` confidence maxes out around 0.79, so most cross-category edges land in the soft tier. Forcing hard-only collapsed the dataset to within-category chains and lost the cross-domain diversity I wanted. The hybrid gate lets soft chains through when the LLM confirms they're plausible, which keeps the diversity while filtering the worst offenders.

> **Why a Low-cost LLM call instead of just tightening the confidence formula?**
>
> I tried tightening the formula first. The problem was edge cases like "Gmail→Zillow" via a generic `address` field — are structurally fine by the numbers, semantically they dont make sense. World knowledge was the missing ingredient. Nano is low-cost enough (~$0.0001 per check) that I can spend it on the ~30–60% of chains that reach the soft tier without worrying about cost.

**`SamplingConstraints` — the constraint interface.** A dataclass exposing every constraint the sampler honors:

```python
@dataclass
class SamplingConstraints:
    min_steps: int = 2
    max_steps: int = 4
    exact_steps: int | None = None        # convenience: collapses min/max to a fixed value
    min_tools: int = 2
    max_tools: int = 3
    max_categories: int = 3
    categories: list[str] | None = None              # "only from these"
    must_include_categories: list[str] | None = None # "at least one from each"
    required_tools: list[str] | None = None          # seed bias
    exclude_tools: list[str] = field(default_factory=list)
```

The two category fields deserve a word, because they mean opposite things.

The sampler needs to honor two different kinds of "category" requests, and they're not the same:

- **"Only use Travel tools."** → a filter. Everything in the chain must be Travel.
- **"Make sure at least one Travel tool shows up."** → a quota. The chain can be mostly anything, as long as Travel is represented.

Originally I only had one field (`categories`) and treated it as a filter. But the project asks for "a chain that includes at least one tool from a given domain" — that's the quota meaning, not the filter meaning. A chain that's *mostly Travel + one Weather* is a different shape from a chain that's *only Travel*. I needed both.

**The fix** — two separate fields on `SamplingConstraints`:

- `categories` = filter. Restrict the whole chain to these.
- `must_include_categories` = quota. At least one endpoint from each listed category must appear; the rest can come from anywhere.

**How the quota is enforced.** Two-step:

1. **Seed bias.** When picking the first endpoint, endpoints in the required categories get a 3× weight boost, so the walk usually starts from a required category. A cheap nudge at the start.
2. **Post-walk verification.** After the chain is built, `_chain_satisfies_must_include` checks that every required category actually appears. If not, throw the walk away and retry (up to 8 times).

So it's "try to land in the right place + verify afterwards."

**The other small constraints in the same dataclass:**

- **`exact_steps`** — just a convenience. Setting `exact_steps=3` is equivalent to writing `min_steps=3, max_steps=3`. `__post_init__` collapses it.
- **`required_tools`** — biases only the seed (the first hop). Later hops can go anywhere. Used when I want a specific tool to anchor the chain but don't care about the rest.
- **`exclude_tools`** — hard filter on the entire candidate pool. Any tool in this list cannot appear anywhere in the chain. This is what cross-conversation steering uses to forbid recently-overused tools.

**The mental model:**

- `categories` / `must_include_categories` shape **where** the chain is allowed to live.
- `required_tools` / `exclude_tools` shape **which specific tools** are in or out.
- `exact_steps` / `min_steps` / `max_steps` shape **how long** the chain is.

#### Director (`conv_gen/agents/director.py`)

The Director decides *what kind* of conversation to generate next, then hands the sampler concrete constraints.

Five conversation types:

| Type | Distribution | Chain shape | Purpose |
|---|---|---|---|
| `quick_lookup` | 10% | 1 tool, 1 step, SINGLE | Short factual lookup |
| `parallel_lookup` | 10% | 2–3 tools, PARALLEL | Simultaneous independent tools |
| `simple_sequential` | 10% | 2 tools, 2 steps, chained | Basic A→B flow |
| `multi_step_chain` | 40% | 3–4 tools, ≥2 distinct, chained | MS+MT workhorse |
| `full_workflow` | 30% | 4 tools, ≥3 distinct, 4 steps exactly | Long coherent workflow |

The Director tracks observed counts and picks whichever type is currently furthest below its target percentage. This gives me a self-balancing mix without hard quotas — on any individual slot the type is stochastic (in so far as "biggest gap" is stochastic when multiple types are equidistant), but across 150 conversations the totals converge to within ~1 of the targets.

> **Why separate the Director from the Sampler?**
>
> They do different things and I wanted them independently changeable. The sampler's job is graph traversal — pick coherent tool chains given a set of constraints. The director's job is dataset composition — decide what mix of conversation shapes the corpus should have. Mixing the two would couple "what shapes I want" with "how I traverse the graph." Separating them means I can change the distribution without touching the graph walker, and the sampler can serve other callers (tests, ad-hoc scripts, future evaluation modes) without inheriting director state.

> **Why 40% + 30% = 70% allocation to the MS+MT-eligible types?**
>
> The project requires 50–60% MS+MT in the final output. I over-allocate because there's execution drop-off — some `multi_step_chain` runs fail to reach 3 calls because Claude refuses, narrates instead of calling, or hits a budget cap. Starting at 70% of slots eligible gives me headroom for that drop-off and empirically lands the final output at ~55–65% MS+MT.

#### Multi-Agent Generator (`conv_gen/agents/`)

Five agent roles were defined with the orchestrator that runs them:

| Agent | LLM | Role | Structured output? |
|---|---|---|---|
| `UserSimulatorAgent` | Claude Sonnet | Generates user messages for each turn | No (free text) |
| `AssistantAgent` | Claude Sonnet | Selects tools, builds arguments, produces replies | **Yes — schema-validated tool call: `{name, input: {...}}` against the endpoint's `input_schema`** |
| `ToolExecutorAgent` | — (wrapper) | Runs the simulator, records results | No (delegates) |
| `ToolSimulator` | GPT-4.1-nano | Generates mock tool responses | **Yes — JSON object matching the endpoint's response schema** |
| `JudgeLLM` | GPT-4o | Scores conversations on three dimensions | **Yes — JSON object with per-dimension `{reasoning, score}` and a `specific_issues` list** |

Four structured-output channels total (including the schema enricher used at build time). The **AssistantAgent** is the most important one to get right, because its output directly drives the next phase of the pipeline — a malformed tool call means a malformed downstream execution. Using Claude's native `tool_use` rather than regex-parsing free text gives me server-side schema validation against my declared `input_schema`, which is the strongest correctness contract available.

#### Orchestrator (`conv_gen/agents/orchestrator.py`)

Runs the turn loop. Responsibilities:

1. Initialize `ConversationContext`, `ConversationPlan`, `SessionState`
2. Loop for up to `max_turns` iterations:
   - Generate a user message (User Sim) — informed by the plan's completion guidance
   - Generate an assistant response (Assistant Agent) — with structured tool calls
   - If tool calls, execute them (Tool Executor) and handle chained follow-ups up to `max_depth=3`
   - Handle refusal/narration with a re-prompt that forces the remaining tools
3. Score the final conversation with the judge
4. If the mean score is below `quality_threshold`, retry the whole conversation with repair hints extracted from the judge's `specific_issues`

The orchestrator is the only component that knows the overall flow. Every other component operates on narrow responsibilities:

- The sampler doesn't know about turns or scoring.
- The agents don't know about sampling or planning.
- The judge doesn't know about retries.
- The director doesn't know about orchestration.

This separation means I can swap any component for a mock in tests without wiring an entire subsystem.

**Two critical invariants enforced in `_execute_and_follow_up`:**

1. **No empty messages.** If an assistant call returns an empty response (no content, no tool calls), I skip it rather than adding it to context. Adding empty messages creates empty turns in the rebuilt history that Claude sees as itself having said nothing, which destabilizes subsequent turns.
2. **No orphaned `tool_use` blocks.** If the loop is on its last depth iteration and the assistant's follow-up has pending tool calls I won't execute, I **don't add it to context**. Adding a `tool_use` block without a matching `tool_result` violates the Claude API contract (every `tool_use` must be followed by a `tool_result` in the next user message) and produces JSONL files that can't be replayed through Claude.

Both invariants were added after I found that 84% of early-session conversations had ghost messages and a non-zero fraction had orphaned tool_use blocks —the fixes are now enforced structurally. Unit tests in `tests/test_orchestrator.py` cover the retry/repair flow; `tests/test_output_format.py` covers the structural invariants.

#### Judge (`conv_gen/judgellm/judge.py`)

`JudgeLLM` scores each conversation on three dimensions (1–5): `naturalness`, `tool_correctness`, `task_completion`.

The prompt uses chain-of-thought (reason then score) and few-shot examples for calibration, and includes explicit critical checks for hallucination, incoherent workflow, empty messages, and scenario mismatch.

After the LLM scores, `_apply_structural_checks` runs mechanical sanity checks that can only lower scores (never raise):
- No tool calls when tools were available → cap tool_correctness at 1.0
- **Chain break** — field-name-aware hallucination check → cap tool_correctness at 2.5
- Conversation too short to complete the task → cap task_completion at 2.0

The chain-break check is the interesting one. Naively, "the value in a tool-call argument isn't found in any prior tool response or user message" flags hallucination. But that same logic flags legitimate rule #11 fabrications (where the assistant is *expected* to make up a plausible ID when no prior tool produced one). The fix: make the check **field-name aware**. For each ID-like argument in call N (N≥2), I check whether a prior response contained a field with the same normalized name. If yes and the value is different, it's a real chain break (cap at 2.5). If no prior response had the field, the assistant had to fabricate — legitimate behavior, no flag. This eliminates ~80% of the old false positives while preserving all real catches. Unit tests in `tests/test_judge.py::TestChainBreakCheck` pin down both situations.

### 1.3 Agent communication protocol

Every agent extends a common `BaseAgent` class and exposes a `.run(context, **kwargs)` method. The orchestrator passes a shared `ConversationContext` object to each agent, which holds:

- `messages: list[Message]` — the authoritative conversation history
- `tool_outputs: list[ToolOutput]` — aggregated tool results (with back-references to their calls)
- `_available_values: dict[str, Any]` — flattened scalar values from tool responses, exposed to the assistant's system prompt as grounding context

A second state object, `SessionState`, lives in the `ToolSimulator` and tracks values across mock tool calls. It's how nano generates chain-consistent responses — "if the first search returned `hotel_id: htl_a8f2`, the second call using `htl_a8f2` as input should find the same hotel".

Two state objects exist because they serve different consumers:

- `ConversationContext` — feeds the assistant and the judge
- `SessionState` — feeds the mock executor

> **Why two state objects instead of one shared dict?**
>
> Honestly, I considered unifying them and didn't. The tradeoff I weighed: unifying gives me one source of truth and less boilerplate, but the extraction logic is different between the two consumers. `ConversationContext._available_values` wants a flat `{key → value}` dict for prompt injection; `SessionState._store` wants the same but also wants reverse `{value → key}` lookups for "did nano already produce this value for some other field?" Unifying would force one of them into a shape that doesn't fit its consumer.
>
> My current split works and the duplication is contained — both use the same recursive walker. If I were at scale with 100x the codebase, I'd invest in unifying. At the current size, keeping them separate is the lower-risk choice.

**Turn-level communication flow (per conversation):**

```
for turn in range(max_turns):
    user_msg      = UserSim.run(ctx, completion_guidance)
    ctx.add(user_msg)

    assistant_msg = Assistant.run(ctx, remaining_tools, plan_status)
    ctx.add(assistant_msg)

    if assistant_msg.tool_calls:
        for depth in range(max_depth=3):
            tool_msg = ToolExecutor.run(ctx, tool_calls, chain_context)
            ctx.add(tool_msg)
            session.add_response(tool_msg)

            followup = Assistant.run(ctx, ...)
            if followup.empty: break
            if followup.tool_calls and is_last_iter: break  # orphan prevention
            ctx.add(followup)
            if followup.tool_calls: current_msg = followup; continue
            # handle refusal/narration with re-prompt...
            break

    if plan.is_complete(): break

scores = Judge.score(conversation)
if scores.mean < quality_threshold:
    retry with repair hints derived from scores.specific_issues
```

The retry/repair loop wraps this whole flow. On retry, the second attempt's assistant prompt is augmented with the judge's specific issues from attempt 1, turning the failure into a targeted second try.

---

## 2. Context Management Design

The pipeline addresses two distinct context-management concerns:

- **Grounding within a conversation** — so tool calls in step N use real values from step N−1's output.
- **Steering across the corpus** — so the whole dataset doesn't collapse into repetition.

Both are design decisions with explicit tradeoffs.

### 2.1 Within-conversation grounding

Grounding is a three-layer problem, and I have a mechanism for each layer.

**Layer 1 — The mock tool executor must produce chain-consistent responses.** If tool 1 returns `{"hotel_id": "htl_a8f2"}` and tool 2's argument references `hotel_id`, nano generating tool 2's response needs to echo back `htl_a8f2` (or produce a response consistent with that ID being real). Otherwise the "chain" is just independent calls with no linkage.

This is handled by `SessionState` in the simulator. Every time a tool is executed, `session.add_response(response)` walks the response dict recursively and extracts all scalar values into a flat `_store`. The next time nano is asked to generate a mock response, the prompt includes:

```
Available values from previous tool calls:
  - hotel_id: htl_a8f2
  - hotel_name: Hotel du Marais
  - price: 175
  ...
```

Nano sees this and uses the values when generating subsequent responses. Verified empirically: my chaining detector (looks at whether tool N's argument value appears as a value in tool N-1's response) reports 63.9% chaining rate on Run B. That's not 100% because some chains don't have a field-level connection (chain coherence gate lets through cross-domain chains on LLM approval), but for chains that should chain, values flow through.

**Layer 2 — The assistant must see prior tool results when choosing the next call.** This is the standard "grounding via context injection" pattern. `ConversationContext._available_values` tracks flattened values from every tool output and exposes them in the assistant's system prompt:

```
Values from previous tool calls:
  - hotel_id: "htl_a8f2"
  - hotel_name: "Hotel du Marais"
```

Combined with assistant rule #3 ("When chaining tool calls, use EXACT values from previous tool responses — not invented values"), this gives Claude the values it needs and tells it to use them exactly. The judge's chain-break check enforces compliance.

**Layer 3 — The user simulator must NOT see tool outputs.** This is subtle and took debugging to find. Early runs had the user simulator inventing values that "looked right" — saying things when that ID had appeared in a prior tool response but was never mentioned in the assistant's text summary. The user simulator was reading the raw context (which includes tool outputs) and leaking those values into its next user message. The judge correctly flagged it as hallucination because, from the user's perspective, they shouldn't have known that value.

The fix is `build_user_visible_history` in `user_simulator.py`, which filters the context down to just `user` and `assistant.content` messages before feeding it to the user simulator. Raw tool outputs, tool_use blocks, and anything else a real user wouldn't see are excluded. The user simulator is now grounded on the assistant's text summaries only, which is what a real user would actually see.

> **Why three layers and not one?**
>
> Because the three consumers have fundamentally different access rights to the conversation state:
>
> - **The mock executor** needs to see everything from prior tools so it can produce chain-consistent responses.
> - **The assistant** needs to see everything so it can reference prior IDs correctly in the next tool call.
> - **The user simulator must NOT see raw tool outputs.** If I let it see them, it leaks values from tool responses into user messages, breaking realism — a real user only knows what the assistant explicitly told them.
>
> A single shared dict can't enforce these access rules. The three-layer split is the minimum structure I found that actually prevents the leakage bug.

**Tradeoffs:**

- **Coupling cost**: three separate extraction paths (one per consumer). Duplication is real. If I changed the value-extraction logic, I'd touch three places.
- **Risk**: the layers can drift. If the mock executor stores a value under a different key than the assistant sees, chaining breaks silently. I mitigate with shared `_extract_values` implementations across the two assistant/simulator paths, and with the chaining-rate metric in `compare` that would catch a regression in layer 1 or 2.
- **What breaks at scale**: if a conversation has thousands of turns or tool calls, injecting all prior values into the system prompt would explode the context window. I cap it today via the natural limit of `max_turns * max_depth` tool calls per conversation (~15 max), but for longer conversations I'd need retrieval — pull only the values that matter for the next call, not all of them.

### 2.2 Cross-conversation steering

**Goal:** across the corpus, avoid over-representing the same tools, categories, and chain patterns. Without steering, the sampler's weighted-random selection gravitates toward high-quality endpoints (those with rich schemas, strong chain potential), which produces hot tools — a handful that appear in many conversations.

**Implementation.** `DiversitySteering` lives in `conv_gen/memory/steering.py` and exposes two outputs to the sampler:

1. `get_exclude_tools(top_n=10)` — returns tools whose usage is above 3× mean usage. Called before each new sample, used as the sampler's `exclude_tools` constraint. **Counter-based, fully deterministic.**
2. `get_steering_guidance(candidate_tools)` — returns a prose hint that's appended to the scenario generator prompt. Contains:
   - Over-represented tool warnings (counter-based)
   - Under-represented category suggestions (counter-based)
   - "Similar past conversations have been used" hint from mem0 vector memory search (**ANN-backed, not deterministic**)

The architecture deliberately keeps the counter-based path (which drives the *hard* sampling decisions) separate from the mem0 path (which drives a *soft* text hint in one prompt). This isolation is important for the reproducibility story — see Section 6 below.

> **Why counter-based and not just mem0?**
>
> Three reasons:
>
> - **Counters are cheap, deterministic, and directly measure what I want to steer.** O(1) updates, zero inference cost, trivially debuggable.
> - **mem0's vector search returns prose-like "similar conversations" results** — useful as a textual hint for the scenario generator, but the wrong data type for hard filtering decisions.
> - **Mixing the two forces me to answer the wrong question.** "What does it mean for a tool to be vector-similar to a conversation?" is fuzzy. "How often has this tool been used in the recent corpus?" is crisp.
>
> So counters do the hard work and mem0 adds a soft nudge on top.

> **If counters do the hard work, what does mem0 actually add?**
>
> A natural-language similarity signal the counters can't express. When the sampler picks a chain similar to one I've seen before ("search for concerts in NYC then buy tickets"), mem0 can surface the prior conversation and inject a hint into the scenario generator: "similar tool combinations have been used before — create a scenario that's distinct." This pushes the scenario writer toward varied user intent even when the raw chain shape has already been seen. In practice the effect is small but real — I observe slight language diversification in Run B scenarios compared to Run A on the same sampled chains.

**Tradeoffs:**

- **Pro**: counter-based primary signal is fast and deterministic, giving me a clean reproducibility story while preserving most of the diversity benefit.
- **Con**: mem0 adds infrastructure (a vector store, an embedding model) for a relatively modest payoff. If I were at scale, I'd measure more carefully before keeping mem0.
- **What breaks at scale**: the counters are unbounded — they grow linearly with conversation count. At 10K+ conversations this is fine (tens of KB), but the threshold logic (3× mean) has a subtle issue: a tool used rarely but in the beginning of the run has its count dominated by later conversations, so the threshold approaches a moving target. For large corpora I'd want sliding-window counters or time-decayed weights.
- **What else breaks**: steering acts locally — "what have I generated recently?" — not globally — "what does a good corpus look like?" For a truly balanced dataset, I'd want a target distribution (e.g., 5% of conversations use each of the top 200 tools) and a planner that picks each conversation to close the gap to that target. Today's steering is reactive, not planned.

### 2.3 Where grounding breaks down

Known limitations where my grounding approach produces the wrong answer:

1. **Nested response structures.** If a tool returns `{"result": {"items": [{"id": "abc"}]}}`, my walker extracts `"abc"` but loses the path. A subsequent call using `item_id=abc` chains fine, but a call using `result.items[0].id` (as some APIs would specify) does not. I deliberately flatten, accepting the tradeoff that complex nested paths won't chain cleanly.

2. **Ambiguous field names across tools.** If Tool A produces `id` and Tool B needs `id`, I'd match them as chainable. But if Tool A's `id` means "invoice ID" and Tool B's `id` means "customer ID," the chain is semantically wrong. The IDF-based specificity penalty helps (generic `id` gets near-zero weight) but doesn't fully solve it.

3. **Chain breaks across many hops.** After 4+ hops, the assistant sometimes reuses a 2-hops-ago value instead of the most recent one. The chain-break check catches the simplest version (field X was in the most recent response with value V, call uses field X with value W) but not the 2-hops-back case.

4. **Scenarios that describe tasks the tool can't do.** The scenario generator sometimes writes "convert PDF to Word" for an endpoint that actually does "convert TeX to PDF." The direction mismatch is invisible to the sampler (both are "conversion tools"). I mitigate with the LLM coherence check, but it's imperfect because the check reads the endpoint description, not the structural input→output direction.

---

## 3. Prompt Design

Prompts are the connective tissue of this pipeline. Every LLM call is shaped by a prompt, and small wording changes produce large behavior changes.

This section covers:

- The key prompts in the system and the structure behind each
- The reasoning that shaped them
- One iteration that didn't work and what I learned from it

### 3.1 Assistant system prompt (`ASSISTANT_SYSTEM_PROMPT`)

This is the most important prompt in the system.

Eleven rules, each targeting a specific failure mode I observed during development.

Selected rules (see `conv_gen/agents/assistant.py` for the full text):

```
1. ALWAYS use tools when available. Never answer from your own knowledge
   when a tool can provide the answer
3. When chaining tool calls, use EXACT values from previous tool responses
   — not invented values
5. Never ask for API keys, tokens, passwords, or auth credentials. Use
   "default" for auth parameters
7. If the user's intent is unclear, ask ONE short clarifying question
11. NEVER ask the user for system-generated IDs ... fabricate one yourself
    and proceed with the tool call. ...
```

Rule #1 is the anti-shortcut rule. Claude's default instinct is to answer questions from its own knowledge, which defeats the purpose of a tool-use dataset. The rule is deliberately absolute.

Rule #3 is the chain-grounding rule. Combined with the `available_values` injection in the system prompt, it's what produces the 63.9% real chaining rate I see in Run B.

Rule #5 exists because asking for API keys triggers a predictable failure loop: user simulator refuses ("I don't know that"), assistant can't proceed, conversation dies. Rule + structural fix (credential stripping from the tool schema so Claude can't even see the parameter) together close the loop.

Rule #7 supports the disambiguation requirement from the project — the assistant must be willing to ask clarifying questions when the user's intent is unclear.

**Rule #11 is the most load-bearing and took several iterations to land.** The initial version was just "fabricate plausible placeholder IDs." Too vague — Claude sometimes asked the user for IDs anyway. The revised version (currently in place) has four specific sub-rules:

```
a) If a PRIOR tool response contains the ID you need, use it EXACTLY
b) If no prior tool has produced the ID, CREATE one yourself and
   proceed. Do NOT ask the user. Generate a plausible identifier from
   what the user said — e.g. "man_utd_vs_liv_2024" for a match.
   This is expected and correct — treat your own generated ID as
   valid input for the tool call.
c) Once a tool has returned a response, subsequent calls use the ID
   from the response, maintaining consistency across the chain.
d) Ask the user ONLY for values a real person would naturally know:
   names, cities, dates, topics, preferences, emails, URLs.
```

Sub-rule (c) is particularly important: when the assistant fabricates an ID in call 1 and the mock generates a response that echoes it back (or not), call 2 must use whatever's in the response, not the original fabrication. This maintains chain consistency even when fabrication starts the chain.

The verbose phrasing ("This is expected and correct") exists because Claude's default is to refuse when asked to "make up an ID" — it's been trained to not confabulate. Rule #11 has to explicitly grant permission or Claude pattern-matches against its training and refuses.

### 3.2 User simulator prompts (`INITIAL_USER_PROMPT`, `FOLLOWUP_USER_PROMPT`)

The user simulator has two prompts:

- **`INITIAL_USER_PROMPT`** — the first turn of a conversation
- **`FOLLOWUP_USER_PROMPT`** — every turn after the first

Both are structured with numbered rules the simulator must follow.

The follow-up prompt is more complex because it has grounding constraints. Abbreviated:

```
GROUNDING RULES — these are critical:
1. You ONLY know what the assistant explicitly told you in their messages
2. If the assistant did not state a value, you do NOT know that value
3. NEVER use IDs, names, prices, or numbers the assistant did not tell you
4. If you need a value the assistant has not provided — ASK for it

CONVERSATION RULES:
5. Ask ONE thing at a time
6. If the assistant asked you a question — answer with just the value
7. Do NOT repeat or echo values back to the assistant
8. Do NOT say goodbye or wrap up unless completion status says
   ALL TASKS COMPLETE
9. Keep to 1 sentence. Maximum 2 if providing a specific value.
```

The rules are numbered (not bulleted) because I observed better compliance with numbered instructions — Claude appears to treat numbered lists as explicit checklist items.

The grounding rules (1–4) exist because of the Layer 3 issue above — without them, the user simulator leaks values from tool outputs it shouldn't see. I reinforce this structurally by using `build_user_visible_history` to filter the context, but the rules provide a second layer of defense.

Rule 8 uses an explicit `completion_guidance` status passed from the orchestrator:

```
STATUS: IN PROGRESS (2/4 tasks done).
2 task(s) still to complete.
Keep the conversation going naturally.
Do NOT say goodbye until every task is done.
```

Vs:

```
STATUS: ALL TASKS COMPLETE.
Say a natural thank you and goodbye in one sentence.
Do NOT ask for anything else.
Do NOT introduce new topics.
```

The concrete task counts (`2/4 tasks done`) are much more effective than vague status phrases ("plan in progress"). Early versions used prose status and the user simulator would wrap up early after the first 2 calls of a 4-call plan. With concrete counts, the simulator correctly keeps the conversation going.

### 3.3 Scenario generator prompt (`SCENARIO_PROMPT`)

Before any turn-taking begins, the scenario generator writes a natural-language user scenario for the sampled tool chain.

The prompt receives the chain (as API capability descriptions, not tool names) and produces 2–4 sentences describing what a real person would want to accomplish and why.

Key design choices:

- **No tool names in the output.** The scenario describes what the user wants, not which API they'll use. Tool/API names come only from the assistant.
- **Filter out auth/infra parameters** when describing the chain to the generator. Otherwise the scenario would mention "API key" as something the user provides.
- **Optional chaining instruction.** For `multi_step_chain` and `full_workflow` types, I add a clause asking the generator to ensure each step depends on the previous. For `parallel_lookup`, I don't — steps should be independent.
- **Optional disambiguation variant.** ~40% of the time I include "leave ONE subjective preference vague" so the assistant has something to clarify. This gives me the natural disambiguation the project requires without making every scenario ambiguous.

### 3.4 Judge prompt (`JUDGE_SYSTEM_PROMPT`)

The judge uses chain-of-thought plus few-shot examples to calibrate scoring.

Structure:

1. **Scoring rubric** — 1–5 scale with concrete anchors for each dimension
2. **Critical checks** — mechanical failures that cap specific dimensions (hallucinated values, parameter misuse, incoherent workflow, scenario mismatch, no tool calls)
3. **Few-shot examples** — three concrete conversations with score explanations, one at each quality tier (5/5/5, 3/2/3, 2/1/1)
4. **CoT instruction** — "For EACH dimension, first write your reasoning (what you observed), THEN assign the score. Reasoning must come before the score — do not decide the score first."

The CoT order matters: asking for reasoning first, score second pushes the LLM to actually examine the conversation rather than pattern-matching to a score and rationalizing. I tested the reverse (score first, reasoning after) and observed noisier scores with less-connected reasoning.

The few-shot examples are spread across quality tiers deliberately. Early versions used only 5/5/5 examples thinking "anchor to the good case," which produced inflated scores — the judge rarely gave anything below 4. Adding explicit mediocre and bad examples fixed the calibration.

**Output format is JSON mode.** The judge returns:

```json
{
    "naturalness_reasoning": "...",
    "naturalness": <float 1-5>,
    "tool_correctness_reasoning": "...",
    "tool_correctness": <float 1-5>,
    "task_completion_reasoning": "...",
    "task_completion": <float 1-5>,
    "specific_issues": ["..."]
}
```

The `specific_issues` field is what makes retry/repair work. When a conversation scores below threshold, I extract this field and pass it to the next attempt's assistant prompt as explicit instructions: "FIX THESE SPECIFIC ISSUES: <list>". This turns a generic retry into a targeted retry.

### 3.5 LLM coherence check prompt (sampler)

For the chain quality gate, I use nano to approve or reject soft chains.

The prompt is short and targets specific failure modes:

```
You are checking whether a sequence of API calls from different services
can actually work together as a chain in one conversation.

Chain: {formatted_chain}

Answer NO if ANY of these are true:
- The tools come from different unrelated platforms/vendors that wouldn't
  share identifiers (e.g. Upcall webhook vs Viber webhook — different
  vendors, different IDs).
- The output of one step cannot realistically feed the input of the next
  step (e.g. a YouTube video list doesn't give you Google Maps place IDs).
- A real user would need external manual steps (copy-paste across
  systems, different logins) to bridge one tool to the next.
- The chain mixes two unrelated tasks with only a thin narrative bridge.

Answer YES only if every step's output can flow into the next step's
input AND a single user would genuinely do all of these in one session
on the same platform ecosystem.

Answer with just one word: YES or NO.
```

I started with a more general prompt ("does this sound like a realistic workflow?") and it was too permissive — chains mixing Upcall + Viber + SIGNL4 (three different webhook providers) got approved because the narrative sounded coherent. The revised prompt lists specific failure modes the reviewer should check, which shifts nano's behavior from "does this sound plausible at the task level?" to "does this work at the platform/data-flow level?" The latter is what I actually need.

**Single-word output.** I ask for exactly YES or NO to keep the API call small and latency low. Parsing is trivial: `answer.startswith("YES")`.

### 3.6 The iteration that didn't work: few-shot hotel examples

Not every prompt change helps. This one actively hurt the dataset, and the postmortem taught me something about debugging at the right layer.

**The idea.** Mid-session, tool_correctness was fluctuating around 2.6–3.4 (below my 3.5 target). I added few-shot examples to the `INITIAL_USER_PROMPT` and `FOLLOWUP_USER_PROMPT` showing a good and bad user message side-by-side. The examples were hotel-specific:

```
GOOD EXAMPLE:
"Can you find me hotels in Paris for next weekend under €200/night?"

BAD EXAMPLE (do not do this):
"I need to check my flights, book a hotel, and get the weather
and convert currency please."
← Too many requests at once.
```

The reasoning seemed solid: show Claude what good looks like, show what bad looks like, anchor its behavior.

**What happened.** Over the next run (10 conversations with seed 42), MS+MT dropped from 60% to 30%. Worse, the ones that hit MS+MT had lower tool_correctness than before (3.3 → 2.8). Naturalness held flat.

I couldn't figure out what went wrong until I looked at individual conversations. Claude was trying to force hotel/travel language onto chains that had nothing to do with travel. A scenario about "set up a security monitoring system" had the user simulator writing "I need to find a security hotel" (literal phrasing from the example). A chain involving Spotify and Goodreads had user messages like "book me a spotify for tonight". The few-shot pattern had completely dominated Claude's output shape.

**What I learned:**

1. **Few-shot examples are domain-contagious.** When the example is from domain X and the actual task is in domain Y, the model doesn't cleanly transfer — it tries to impose domain X phrasing on domain Y content. This is obvious in retrospect but I expected Claude to be more robust.

2. **Few-shot works when examples are (a) domain-generic or (b) drawn from multiple domains.** A single example from one domain is worse than no example at all. I reverted the change and the problem went away.

3. **The underlying issue I was trying to fix wasn't a prompt issue.** tool_correctness was low because of the judge's structural hallucination check firing on legitimate rule #11 fabrications — a scoring issue, not a generation issue. Adding few-shot prompts was the wrong layer. The correct fix came later: replacing the hallucination check with a field-name-aware chain-break check (see 1.2 Judge section) that distinguishes real chain breaks from legitimate fabrications.

4. **Small prompt changes can produce large behavior shifts.** I expected ±5% on MS+MT from the few-shot change. I got -30% and it broke a lot of the dataset. Always A/B test prompt changes, never assume they'll help.

The broader lesson: **debug at the right layer.** When a metric is low, ask why before changing anything. The "why" points at which subsystem to modify. In this case the metric was low because of a scoring quirk in the judge, not a generation quality issue in the agents — so changing the generation prompts didn't help and created new problems. Going forward, I always trace a metric drop back to specific failing conversations before reaching for a prompt change.

---

## 4. Diversity & Quality Analysis

The project requires generating the dataset twice with the same seed — once with cross-conversation steering disabled (Run A) and once enabled (Run B) — and measuring how steering changes the output.

This section covers:

- Which metrics I chose and why
- The results for both runs
- The diversity–quality tradeoff (and why I don't actually see one)

### 4.1 Metrics chosen

I use two primary diversity metrics, chosen because they directly measure what my steering mechanism is **designed to change**:

**Primary #1 — `tool_usage_entropy`**. Normalized Shannon entropy over the frequency distribution of individual tool calls.

```
H(p) = -Σ p_i × log₂(p_i)
normalized = H(p) / log₂(N)
```

where `p_i` is the fraction of all tool calls that went to tool i, and N is the number of distinct tools used. Range [0.0, 1.0]: 1.0 = every tool used equally often (maximum flatness), 0.0 = one tool dominates all calls.

> **Why this captures what steering is designed to change:**
>
> My steering mechanism's first effect is to exclude recently-used tools from sampling (`get_exclude_tools` returns tools above 3× mean usage). That directly flattens the frequency distribution — hot tools appear less often, which raises the entropy. If my steering is working, Run B should have strictly higher entropy than Run A.

**Primary #2 — `unique_tool_coverage_ratio`**. The fraction of the tool registry that was exercised.

```
coverage = |unique_tools_used| / |registry_size|
```

where `registry_size` is the total number of `(tool, api)` endpoints in the registry (11,302 in my build). Range [0.0, 1.0]: higher means the dataset touched more of the available tools.

> **Why this captures what steering is designed to change:**
>
> The second effect of my steering is to push the sampler away from hot endpoints and toward parts of the graph it wouldn't otherwise visit. This metric directly counts how much of the registry was actually explored. If my steering is working, Run B's coverage should be strictly higher than Run A's.

**These two metrics are orthogonal.** A dataset can have high entropy within a small set of tools (flat but narrow) or high coverage with a few tools dominating (broad but skewed). Reporting both separates the "flatness" question from the "breadth" question. My steering is intended to improve both, so I want to see both improve.

Five supplementary diversity metrics also reported but not primary:

- `top_5_tool_concentration` — fraction of all calls going to the top 5 tools. Inverse view of flatness; easier to interpret for "is one tool dominating?"
- `tool_combination_entropy` — Shannon entropy over unique chain signatures (the sorted set of tools used per conversation). Chain-level diversity rather than tool-level.
- `domain_coverage_uniformity` — normalized Shannon entropy over category usage. Tracks whether categories are evenly covered.
- `unique_chain_ratio` — unique chain signatures / total conversations. 1.0 = every conversation has a distinct chain.
- `unique_chain_combinations` — raw count of distinct chains.

Quality metrics come from the LLM-as-judge (three dimensions + overall mean) plus two project-compliance metrics:

- `mean naturalness` — does the conversation feel like a real user-assistant exchange?
- `mean tool_correctness` — are tools used correctly, with valid arguments, with chain values?
- `mean task_completion` — was the user's goal met?
- `mean overall` — average of the three
- `ms_mt_rate` — fraction of conversations with ≥3 tool calls AND ≥2 distinct tools (project requires 50–60%)
- `real_chaining_rate` — fraction of conversations where a later call uses a value from an earlier response (actual output→input data flow)

### 4.2 Results for both runs

**Run configuration (both runs identical except for steering flag):**

```bash
python3 cli.py generate --seed 42 --count 150 --no-cross-conversation-steering \
    --output output/run_no_steering.jsonl
python3 cli.py generate --seed 42 --count 150 \
    --output output/run_with_steering.jsonl
python3 cli.py compare output/run_no_steering.jsonl output/run_with_steering.jsonl \
    --label-a "Run A (no steering)" --label-b "Run B (steering)"
```

Both runs used seed 42, 150 conversations, identical prompts and registry. The only difference is the `--no-cross-conversation-steering` flag on Run A.

**Quality metrics (mean across all scored conversations):**

| Metric           | Run A | Run B | Δ (B − A) |
| ---------------- | ----- | ----- | --------- |
| naturalness      | 4.507 | 4.527 | **+0.020** |
| tool_correctness | 3.667 | 3.650 | −0.017    |
| task_completion  | 4.043 | 4.120 | **+0.077** |
| overall_mean     | 4.072 | 4.099 | **+0.027** |

**Project compliance:**

| Metric                                  | Run A  | Run B  | Δ            |
| --------------------------------------- | ------ | ------ | ------------ |
| MS+MT rate (≥3 calls, ≥2 distinct tools) | 56.7%  | 62.0%  | **+5.3 pp**  |
| real chaining rate                      | 58.0%  | 60.0%  | **+2.0 pp**  |

Both runs land in the project's 50–60% MS+MT window, and Run B pushes just above it at 62%.

**Primary diversity metrics (the ones steering is designed to change):**

| Metric                                             | Run A   | Run B   | Δ             |
| -------------------------------------------------- | ------- | ------- | ------------- |
| tool_usage_entropy (higher = flatter)              | 98.78%  | 99.14%  | **+0.36 pp**  |
| unique_tools_used (distinct tools)                 | 182     | 190     | **+8**        |
| unique_tool_coverage_ratio (of 500 tools)          | 36.40%  | 38.00%  | **+1.60 pp**  |

Both primary metrics move in the intended direction. Entropy is already very high in Run A (98.78%), so the absolute delta is small, but it represents real flattening — a few more unique tools, fewer hot-tool hits. Coverage grows by 8 distinct tools, from 36.40% of the registry to 38.00%.

**Secondary diversity metrics:**

| Metric                                    | Run A  | Run B  | Δ             |
| ----------------------------------------- | ------ | ------ | ------------- |
| top_5_tool_concentration (lower = better) | 5.25%  | 4.03%  | **−1.22 pp**  |
| tool_combination_entropy (chain-level, bits) | 7.215  | 7.175  | −0.040     |
| domain_coverage_uniformity                | 77.68% | 79.25% | **+1.58 pp**  |
| unique_chain_ratio                        | 99.33% | 98.00% | −1.33 pp      |
| unique_chain_combinations                 | 149    | 147    | −2            |

Top-5 concentration drops meaningfully (−23% relative) — the exclude-tools signal is doing its job. Domain coverage gets meaningfully more uniform. Chain-level metrics move slightly the "wrong" way, which I attribute to steering pushing the sampler into repeated category/tool combinations as it avoids the hot tools — a second-order artifact of the steering mechanism being tool-level rather than chain-level.

**Per-type MS+MT rate:**

| Type                                   | Run A    | Run B    | Run B rate |
| -------------------------------------- | -------- | -------- | ---------- |
| quick_lookup (not eligible by design)  | —        | —        | —          |
| parallel_lookup                        | 1 / 15   | 4 / 15   | 27%        |
| simple_sequential                      | 1 / 15   | 2 / 15   | 13%        |
| multi_step_chain                       | 46 / 60  | 47 / 60  | 78%        |
| full_workflow                          | 37 / 45  | 40 / 45  | 89%        |

The MS+MT workhorses (`multi_step_chain` and `full_workflow`) hit 78% and 89% on Run B. `parallel_lookup` quadrupled from 1 to 4 hits (still low — Claude prefers sequential tool calls even with parallel hints; this is a known limitation discussed in §8). `simple_sequential` stays low by design (it's a 2-tool type and MS+MT requires ≥3 tool calls; hits come only when the assistant makes extra calls on its own).

**Tool overlap between runs:**

| Metric                        | Value |
| ----------------------------- | ----- |
| Tools used in BOTH runs       | 155   |
| Tools only in Run A           | 207   |
| Tools only in Run B           | 221   |
| Total unique across both      | 583   |

Only 155 tools appear in both runs. 207 are unique to Run A and 221 are unique to Run B — meaning the two runs barely overlap at the tool level. Together they exercise 583 distinct tools, which is ~61% more coverage than either run alone. This is the clearest evidence that steering doesn't just flatten the same distribution — it actually navigates to different parts of the registry.

### 4.3 Diversity–quality tradeoff analysis

The expected story is that steering improves diversity at the cost of quality. That's **not what I observe**.

The 150-conversation runs above tell a consistent story:

1. **Primary diversity metrics improve.** Tool usage entropy goes up (+0.36 pp), unique tools used goes up (+8), coverage ratio goes up (36.40% → 38.00%, +1.60 pp). All three move in the direction I'd expect from the steering mechanism's design intent.

2. **Secondary diversity metrics also improve.** Top-5 concentration drops by 23% relative (5.25% → 4.03%), domain coverage uniformity goes up (77.68% → 79.25%). Chain-level entropy and unique chain ratio drift very slightly the other way — a second-order effect of tool-level steering pushing the sampler into repeated chain combinations as it avoids hot tools.

3. **Quality metrics do not degrade — most actually improve.** Naturalness is +0.020, task_completion is +0.077, overall mean is +0.027. Only `tool_correctness` moves the "wrong" way and by a tiny amount (−0.017). The tradeoff I'd expected — steering pushing the sampler into harder-to-execute chains that Claude struggles with — doesn't happen in my measurement range.

4. **MS+MT rate goes up, not down.** 56.7% → 62.0%, a +5.3 point gain. Counterintuitively, with steering the MS+MT rate is *higher* than without. This is the most interesting finding and I want to address it head-on.

5. **The two runs barely overlap at the tool level.** Only 155 tools appear in both runs out of 583 total distinct tools exercised across the pair. Steering isn't flattening the same distribution — it's visiting a genuinely different part of the registry.

> **Why does steering boost MS+MT instead of hurting it?**
>
> My hypothesis: without steering, the sampler gravitates toward "easy" hot tools — endpoints with high schema quality and strong chain potential at the graph level. Those same hot tools appear in many conversations and produce repetitive chains. Repetitive chains are structurally valid but often **collapse to 2 calls rather than 3+**, because the assistant can complete the user's request with fewer steps when the chain is familiar and shallow.
>
> When I turn steering on, hot tools get excluded once they've been used a few times. This pushes the sampler toward less-obvious endpoints where the multi-step workflow is more central to the task — and those chains tend to land MS+MT more reliably because they actually need the full sequence to complete the user's goal.
>
> This is the opposite of the expected pattern. **Diversity and quality are not in opposition in my measurement range.** If I pushed steering much harder (say 3× stricter exclusion), I'd eventually hit a point where the sampler is forced into genuinely poor-quality chains — but at my current setting (3× mean usage threshold, ~10 top tools excluded), I'm in the regime where increased diversity and increased quality move together.

**Caveats for interpreting these numbers:**

- **Sample size.** 150 conversations is a small sample for metric comparisons. The deltas I report are point estimates, not distributions. I don't have error bars. A rigorous evaluation would run each configuration 3–5 times and report mean ± stdev.
- **Single seed.** Both runs use seed 42. Different seeds would produce different mixes of sampled chains. Generalizing from one seed's outcome to "steering always does X" would overclaim.
- **Non-determinism from mem0 ANN.** Run B has a small amount of non-determinism from mem0's vector search in `get_steering_guidance`. See section 6 for details on how this affects reproducibility
- **No causal isolation.** I can't separate the effect of counter-based exclusion from mem0-based textual hints, because they're bundled in the same `steering=True` mode. A proper ablation would turn each off independently.

For a project-grade claim, the honest framing is: **on this specific 150-conversation experiment with seed 42, steering improves diversity and slightly improves quality.** Whether that generalizes requires more trials.

---

## 5. Output Format & Metadata Schema

### 5.1 Wire format

Each generated conversation is written as one line of JSONL in the wire format.

A record looks like:

Example (abridged from an actual Run B conversation, `conv_35b34978`):

```json
{
  "conversation_id": "conv_35b34978",
  "messages": [
    {
      "role": "user",
      "content": "Hi, I need to pull up all the accepted email messages from our recent marketing campaign so I can review which ones got positive responses from potential clients."
    },
    {
      "role": "assistant",
      "content": "I can help you retrieve the accepted email messages. Could you please provide the email domain for your marketing campaign?"
    },
    {
      "role": "user",
      "content": "The domain is marketingcampaign.company.com."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "endpoint": "MailGun/getAcceptedMessages",
          "arguments": {"domain": "marketingcampaign.company.com"}
        }
      ]
    },
    {
      "role": "tool",
      "content": {
        "messages": [
          {"messageId": "msg_8f2e3d4b5a", "recipient": "lead3@example.com",
           "deliveryStatus": "accepted", "tags": ["hot_lead", "priority"]}
        ]
      }
    },
    {
      "role": "assistant",
      "content": "I retrieved 3 accepted messages. The strongest lead is lead3@example.com (hot_lead, high engagement)."
    },
    {
      "role": "user",
      "content": "Forward the lead3 response to john@company.com and lisa@company.com with a note explaining why this lead should be prioritized."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "endpoint": "OutlookMail/createDraftForwardMessage",
          "arguments": {
            "messageId": "msg_8f2e3d4b5a",
            "recipients": ["john@company.com", "lisa@company.com"],
            "comment": "High-priority lead — strong engagement metrics, prioritize follow-up."
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": {"draftId": "draft_9c1f", "status": "draft_created"}
    },
    {
      "role": "assistant",
      "content": "I've created the forward draft to john@company.com and lisa@company.com with your prioritization note."
    },
    {
      "role": "user",
      "content": "Thanks, that's exactly what I needed."
    }
  ],
  "judge_scores": {
    "naturalness": 5.0,
    "tool_correctness": 5.0,
    "task_completion": 5.0,
    "reasoning": "Natural exchange, both tools used correctly, IDs chained between calls, task fully completed."
  },
  "metadata": {
    "seed": 42,
    "conversation_type": "simple_sequential",
    "pattern": "sequential",
    "steering_enabled": true,
    "scenario": "Sarah is a marketing manager who needs to review all accepted email messages in her company's marketing campaign domain to identify which ones received positive responses. After finding the accepted messages, she wants to forward the most promising response to her sales team members with a comment explaining why this lead should be prioritized.",
    "chain": [
      "MailGun/getAcceptedMessages",
      "OutlookMail/createDraftForwardMessage"
    ],
    "tools_used": [
      "MailGun/getAcceptedMessages",
      "OutlookMail/createDraftForwardMessage"
    ],
    "num_turns": 11,
    "tools_planned": 2,
    "tools_executed": 2,
    "num_retries": 0,
    "categories": {
      "MailGun/getAcceptedMessages": "Email",
      "OutlookMail/createDraftForwardMessage": "Email"
    },
    "categories_list": ["Email"],
    "plan_summary": {
      "status": "completing",
      "turn": 3,
      "message_count": 11,
      "progress": 1.0,
      "planned_tools": [
        "MailGun/getAcceptedMessages",
        "OutlookMail/createDraftForwardMessage"
      ],
      "completed_tools": [
        "MailGun/getAcceptedMessages",
        "OutlookMail/createDraftForwardMessage"
      ],
      "tools_remaining": [],
      "unplanned_tools": [],
      "clarification_pending": false
    }
  }
}
```

Note the chain flow: Tool 1 returns `messageId: "msg_8f2e3d4b5a"`, Tool 2 uses that exact ID as its `messageId` input — real output→input chaining visible in the JSONL.

**Shape rules:**

- **Tool calls** are `{"endpoint": "<tool>/<api>", "arguments": {...}}` — a single `endpoint` string combining tool and API names, plus a typed argument dict.
- **Tool messages** hold the response dict directly in `content`. No separate `tool_outputs` list.
- **Parallel tool calls** expand to multiple consecutive tool messages (one per call), in the same order as `tool_calls` on the preceding assistant message. This preserves 1-to-1 call/response pairing in a linear format.
- **Top-level `tool_calls` and `tool_outputs` arrays are NOT emitted.** The internal Pydantic model has these as aggregated convenience fields, but they're redundant with the inline representation and the wire format omits them.

The loader `from_any_json` auto-detects wire format vs. legacy Pydantic format (older runs before this format was standardized). Wire-format records have no top-level `tool_calls` / `tool_outputs` keys; legacy records do. The discrimination is structural, not versioned.

### 5.2 Metadata schema

The `metadata` dict carries everything a reviewer or downstream consumer needs to reproduce, filter, or analyze the dataset.

Every field:

| Field | Type | Meaning | Required? |
|---|---|---|---|
| `seed` | int | Random seed used for this run | Yes |
| `conversation_type` | str | Director type: `quick_lookup` / `parallel_lookup` / `simple_sequential` / `multi_step_chain` / `full_workflow` | Yes |
| `pattern` | str | Sampling pattern: `sequential` / `parallel` / `single` | Yes |
| `steering_enabled` | bool | Whether cross-conversation steering was active for this run | Yes |
| `scenario` | str | The natural-language scenario the generator wrote | Yes |
| `chain` | list[str] | Planned chain as `["tool/api", "tool/api", ...]` in order | Yes |
| `tools_used` | list[str] | Tools actually called during the conversation | Yes |
| `num_turns` | int | Number of messages in the conversation | Yes |
| `tools_planned` | int | Length of the planned chain | Yes |
| `tools_executed` | int | Number of unique tools actually executed | Yes |
| `num_retries` | int | How many retry attempts the orchestrator made before accepting this conversation | Yes |
| `categories` | dict[str, str] | Map from `"tool/api"` → category name | Yes |
| `categories_list` | list[str] | Unique categories touched | Yes |
| `plan_summary` | dict | Final `ConversationPlan` state with completed/remaining tools, progress %, clarification flag | Yes |

**Filtering examples.** The schema supports:
- `seed == 42` → all conversations from a specific run
- `steering_enabled == true` → Run B only
- `conversation_type == "multi_step_chain"` → just the multi-step type
- `"Email" in categories_list` → conversations touching Email tools (the example above would match)
- `num_retries > 0` → conversations that hit the retry/repair loop
- `tools_executed >= 3` → Multi-step compliant (combined with `len(set(tools_used)) >= 2` for MS+MT)

**Reproducing a specific conversation.** Given a record, a reviewer can:
1. Look at `metadata.seed` to know the random seed
2. Use `metadata.chain` to see exactly which tools were meant to be called
3. Use `metadata.scenario` to see the scenario text that drove it
4. Use `metadata.plan_summary` to see the plan's final state

---

## 6. Reproducibility & Determinism

The project asks explicitly about ANN non-determinism since I use mem0.

Here's the honest story.

### 6.1 What's deterministic

- **Run A (steering disabled).** Seed 42 produces identical conversations across runs, modulo any stochastic LLM outputs (which are not under my control — Anthropic and OpenAI both have some run-to-run variance even at temperature 0).
- **All metric computations.** Given a fixed JSONL file, `cli.py evaluate` and `cli.py compare` produce exactly the same numbers on every invocation. The metric code is pure — no LLMs, no randomness, no vector search.
- **The tool graph.** Built once, saved to `.cache/tool_graph.pkl`, and deterministic on load. The build itself uses an embedding computation (sentence-transformers) which is mostly deterministic on fixed hardware but may have small floating-point drift across architectures.
- **The sampler's `get_exclude_tools` signal.** Reads a plain Python `Counter` of tool-usage frequencies. `Counter.most_common()` is deterministic on insertion order.

### 6.2 What's non-deterministic

- **mem0's vector search in `get_steering_guidance`.** When steering is enabled, mem0 searches for "similar past conversations" to the current chain and returns the top-K neighbors. ANN algorithms (HNSW, the default) are not guaranteed to return identical results across runs — index construction order, floating-point arithmetic, and internal library state all contribute.

- **Run B conversations have small textual drift.** The ANN variance affects a single component of the scenario generator's prompt: the "similar past conversations have been used before" hint. Different neighbors → slightly different hint text → slightly different scenario → slightly different conversation. The propagation is dampened by several LLM layers so the net effect is small, but real.

### 6.3 How I handle this in the metrics

My "handling" is structural, not compensatory:

1. **Metric computation is deterministic.** Given a fixed JSONL, numbers are reproducible 100%. No ANN in the compute path.
2. **The primary steering signal is counter-based.** `get_exclude_tools` reads `_usage_counts` — a plain Counter — not mem0. This is what the sampler's hard filter reads, so the biggest impact of steering on the generated chains is deterministic.
3. **The ANN path is confined to a textual hint.** mem0 is queried only for a prose "similar past conversations" hint. That hint joins the scenario generator's guidance string but doesn't change which chain is sampled.
4. **mem0 failures fall back gracefully.** The `self.memory.search()` call is wrapped in try/except. If mem0 is unavailable or misbehaves, I silently fall back to counter-only steering, which is fully deterministic.
5. **The `--no-cross-conversation-steering` flag gives a fully deterministic escape hatch.** Run A uses this and is bit-exact reproducible (modulo LLM non-determinism I inherit from Anthropic/OpenAI).
6. **The ANN variance is not quantified.** I didn't run Run B multiple times to empirically measure the spread. The analysis in section 4.3 reports point estimates from one trial of each configuration. For rigor, a reviewer running their own Run B should expect small variation (maybe ±1 percentage point on MS+MT, ±0.005 on entropy) that reflects ANN noise, not pipeline regression.

### 6.4 What I don't do (and honest caveats)

- **No explicit `--deterministic-steering` flag** that disables mem0 but keeps counter-based steering active. Would be useful for a "steering enabled but bit-exact reproducible" mode. I considered adding it and decided against for the project scope; it's a straightforward 20-line addition if wanted.
- **No multi-trial reporting.** My metrics are single-trial point estimates. Confidence intervals or trial-over-trial variance would strengthen the claims.
- **No run_id / generated_at / generator_version metadata.** A reviewer loading a JSONL file can't tell which code version produced it or when. Useful for a production pipeline but not strictly required by the project.
- **No pinned mem0 embedding model.** mem0 defaults to its configured embedding provider; if that changes, neighbor rankings shift. For reproducibility across environments, you'd want to pin the embedder version.

**My honest assessment.** For the project, the reproducibility story is: *"Run A is fully deterministic at the Python seed layer. Run B has small ANN-driven variance from mem0's textual-hint path, confined to the scenario generator's prompt and propagated through several downstream LLM layers. The primary steering signal — the one that shapes which chains are actually sampled — is counter-based and deterministic. Metrics computed on fixed JSONL files are reproducible exactly.

---

## 7. Model Choices

Three LLMs do different jobs in the pipeline, and the split is deliberate.

| Model | Used by | Why |
|---|---|---|
| **Claude Sonnet 4** | User Simulator, Assistant, Scenario Generator | Native `tool_use` structured output with input_schema validation, strong at natural conversation, reasonable latency |
| **GPT-4o** | Judge | Reliable JSON mode, strong at rubric-based evaluation, widely used so calibration is predictable |
| **GPT-4.1-nano** | Mock Executor, Schema Enricher, Chain Coherence Checker | Very low cost per call, good enough for JSON generation and yes/no judgments |

> **Why Claude for generation and OpenAI for scoring and mocks?**
>
>
> 1. **Structured output guarantees differ.** Claude's native `tool_use` gives me server-side schema validation against my own `input_schema` — the strongest structured-output contract available today. OpenAI's JSON mode is good but not schema-validated (the model can emit JSON with wrong field types). For the assistant, where every tool call must be parseable and a malformed call breaks the chain, I wanted the strongest contract. For the judge, where the schema is simple and failures are recoverable, OpenAI JSON mode is sufficient.
>
> 2. **Separation of generator and evaluator reduces self-scoring bias.** When a model evaluates its own outputs, it tends to be favorable toward them — they share training data, prompt-following habits, and failure modes. Using different model families breaks that correlation: Claude generates, GPT scores. I can't quantify the bias reduction exactly, but the qualitative intuition (and industry practice) is that cross-family judging is more trustworthy than self-judging.

**Cost.** A full 150-conversation generate + evaluate cycle costs roughly:
- Claude Sonnet: ~$0.20 (user sim + assistant + scenarios)
- GPT-4.1-nano: ~$0.05 (~400 mock calls + coherence checks + schema enrichment)
- GPT-4o: ~$0.15 (150 judge calls)
- **Total: ~$0.40 per 150-conversation run.**

**Tradeoffs I didn't explore:**
- Running everything on Claude (no OpenAI dependency). Would require an Anthropic JSON mode for the judge; possible but not tested.
- Running everything on OpenAI (no Anthropic dependency). Would require abandoning native tool_use; GPT-4o's function calling is comparable but the structured-output contract is slightly weaker.
- Using open-source models (Llama, Mistral). Would reduce API costs to zero but require self-hosting.

---

## 8. Known Limitations & What I'd Do Next

Reading the Run A / Run B numbers honestly, here's where the pipeline is weakest and what I'd try next. These are genuine open questions — the fact that I haven't implemented them means I'm not certain they'd help, but each one is motivated by something I actually see in the results.

- **`tool_correctness` is the weakest judge dimension** (3.65 in both runs, vs naturalness 4.53 and task_completion 4.12). Argument grounding is where the pipeline loses most of its score. I could try tightening the field-name-aware chain-break check to cover references that point more than one hop back in the history. *Why it might help:* the current check only compares against the immediately preceding response, so chains that reuse a value from an earlier step slip through, and the judge catches them as mismatches. A multi-hop-aware check could turn some of those false-positive chain breaks into legitimate reuse.

- **`tool_correctness` dipped slightly with steering** (3.667 → 3.650). It's a small delta, but it's the *only* metric that moved the "wrong" way. I suspect steering is pushing the sampler toward colder endpoints whose response schemas are lower-quality, and the assistant misuses them more often. I could reweight `schema_quality` in the graph's confidence formula to prefer well-described endpoints even under exclusion. *Why it might help:* it would decouple "has this tool been used recently?" from "can Claude actually figure out how to call this tool?"

- **Registry coverage is only 36–38%** of the 500-tool registry in a single run. Even combining both runs I only touch ~37% of distinct tools (though 155/583 of endpoints appear in both). I could add a target-distribution planner that tracks under-represented tools and nudges the Director toward them. *Why it might help:* today's steering is reactive ("avoid hot tools"), but "avoid hot" doesn't automatically mean "visit cold" — a target-based planner would close that gap directly instead of hoping exclusion is enough.

- **`real_chaining_rate` sits at 60%** — I could require at least one `hard` feeds_into edge on every sampled chain, rejecting soft-only chains even if the LLM coherence gate approves them. *Why it might help:* the chaining-rate ceiling seems to be driven by soft chains where the narrative hangs together but no field actually connects the calls. A harder structural requirement would move the floor up at the cost of some cross-domain diversity.

- **`parallel_lookup` MS+MT is 27%** (4/15 in Run B) — by far the weakest type. Claude ignores the `parallel_hint` and calls tools sequentially. I could rewrite the user-simulator prompt for this type to have the user explicitly ask for multiple things in one message, suspending the "ask ONE thing at a time" rule. *Why it might help:* the real cause isn't the assistant's refusal to parallelize, it's the user simulator never giving the assistant a multi-part question to parallelize *on*. Fixing it at the user layer addresses the root.

- **`unique_chain_ratio` dipped with steering** (99.33% → 98.00%). Tool-level exclusion forces the sampler into slightly repeated chain shapes. I could add a chain-signature counter alongside the tool counter, so steering avoids re-walking the same `(sorted tool set)` — not just the same individual tools. *Why it might help:* the current steering signal is tool-level, but "chain diversity" is a set-level property. Matching the signal to the property should fix the drift without giving up the tool-level gains.

- **Single-trial point estimates.** Every number reported is from one run of each config — no error bars, no confidence intervals. I could run each configuration 3–5 times with different seeds and report mean ± stdev. *Why it might help:* some of the deltas I cite (especially the ±0.02 quality moves) are small enough that I can't rule out seed noise. Multi-trial reporting would tell me which findings are signal and which are within the noise floor.

- **No causal isolation between counter-based and mem0-based steering.** They're bundled as one flag. I could add a `--steering-mode=counter|mem0|both` and run the three-way ablation. *Why it might help:* mem0 adds infrastructure cost (a vector store, an embedder, ANN non-determinism). If the counter-only mode is within 90% of the "both" mode's gains, I'd drop mem0 entirely and keep a fully deterministic pipeline — which would also simplify the reproducibility story in Section 6.

- **The sampler has no lookahead.** It picks the locally-best next hop without checking whether that hop leads anywhere. I notice a few chains in Run B that end earlier than the Director asked for, and I think this is the cause. A one-step lookahead (check the next hop has at least one valid continuation) could shore this up. *Why it might help:* it would close the gap between `tools_planned` and `tools_executed` without changing any scoring or prompts — a pure structural improvement.

- **Chain coherence gate has domain-specific blind spots** — sports APIs that demand fixture IDs the user can't know, image APIs requiring base64 payloads, auth APIs expecting tokens. These chains get approved structurally, then die in the turn loop when the user can't provide what the tool wants. I could filter at selector time any endpoint whose required params are entirely "system-generated IDs a real user wouldn't have," and separately detect "binary input" parameters. *Why it might help:* the current retry/repair loop burns API calls trying to rescue these chains and often fails. Catching them before the turn loop starts would save cost *and* lift `task_completion` by removing a class of chains that always score low.

---

## Appendix: Running the pipeline

See `README.md` for the end-to-end quick-start. In short:

```bash
# one-time build
python3 cli.py build

# generate Run A and Run B
python3 cli.py generate --seed 42 --count 150 \
    --no-cross-conversation-steering --output output/run_a.jsonl
python3 cli.py generate --seed 42 --count 150 --output output/run_b.jsonl

# compare the two runs
python3 cli.py compare output/run_a.jsonl output/run_b.jsonl \
    --json output/comparison.json

# evaluate individual runs with threshold gates
python3 cli.py evaluate --input output/run_b.jsonl \
    --threshold-overall 3.6 --threshold-ms-mt 0.50
```

For the test suite: `python3 -m pytest tests/` runs 140+ unit and integration tests in ~60 seconds. The opt-in real e2e test runs via `pytest -m e2e_real` (requires API keys + cached build artifacts, ~40 minutes).
