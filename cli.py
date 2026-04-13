"""CLI for the Multi-Agent Tool-Use Conversation Generator."""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--data-dir", default=".cache", type=click.Path(), help="Directory for cached artifacts")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, data_dir, verbose):
    """Multi-Agent Tool-Use Conversation Generator."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = Path(data_dir)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.pass_context
def build(ctx):
    """Download ToolBench data, select tools, and build Knowledge Graph."""
    from conv_gen.graph.builder import ToolGraphBuilder
    from conv_gen.ingestor.downloader import ToolBenchDownloader
    from conv_gen.ingestor.parser import ToolBenchParser
    from conv_gen.ingestor.registry import ToolRegistry
    from conv_gen.ingestor.selector import select_tools

    data_dir = ctx.obj["data_dir"]

    click.echo("Step 1/5: Downloading ToolBench data (full dataset, ~10,600 tools)...")
    downloader = ToolBenchDownloader(cache_dir=str(data_dir / "toolbench"))
    tools_file = downloader.download()
    click.echo(f"  Downloaded to {tools_file}")

    click.echo("Step 2/5: Parsing tools...")
    parser = ToolBenchParser()
    all_tools = parser.parse_file(tools_file)
    click.echo(f"  Parsed: {len(all_tools)} tools")

    full_registry = ToolRegistry(all_tools)
    full_registry.save(data_dir / "registry_full.json")

    click.echo("Step 3/5: Selecting 500 best tools...")
    selected = select_tools(all_tools, target_count=500)

    summary_before = sum(len(t.api_list) for t in selected)
    click.echo(f"  Selected: {len(selected)} tools, "
               f"{summary_before} endpoints")

    click.echo("Step 4/5: Enriching response schemas with LLM...")
    import os
    import openai as openai_lib
    from conv_gen.ingestor.schema_enricher import enrich_schemas

    openai_client = openai_lib.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    selected = enrich_schemas(selected, openai_client)

    registry = ToolRegistry(selected)
    registry.save(data_dir / "registry.json")

    summary = registry.summary()
    click.echo(f"  Registry: {summary['num_tools']} tools, "
               f"{summary['num_endpoints']} endpoints, "
               f"{summary['num_categories']} categories")

    click.echo("Step 5/5: Building Knowledge Graph...")
    builder = ToolGraphBuilder(registry)
    graph = builder.build()
    graph_path = data_dir / "tool_graph.pkl"
    builder.save(graph_path)

    builder.export_json(Path("output/kg_export.json"))
    builder.export_html(Path("output/kg_visualization.html"))

    click.echo(f"  KG: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    click.echo(f"\nBuild complete! Artifacts saved to: {data_dir}")
    click.echo(f"  KG visualization: output/kg_visualization.html")
    click.echo(f"  KG export (editable): output/kg_export.json")


@cli.command()
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--count", type=int, default=100, help="Number of conversations to generate")
@click.option("--no-cross-conversation-steering", is_flag=True,
              help="Disable cross-conversation diversity steering")
@click.option("--no-llm-mocks", is_flag=True,
              help="Disable LLM mocks, use schema-derived mocks instead (fully offline)")
@click.option("--output", "-o", type=click.Path(), default="output/conversations.jsonl",
              help="Output JSONL file path")
@click.pass_context
def generate(ctx, seed, count, no_cross_conversation_steering, no_llm_mocks, output):
    """Generate synthetic conversations."""
    import os

    import anthropic
    import openai

    from conv_gen.agents.assistant import AssistantAgent
    from conv_gen.agents.orchestrator import ConversationOrchestrator
    from conv_gen.agents.tool_executor import ToolExecutorAgent
    from conv_gen.agents.user_simulator import UserSimulatorAgent
    from conv_gen.graph.builder import ToolGraphBuilder
    from conv_gen.ingestor.registry import ToolRegistry
    from conv_gen.judgellm.judge import JudgeLLM
    from conv_gen.memory.steering import DiversitySteering
    from conv_gen.output_format import to_wire_json
    from conv_gen.sampler.sampler import SamplingConstraints, SamplingPattern, ToolChainSampler
    from conv_gen.sampler.scenario import ScenarioGenerator
    from conv_gen.simulator.executor import SessionState, ToolSimulator

    data_dir = ctx.obj["data_dir"]

    rng = random.Random(seed)
    if seed is not None:
        random.seed(seed)

    click.echo("Loading build artifacts...")
    registry = ToolRegistry.load(data_dir / "registry.json")
    graph = ToolGraphBuilder.load(data_dir / "tool_graph.pkl")
    click.echo(f"  Registry: {len(registry)} tools, Graph: {graph.number_of_nodes()} nodes")

    anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    sampler = ToolChainSampler(
        graph, registry, rng=rng,
        coherence_client=openai_client,
        coherence_model="gpt-4.1-nano",
    )
    scenario_gen = ScenarioGenerator(anthropic_client, rng=rng)
    simulator = ToolSimulator(registry, openai_client=openai_client, use_llm_mocks=not no_llm_mocks, rng=rng)
    steering = DiversitySteering(enabled=not no_cross_conversation_steering)

    user_agent = UserSimulatorAgent(anthropic_client)
    assistant_agent = AssistantAgent(anthropic_client, registry)
    judge = JudgeLLM(openai_client)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"\nGenerating {count} conversations...")
    click.echo(f"  Steering: {'disabled' if no_cross_conversation_steering else 'enabled'}")
    click.echo(f"  Seed: {seed}")
    click.echo(f"  Output: {output_path}\n")

    from conv_gen.agents.director import ConversationDirector, CONVERSATION_TYPES

    conversations = []
    director = ConversationDirector(rng=rng)

    MAX_SLOT_ATTEMPTS = 3

    with open(output_path, "w") as f:
        for i in range(count):
            # Director picks the type once per slot; retries resample within the same type.
            conv_type = director.next_type()

            conversation = None
            last_error: Exception | None = None

            for retry in range(MAX_SLOT_ATTEMPTS):
                try:
                    exclude = steering.get_exclude_tools() if steering.enabled else []
                    constraints = director.build_sampler_constraints(conv_type, exclude)

                    if conv_type.pattern == SamplingPattern.PARALLEL:
                        sampled = sampler.sample_parallel(constraints)
                    elif conv_type.pattern == SamplingPattern.SINGLE:
                        sampled = sampler.sample_single(constraints)
                    else:
                        sampled = sampler.sample_sequential(constraints)

                    chain = sampled.flat_chain
                    if not chain:
                        logger.warning(
                            "Empty chain for conversation %d attempt %d (type=%s), falling back",
                            i, retry + 1, conv_type.name,
                        )
                        chain = sampler.sample_chain(SamplingConstraints())

                    if not chain:
                        raise RuntimeError(
                            f"Could not sample any chain for slot {i} on attempt {retry + 1}"
                        )

                    scenario = scenario_gen.generate_scenario(
                        chain, registry, require_chaining=conv_type.require_chaining,
                    )

                    session = SessionState()
                    tool_executor = ToolExecutorAgent(simulator, session)

                    plan_kwargs = {
                        "max_turns": conv_type.max_turns,
                        "max_messages": conv_type.max_messages,
                    }

                    orchestrator = ConversationOrchestrator(
                        user_agent=user_agent,
                        assistant_agent=assistant_agent,
                        tool_executor=tool_executor,
                        judge=judge,
                        steering=steering,
                    )

                    categories = {}
                    for tool_name, api_name in chain:
                        tool = registry.get_tool(tool_name)
                        if tool:
                            categories[f"{tool_name}/{api_name}"] = tool.category
                    meta = {
                        "seed": seed,
                        "categories": categories,
                        "categories_list": list(set(categories.values())),
                        "steering_enabled": not no_cross_conversation_steering,
                        "pattern": sampled.pattern.value,
                        "conversation_type": conv_type.name,
                    }

                    conversation = orchestrator.generate_conversation(
                        chain, scenario, meta,
                        pattern=sampled.pattern.value,
                        steps=sampled.steps,
                        plan_kwargs=plan_kwargs,
                    )
                    break

                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Slot %d attempt %d/%d failed: %s. Resampling fresh chain.",
                        i, retry + 1, MAX_SLOT_ATTEMPTS, e,
                    )
                    continue

            if conversation is None:
                logger.error(
                    "Slot %d: all %d attempts failed, skipping. Last error: %s",
                    i, MAX_SLOT_ATTEMPTS, last_error,
                )
                continue

            conversations.append(conversation)

            director.record(conv_type.name)

            f.write(to_wire_json(conversation) + "\n")
            f.flush()

            scores_str = ""
            if conversation.judge_scores:
                s = conversation.judge_scores
                scores_str = (
                    f" | scores: nat={s.naturalness:.1f} "
                    f"tool={s.tool_correctness:.1f} "
                    f"task={s.task_completion:.1f}"
                )

            click.echo(
                f"  [{i+1}/{count}] {conversation.conversation_id}: "
                f"{conversation.num_turns} turns, "
                f"{conversation.num_tool_calls} tool calls, "
                f"{conversation.num_distinct_tools} tools"
                f"{scores_str}"
            )

    click.echo(f"\nGeneration complete: {len(conversations)} conversations")
    if conversations:
        multi_step = sum(
            1 for c in conversations
            if c.num_tool_calls >= 3 and c.num_distinct_tools >= 2
        )
        click.echo(f"  Multi-step/multi-tool: {multi_step}/{len(conversations)} "
                    f"({multi_step/len(conversations)*100:.0f}%)")

        scored = [c for c in conversations if c.judge_scores]
        if scored:
            mean_nat = sum(c.judge_scores.naturalness for c in scored) / len(scored)
            mean_tool = sum(c.judge_scores.tool_correctness for c in scored) / len(scored)
            mean_task = sum(c.judge_scores.task_completion for c in scored) / len(scored)
            click.echo(f"  Mean scores: naturalness={mean_nat:.2f}, "
                       f"tool_correctness={mean_tool:.2f}, "
                       f"task_completion={mean_task:.2f}")

    click.echo(f"\nOutput saved to: {output_path}")


@cli.command()
@click.option("--input", "input_path", type=click.Path(exists=True), required=True,
              help="Input JSONL file with conversations")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output JSONL file with scores (default: overwrite input)")
@click.option("--registry", type=click.Path(), default=None,
              help="Path to registry.json for computing tool coverage ratio "
                   "(default: <data-dir>/registry.json)")
@click.option("--json", "json_out", type=click.Path(), default=None,
              help="Write the full metrics summary to a JSON file")
@click.option("--threshold-naturalness", type=float, default=None,
              help="Fail (exit 1) if mean naturalness is below this value")
@click.option("--threshold-tool", type=float, default=None,
              help="Fail (exit 1) if mean tool_correctness is below this value")
@click.option("--threshold-task", type=float, default=None,
              help="Fail (exit 1) if mean task_completion is below this value")
@click.option("--threshold-overall", type=float, default=None,
              help="Fail (exit 1) if overall mean (avg of the three) is below this value")
@click.option("--threshold-ms-mt", type=float, default=None,
              help="Fail (exit 1) if MS+MT rate (0.0-1.0) is below this value")
@click.pass_context
def evaluate(ctx, input_path, output, registry, json_out,
             threshold_naturalness, threshold_tool, threshold_task,
             threshold_overall, threshold_ms_mt):
    """Score conversations with LLM-as-judge and compute metrics.

    Optionally asserts quality thresholds — when any --threshold-* flag
    is set, the command exits with code 1 if the corresponding mean
    score falls below the threshold. This makes `evaluate` usable as the
    assertion step of an end-to-end validation workflow:

        python cli.py build
        python cli.py generate --seed 42 --count 100 --output output/validation.jsonl
        python cli.py evaluate --input output/validation.jsonl \\
            --threshold-overall 3.6 --threshold-naturalness 4.0

    Exit code 0 means all specified thresholds were met; exit code 1
    means at least one threshold failed (details printed to stderr).
    """
    import json as json_lib
    import os
    import sys

    import openai as openai_lib

    from conv_gen.ingestor.registry import ToolRegistry
    from conv_gen.judgellm.judge import JudgeLLM
    from conv_gen.memory.steering import DiversityMetrics, QualityMetrics
    from conv_gen.models import Conversation
    from conv_gen.output_format import from_any_json, to_wire_json

    data_dir = ctx.obj["data_dir"]
    input_path = Path(input_path)
    output_path = Path(output) if output else input_path

    click.echo(f"Loading conversations from {input_path}...")
    conversations = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(from_any_json(line))

    click.echo(f"  Loaded {len(conversations)} conversations")

    unscored = [c for c in conversations if c.judge_scores is None]
    if unscored:
        click.echo(f"\nScoring {len(unscored)} unscored conversations...")
        openai_client = openai_lib.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        judge = JudgeLLM(openai_client)

        scores = judge.batch_score(unscored)
        for conv, score in zip(unscored, scores):
            conv.judge_scores = score

    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(to_wire_json(conv) + "\n")

    scored = [c for c in conversations if c.judge_scores]
    if not scored:
        click.echo("No scored conversations found.")
        return

    registry_size = None
    all_categories = None
    registry_path = Path(registry) if registry else data_dir / "registry.json"
    if registry_path.exists():
        try:
            reg = ToolRegistry.load(registry_path)
            summary = reg.summary()
            registry_size = summary.get("num_endpoints")
            all_categories = summary.get("categories")
        except Exception as e:
            logger.warning("Could not load registry for coverage ratio: %s", e)

    diversity = DiversityMetrics.summary(
        conversations, registry_size=registry_size, all_categories=all_categories,
    )
    quality = QualityMetrics.summary(conversations)

    _print_metrics_report(
        title="EVALUATION RESULTS",
        conversations=conversations,
        diversity=diversity,
        quality=quality,
    )

    if json_out:
        full_summary = {
            "input": str(input_path),
            "diversity": diversity,
            "quality": quality,
            "total_conversations": len(conversations),
        }
        Path(json_out).write_text(json_lib.dumps(full_summary, indent=2))
        click.echo(f"\nJSON summary written to: {json_out}")

    click.echo(f"\nScored output saved to: {output_path}")

    threshold_checks = [
        ("naturalness", threshold_naturalness, quality["quality"]["naturalness"]),
        ("tool_correctness", threshold_tool, quality["quality"]["tool_correctness"]),
        ("task_completion", threshold_task, quality["quality"]["task_completion"]),
        ("overall_mean", threshold_overall, quality["quality"]["overall_mean"]),
        ("ms_mt_rate", threshold_ms_mt, quality["spec"]["ms_mt_rate"]),
    ]
    active_checks = [(name, th, val) for name, th, val in threshold_checks if th is not None]

    if active_checks:
        click.echo(f"\n{'='*68}")
        click.echo("  THRESHOLD CHECKS")
        click.echo(f"{'='*68}")
        failures = []
        for name, threshold, value in active_checks:
            passed = value >= threshold
            mark = "PASS" if passed else "FAIL"
            click.echo(f"  [{mark}] {name:<20} {value:.3f}  (threshold >= {threshold})")
            if not passed:
                failures.append((name, threshold, value))

        if failures:
            click.echo(f"\n  ❌ {len(failures)} threshold check(s) FAILED:", err=True)
            for name, threshold, value in failures:
                click.echo(
                    f"     {name}: {value:.3f} < {threshold}", err=True
                )
            click.echo(f"\n  Overall result: FAIL", err=True)
            sys.exit(1)
        else:
            click.echo(f"\n  ✓ All {len(active_checks)} threshold check(s) passed.")
            click.echo(f"  Overall result: PASS")


@cli.command()
@click.argument("run_a", type=click.Path(exists=True))
@click.argument("run_b", type=click.Path(exists=True))
@click.option("--registry", type=click.Path(), default=None,
              help="Path to registry.json (default: <data-dir>/registry.json)")
@click.option("--label-a", default="Run A", help="Label for the first run")
@click.option("--label-b", default="Run B", help="Label for the second run")
@click.option("--json", "json_out", type=click.Path(), default=None,
              help="Write the comparison report to a JSON file")
@click.pass_context
def compare(ctx, run_a, run_b, registry, label_a, label_b, json_out):
    """Side-by-side quality + diversity comparison for two runs (Δ = B − A)."""
    import json as json_lib
    from collections import Counter

    from conv_gen.ingestor.registry import ToolRegistry
    from conv_gen.memory.steering import DiversityMetrics, QualityMetrics
    from conv_gen.models import Conversation
    from conv_gen.output_format import from_any_json

    data_dir = ctx.obj["data_dir"]

    def load_jsonl(path: Path) -> list[Conversation]:
        convs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    convs.append(from_any_json(line))
        return convs

    click.echo(f"Loading {label_a}: {run_a}")
    convs_a = load_jsonl(Path(run_a))
    click.echo(f"Loading {label_b}: {run_b}")
    convs_b = load_jsonl(Path(run_b))
    click.echo(f"  {label_a}: {len(convs_a)} conversations")
    click.echo(f"  {label_b}: {len(convs_b)} conversations")

    registry_size = None
    all_categories = None
    registry_path = Path(registry) if registry else data_dir / "registry.json"
    if registry_path.exists():
        try:
            reg = ToolRegistry.load(registry_path)
            summary = reg.summary()
            registry_size = summary.get("num_endpoints")
            all_categories = summary.get("categories")
            click.echo(f"  Registry: {registry_size} endpoints, {len(all_categories) if all_categories else 0} categories")
        except Exception as e:
            logger.warning("Could not load registry: %s", e)
    else:
        click.echo(f"  (registry not found at {registry_path} — coverage ratio will be omitted)")

    div_a = DiversityMetrics.summary(convs_a, registry_size=registry_size, all_categories=all_categories)
    div_b = DiversityMetrics.summary(convs_b, registry_size=registry_size, all_categories=all_categories)
    qual_a = QualityMetrics.summary(convs_a)
    qual_b = QualityMetrics.summary(convs_b)

    def per_type_ms_mt(convs):
        total = Counter()
        hits = Counter()
        for c in convs:
            ctype = c.metadata.get("conversation_type", "?")
            total[ctype] += 1
            tcs = c.tool_calls or []
            distinct = {(tc.tool_name, tc.api_name) for tc in tcs}
            if len(tcs) >= 3 and len(distinct) >= 2:
                hits[ctype] += 1
        return total, hits

    type_total_a, type_hits_a = per_type_ms_mt(convs_a)
    type_total_b, type_hits_b = per_type_ms_mt(convs_b)
    all_types = sorted(set(type_total_a) | set(type_total_b))

    def tools_of(convs):
        s = set()
        for c in convs:
            for tc in c.tool_calls or []:
                s.add((tc.tool_name, tc.api_name))
        return s

    tools_a = tools_of(convs_a)
    tools_b = tools_of(convs_b)
    tools_both = tools_a & tools_b
    tools_only_a = tools_a - tools_b
    tools_only_b = tools_b - tools_a

    click.echo()
    click.echo("=" * 78)
    click.echo(f"  COMPARISON: {label_a}  vs  {label_b}")
    click.echo("=" * 78)

    def _row(name: str, a, b, fmt=".4f"):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = b - a
            a_s = format(a, fmt)
            b_s = format(b, fmt)
            d_s = f"{delta:+{fmt}}"
            click.echo(f"  {name:<40}{a_s:>14}{b_s:>14}{d_s:>12}")
        else:
            click.echo(f"  {name:<40}{str(a):>14}{str(b):>14}")

    click.echo(f"\n  {'Metric':<40}{label_a:>14}{label_b:>14}{'Δ (B-A)':>12}")
    click.echo("  " + "-" * 76)

    click.echo("\n  QUALITY (LLM judge means, 1-5)")
    _row("  naturalness", qual_a["quality"]["naturalness"], qual_b["quality"]["naturalness"], ".3f")
    _row("  tool_correctness", qual_a["quality"]["tool_correctness"], qual_b["quality"]["tool_correctness"], ".3f")
    _row("  task_completion", qual_a["quality"]["task_completion"], qual_b["quality"]["task_completion"], ".3f")
    _row("  overall_mean", qual_a["quality"]["overall_mean"], qual_b["quality"]["overall_mean"], ".3f")

    click.echo("\n  SPEC COMPLIANCE")
    _row("  MS+MT rate (≥3 calls, ≥2 tools)", qual_a["spec"]["ms_mt_rate"], qual_b["spec"]["ms_mt_rate"], ".3f")
    _row("  real chaining rate", qual_a["spec"]["real_chaining_rate"], qual_b["spec"]["real_chaining_rate"], ".3f")

    click.echo("\n  DIVERSITY — PRIMARY (what steering is designed to change)")
    _row("  tool_usage_entropy (0-1, higher=flatter)",
         div_a["primary"]["tool_usage_entropy"], div_b["primary"]["tool_usage_entropy"], ".4f")
    _row("  unique_tools_used",
         div_a["primary"]["unique_tools_used"], div_b["primary"]["unique_tools_used"], "d")
    if "unique_tool_coverage_ratio" in div_a["primary"]:
        _row("  unique_tool_coverage_ratio (of registry)",
             div_a["primary"]["unique_tool_coverage_ratio"],
             div_b["primary"]["unique_tool_coverage_ratio"], ".4f")

    click.echo("\n  DIVERSITY — SECONDARY")
    _row("  top_5_tool_concentration (lower=better)",
         div_a["secondary"]["top_5_tool_concentration"], div_b["secondary"]["top_5_tool_concentration"], ".4f")
    _row("  tool_combination_entropy (chain-level)",
         div_a["secondary"]["tool_combination_entropy"], div_b["secondary"]["tool_combination_entropy"], ".3f")
    _row("  domain_coverage_uniformity",
         div_a["secondary"]["domain_coverage_uniformity"], div_b["secondary"]["domain_coverage_uniformity"], ".4f")
    _row("  unique_chain_ratio",
         div_a["secondary"]["unique_chain_ratio"], div_b["secondary"]["unique_chain_ratio"], ".4f")
    _row("  unique_chain_combinations",
         div_a["secondary"]["unique_chain_combinations"], div_b["secondary"]["unique_chain_combinations"], "d")

    click.echo("\n  PER-TYPE MS+MT RATE")
    click.echo(f"  {'Type':<40}{'A hits/tot':>14}{'B hits/tot':>14}{'B rate':>12}")
    click.echo("  " + "-" * 76)
    for t in all_types:
        a_hits, a_tot = type_hits_a.get(t, 0), type_total_a.get(t, 0)
        b_hits, b_tot = type_hits_b.get(t, 0), type_total_b.get(t, 0)
        b_rate = f"{100*b_hits/b_tot:.0f}%" if b_tot else "—"
        click.echo(f"  {t:<40}{f'{a_hits}/{a_tot}':>14}{f'{b_hits}/{b_tot}':>14}{b_rate:>12}")

    click.echo("\n  TOOL OVERLAP")
    click.echo(f"  Tools used in BOTH runs:        {len(tools_both)}")
    click.echo(f"  Tools only in {label_a}:           {len(tools_only_a)}")
    click.echo(f"  Tools only in {label_b}:           {len(tools_only_b)}")
    click.echo(f"  Total unique across both:       {len(tools_a | tools_b)}")

    click.echo("\n" + "=" * 78)

    if json_out:
        report = {
            "label_a": label_a,
            "label_b": label_b,
            "run_a_path": str(run_a),
            "run_b_path": str(run_b),
            "run_a": {
                "diversity": div_a,
                "quality": qual_a,
                "total_conversations": len(convs_a),
                "per_type": {
                    t: {"total": type_total_a.get(t, 0), "ms_mt": type_hits_a.get(t, 0)}
                    for t in all_types
                },
            },
            "run_b": {
                "diversity": div_b,
                "quality": qual_b,
                "total_conversations": len(convs_b),
                "per_type": {
                    t: {"total": type_total_b.get(t, 0), "ms_mt": type_hits_b.get(t, 0)}
                    for t in all_types
                },
            },
            "tool_overlap": {
                "both": len(tools_both),
                "only_a": len(tools_only_a),
                "only_b": len(tools_only_b),
                "total_unique": len(tools_a | tools_b),
            },
        }
        Path(json_out).write_text(json_lib.dumps(report, indent=2))
        click.echo(f"\nJSON comparison report written to: {json_out}")


def _print_metrics_report(title, conversations, diversity, quality):
    """Print a single-run metrics report."""
    click.echo()
    click.echo("=" * 68)
    click.echo(f"  {title}")
    click.echo("=" * 68)

    click.echo(f"\n  Conversations scored: {quality['quality']['scored_count']}")

    click.echo("\n  QUALITY (LLM judge means, 1-5)")
    click.echo(f"    Naturalness:       {quality['quality']['naturalness']:.2f}")
    click.echo(f"    Tool correctness:  {quality['quality']['tool_correctness']:.2f}")
    click.echo(f"    Task completion:   {quality['quality']['task_completion']:.2f}")
    click.echo(f"    Overall mean:      {quality['quality']['overall_mean']:.2f}")

    click.echo("\n  SPEC COMPLIANCE")
    click.echo(f"    MS+MT rate:        {quality['spec']['ms_mt_rate']*100:.1f}%")
    click.echo(f"    Real chaining:     {quality['spec']['real_chaining_rate']*100:.1f}%")

    click.echo("\n  DIVERSITY — PRIMARY (steering target metrics)")
    click.echo(f"    tool_usage_entropy:             {diversity['primary']['tool_usage_entropy']:.4f}")
    click.echo(f"    unique_tools_used:              {diversity['primary']['unique_tools_used']}")
    if "unique_tool_coverage_ratio" in diversity["primary"]:
        click.echo(f"    unique_tool_coverage_ratio:     {diversity['primary']['unique_tool_coverage_ratio']:.4f}")

    click.echo("\n  DIVERSITY — SECONDARY")
    click.echo(f"    top_5_tool_concentration:       {diversity['secondary']['top_5_tool_concentration']:.4f}")
    click.echo(f"    tool_combination_entropy:       {diversity['secondary']['tool_combination_entropy']:.3f}")
    click.echo(f"    domain_coverage_uniformity:     {diversity['secondary']['domain_coverage_uniformity']:.4f}")
    click.echo(f"    unique_chain_ratio:             {diversity['secondary']['unique_chain_ratio']:.4f}")
    click.echo(f"    unique_chain_combinations:      {diversity['secondary']['unique_chain_combinations']}/{len(conversations)}")

    click.echo("\n" + "=" * 68)


if __name__ == "__main__":
    cli()
