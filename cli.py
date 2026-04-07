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
    """Download ToolBench data and build graph + indexes."""
    from saplvl.graph.builder import ToolGraphBuilder
    from saplvl.ingestor.downloader import ToolBenchDownloader
    from saplvl.ingestor.parser import ToolBenchParser
    from saplvl.ingestor.registry import ToolRegistry

    data_dir = ctx.obj["data_dir"]

    # Step 1: Download
    click.echo("Step 1/3: Downloading ToolBench data...")
    downloader = ToolBenchDownloader(cache_dir=str(data_dir / "toolbench"))
    tools_file = downloader.download()
    click.echo(f"  Downloaded to {tools_file}")

    # Step 2: Parse and build registry
    click.echo("Step 2/3: Parsing tools and building registry...")
    parser = ToolBenchParser()
    tools = parser.parse_file(tools_file)
    registry = ToolRegistry(tools)
    registry_path = data_dir / "registry.json"
    registry.save(registry_path)

    summary = registry.summary()
    click.echo(f"  Registry: {summary['num_tools']} tools, "
               f"{summary['num_endpoints']} endpoints, "
               f"{summary['num_categories']} categories")

    # Step 3: Build graph
    click.echo("Step 3/3: Building tool graph...")
    builder = ToolGraphBuilder(registry)
    graph = builder.build()
    graph_path = data_dir / "tool_graph.pkl"
    builder.save(graph_path)
    click.echo(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    click.echo("\nBuild complete! Artifacts saved to: " + str(data_dir))


@cli.command()
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--count", type=int, default=100, help="Number of conversations to generate")
@click.option("--no-cross-conversation-steering", is_flag=True,
              help="Disable cross-conversation diversity steering")
@click.option("--output", "-o", type=click.Path(), default="output/conversations.jsonl",
              help="Output JSONL file path")
@click.pass_context
def generate(ctx, seed, count, no_cross_conversation_steering, output):
    """Generate synthetic conversations."""
    import os

    import anthropic
    import openai

    from saplvl.agents.assistant import AssistantAgent
    from saplvl.agents.orchestrator import ConversationOrchestrator
    from saplvl.agents.tool_executor import ToolExecutorAgent
    from saplvl.agents.user_simulator import UserSimulatorAgent
    from saplvl.graph.builder import ToolGraphBuilder
    from saplvl.ingestor.registry import ToolRegistry
    from saplvl.judgellm.judge import JudgeLLM
    from saplvl.memory.steering import DiversitySteering
    from saplvl.sampler.sampler import SamplingConstraints, ToolChainSampler
    from saplvl.sampler.scenario import ScenarioGenerator
    from saplvl.simulator.executor import SessionState, ToolSimulator

    data_dir = ctx.obj["data_dir"]

    # Set seed
    rng = random.Random(seed)
    if seed is not None:
        random.seed(seed)

    # Load artifacts
    click.echo("Loading build artifacts...")
    registry = ToolRegistry.load(data_dir / "registry.json")
    graph = ToolGraphBuilder.load(data_dir / "tool_graph.pkl")
    click.echo(f"  Registry: {len(registry)} tools, Graph: {graph.number_of_nodes()} nodes")

    # Initialize clients
    anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Initialize components — Claude for generation, OpenAI for judge + mocks
    sampler = ToolChainSampler(graph, registry, rng=rng)
    scenario_gen = ScenarioGenerator(anthropic_client)
    simulator = ToolSimulator(registry, openai_client=openai_client, rng=rng)
    steering = DiversitySteering(enabled=not no_cross_conversation_steering)

    user_agent = UserSimulatorAgent(anthropic_client)
    assistant_agent = AssistantAgent(anthropic_client, registry)
    judge = JudgeLLM(openai_client)

    # Ensure output directory exists
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"\nGenerating {count} conversations...")
    click.echo(f"  Steering: {'disabled' if no_cross_conversation_steering else 'enabled'}")
    click.echo(f"  Seed: {seed}")
    click.echo(f"  Output: {output_path}\n")

    conversations = []
    # Track desired proportions
    multi_step_count = 0
    target_multi_step_ratio = 0.55  # Target 50-60%

    with open(output_path, "w") as f:
        for i in range(count):
            try:
                # Adjust constraints to meet multi-step/multi-tool requirements
                need_multi_step = (multi_step_count / max(i, 1)) < target_multi_step_ratio
                constraints = SamplingConstraints(
                    min_tools=2 if need_multi_step else 1,
                    max_tools=5,
                    min_steps=3 if need_multi_step else 2,
                    max_steps=7,
                    exclude_tools=steering.get_exclude_tools() if steering.enabled else [],
                )

                # Sample tool chain
                chain = sampler.sample_chain(constraints)
                if not chain:
                    logger.warning("Empty chain sampled for conversation %d, retrying", i)
                    chain = sampler.sample_chain(SamplingConstraints())

                if not chain:
                    logger.error("Could not sample any chain for conversation %d", i)
                    continue

                # Generate scenario
                scenario = scenario_gen.generate_scenario(chain, registry)

                # Create fresh session and tool executor per conversation
                session = SessionState()
                tool_executor = ToolExecutorAgent(simulator, session)

                # Generate conversation
                orchestrator = ConversationOrchestrator(
                    user_agent=user_agent,
                    assistant_agent=assistant_agent,
                    tool_executor=tool_executor,
                    judge=judge,
                    steering=steering,
                )

                # Add category info to metadata
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
                }

                conversation = orchestrator.generate_conversation(chain, scenario, meta)
                conversations.append(conversation)

                # Track multi-step ratio
                if conversation.num_tool_calls >= 3 and conversation.num_distinct_tools >= 2:
                    multi_step_count += 1

                # Write to JSONL
                f.write(conversation.model_dump_json() + "\n")
                f.flush()

                # Progress
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

            except Exception as e:
                logger.error("Failed to generate conversation %d: %s", i, e, exc_info=True)
                continue

    # Print summary
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
@click.pass_context
def evaluate(ctx, input_path, output):
    """Score conversations with LLM-as-judge and compute metrics."""
    import os

    import openai as openai_lib

    from saplvl.judgellm.judge import JudgeLLM
    from saplvl.memory.steering import DiversityMetrics
    from saplvl.models import Conversation

    input_path = Path(input_path)
    output_path = Path(output) if output else input_path

    # Load conversations
    click.echo(f"Loading conversations from {input_path}...")
    conversations = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(Conversation.model_validate_json(line))

    click.echo(f"  Loaded {len(conversations)} conversations")

    # Score unscored conversations
    unscored = [c for c in conversations if c.judge_scores is None]
    if unscored:
        click.echo(f"\nScoring {len(unscored)} unscored conversations...")
        openai_client = openai_lib.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        judge = JudgeLLM(openai_client)

        scores = judge.batch_score(unscored)
        for conv, score in zip(unscored, scores):
            conv.judge_scores = score

    # Write scored conversations
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(conv.model_dump_json() + "\n")

    # Compute and display metrics
    scored = [c for c in conversations if c.judge_scores]
    if not scored:
        click.echo("No scored conversations found.")
        return

    click.echo(f"\n{'='*60}")
    click.echo("EVALUATION RESULTS")
    click.echo(f"{'='*60}")

    # Quality metrics
    mean_nat = sum(c.judge_scores.naturalness for c in scored) / len(scored)
    mean_tool = sum(c.judge_scores.tool_correctness for c in scored) / len(scored)
    mean_task = sum(c.judge_scores.task_completion for c in scored) / len(scored)
    mean_overall = (mean_nat + mean_tool + mean_task) / 3

    click.echo(f"\nQuality Scores (n={len(scored)}):")
    click.echo(f"  Naturalness:       {mean_nat:.2f}/5.0")
    click.echo(f"  Tool Correctness:  {mean_tool:.2f}/5.0")
    click.echo(f"  Task Completion:   {mean_task:.2f}/5.0")
    click.echo(f"  Overall Mean:      {mean_overall:.2f}/5.0")

    # Diversity metrics
    diversity = DiversityMetrics.summary(conversations)
    click.echo(f"\nDiversity Metrics:")
    click.echo(f"  Tool Combination Entropy:  {diversity['tool_combination_entropy']:.2f}")
    click.echo(f"  Domain Coverage Uniformity: {diversity['domain_coverage_uniformity']:.2f}")
    click.echo(f"  Unique Tool Ratio:         {diversity['unique_tool_ratio']:.2f}")
    click.echo(f"  Unique Combinations:       {diversity['unique_combinations']}/{diversity['total_conversations']}")

    # Dataset statistics
    multi_step = sum(1 for c in conversations if c.num_tool_calls >= 3 and c.num_distinct_tools >= 2)
    avg_turns = sum(c.num_turns for c in conversations) / len(conversations)
    avg_tools = sum(c.num_tool_calls for c in conversations) / len(conversations)

    click.echo(f"\nDataset Statistics:")
    click.echo(f"  Total Conversations:    {len(conversations)}")
    click.echo(f"  Multi-step/multi-tool:  {multi_step} ({multi_step/len(conversations)*100:.0f}%)")
    click.echo(f"  Avg Turns:              {avg_turns:.1f}")
    click.echo(f"  Avg Tool Calls:         {avg_tools:.1f}")

    click.echo(f"\nScored output saved to: {output_path}")


if __name__ == "__main__":
    cli()
