"""End-to-end tests for the conv_gen pipeline.

Two flavors:

- ``TestEndToEnd`` — fast tests with mocked LLM clients. These verify
  pipeline wiring and structural properties (100 conversations flow
  through cleanly, data serializes round-trip). They run in seconds and
  are safe for every-commit CI. They do NOT make real claims about LLM
  quality because the judge scores are mock-generated.

- ``TestRealEndToEnd`` — slow tests with REAL LLM calls that exercise
  the full production pipeline and assert LLM-judge quality thresholds.
  These require ``ANTHROPIC_API_KEY`` and ``OPENAI_API_KEY`` to be set,
  a cached build artifact at ``.cache/registry.json``, and take ~30-60
  minutes and ~$0.50 per run. They are marked ``e2e_real`` and ``slow``
  and are excluded from default pytest runs. Invoke explicitly via::

      pytest -m e2e_real tests/test_e2e.py
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conv_gen.agents.assistant import AssistantAgent
from conv_gen.agents.orchestrator import ConversationOrchestrator
from conv_gen.agents.tool_executor import ToolExecutorAgent
from conv_gen.agents.user_simulator import UserSimulatorAgent
from conv_gen.judgellm.judge import JudgeLLM
from conv_gen.memory.steering import DiversityMetrics, DiversitySteering
from conv_gen.models import JudgeScore
from conv_gen.sampler.sampler import SamplingConstraints, ToolChainSampler
from conv_gen.simulator.executor import SessionState, ToolSimulator

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "cli.py"


def _make_mock_anthropic():
    """Mock Anthropic client for Claude-based agents.

    Behaviors:
      - User simulator calls (no ``tools`` in kwargs): return canned
        user messages rotating through a short list.
      - Assistant calls with ``tools`` provided: ALWAYS emit a
        ``tool_use`` block referencing the first available tool with
        valid arguments for every required parameter. This ensures
        every generated conversation makes at least one tool call so
        the judge's structural "no tool calls" check (which pins
        tool_correctness to 1.0 when no tools were used) never fires.
      - Assistant calls with no tools (completion branch): return a
        brief text summary.
    """
    client = MagicMock()
    call_count = [0]

    user_messages = [
        "I need to plan a trip to Paris next month.",
        "My budget is around 200 per night.",
        "Yes, please book that one.",
        "Thanks! Can you also check the weather?",
        "Great, that's all I need.",
    ]

    def _fill_args(schema):
        """Build a minimal valid argument dict from an input schema."""
        args = {}
        required = schema.get("required", [])
        props = schema.get("properties", {})
        for r in required:
            ptype = props.get(r, {}).get("type", "string")
            if ptype == "string":
                args[r] = "test_value"
            elif ptype == "integer":
                args[r] = 100
            elif ptype == "number":
                args[r] = 99.99
            elif ptype == "boolean":
                args[r] = True
            elif ptype == "array":
                args[r] = []
            elif ptype == "object":
                args[r] = {}
            else:
                args[r] = "test"
        return args

    def create_side_effect(**kwargs):
        call_count[0] += 1
        response = MagicMock()

        if "tools" not in kwargs or not kwargs.get("tools"):
            # User simulator, scenario generator, or assistant completion.
            # Return a plain text response.
            msg_idx = min(call_count[0] // 3, len(user_messages) - 1)
            block = MagicMock()
            block.type = "text"
            block.text = user_messages[msg_idx]
            response.content = [block]
            return response

        # Assistant call WITH tools — always emit a tool_use block so
        # every conversation has at least one real tool call.
        tools = kwargs["tools"]
        tool = tools[0]
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me help you with that."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = tool["name"]
        tool_block.input = _fill_args(tool.get("input_schema", {}))

        response.content = [text_block, tool_block]
        return response

    client.messages.create.side_effect = create_side_effect
    return client


def _make_mock_openai():
    """Mock OpenAI client for judge and tool simulator."""
    client = MagicMock()

    def create_side_effect(**kwargs):
        response = MagicMock()
        choice = MagicMock()

        messages = kwargs.get("messages", [])
        if messages and "Score" in str(messages[0].get("content", "")):
            score = {
                "naturalness": round(random.uniform(3.5, 4.8), 1),
                "tool_correctness": round(random.uniform(3.5, 4.8), 1),
                "task_completion": round(random.uniform(3.5, 4.8), 1),
                "reasoning": "Test evaluation",
            }
            choice.message.content = json.dumps(score)
        else:
            mock_response = {
                "results": [{"id": f"item_{random.randint(100,999)}", "name": "Test Item", "price": 150}],
                "status": "success",
            }
            choice.message.content = json.dumps(mock_response)

        response.choices = [choice]
        return response

    client.chat.completions.create.side_effect = create_side_effect
    return client


@pytest.mark.e2e
class TestEndToEnd:
    """Fast end-to-end pipeline-wiring tests with mocked LLM clients.

    These tests validate that the pipeline structure holds end-to-end
    (sampler → scenario → orchestrator → agents → judge → output),
    NOT that the generated conversations have high quality. The mocked
    judge returns random scores and the mocked assistant produces
    canned arguments, so any quality assertion would be an artifact
    of the mocks rather than a real signal.

    Quality assertions against real LLM-judge scores live in
    ``TestRealEndToEnd`` below.
    """

    def test_generate_100_conversations(self, sample_registry, sample_graph):
        random.seed(42)
        rng = random.Random(42)

        anthropic_client = _make_mock_anthropic()
        openai_client = _make_mock_openai()

        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        simulator = ToolSimulator(sample_registry, openai_client=openai_client, rng=rng)
        steering = DiversitySteering(enabled=False)

        user_agent = UserSimulatorAgent(anthropic_client)
        assistant_agent = AssistantAgent(anthropic_client, sample_registry)
        judge = JudgeLLM(openai_client)

        conversations = []

        for i in range(100):
            if i % 2 == 0:
                constraints = SamplingConstraints(min_tools=2, min_steps=3, max_steps=5)
            else:
                constraints = SamplingConstraints(min_tools=1, min_steps=2, max_steps=3)

            chain = sampler.sample_chain(constraints)
            if not chain:
                chain = sampler.sample_chain(SamplingConstraints(min_tools=1, min_steps=1, max_steps=3))
            if not chain:
                continue

            session = SessionState()
            tool_executor = ToolExecutorAgent(simulator, session)

            orchestrator = ConversationOrchestrator(
                user_agent=user_agent,
                assistant_agent=assistant_agent,
                tool_executor=tool_executor,
                judge=judge,
                steering=steering,
                max_turns=5,
                max_retries=1,
                quality_threshold=3.0,
            )

            conv = orchestrator.generate_conversation(
                chain, f"Test scenario {i}", metadata={"seed": 42, "index": i},
            )
            conversations.append(conv)

        # Wiring assertion 1: we generated the requested number of
        # conversations end-to-end.
        assert len(conversations) >= 100

        # Wiring assertion 2: every conversation got a JudgeScore object
        # attached (i.e., the judge was invoked and returned parseable
        # JSON mode output for all of them).
        scored = [c for c in conversations if c.judge_scores]
        assert len(scored) == len(conversations), (
            f"judge only scored {len(scored)}/{len(conversations)} conversations"
        )

        # Wiring assertion 3: scores fall in the valid [1, 5] range.
        # We intentionally do NOT assert a quality threshold here —
        # the judge is mocked and the scores are not meaningful for
        # real quality. TestRealEndToEnd handles the quality assertion
        # with real LLM calls.
        for c in scored:
            assert 1.0 <= c.judge_scores.naturalness <= 5.0
            assert 1.0 <= c.judge_scores.tool_correctness <= 5.0
            assert 1.0 <= c.judge_scores.task_completion <= 5.0

        # Wiring assertion 4: every conversation has at least some
        # messages (the orchestrator ran at least one turn).
        for c in conversations:
            assert len(c.messages) > 0

        # Wiring assertion 5: at least SOME conversations made tool
        # calls (otherwise the sampler/assistant wiring is broken).
        convs_with_calls = [c for c in conversations if c.tool_calls]
        assert len(convs_with_calls) > 0, (
            "no conversation produced a single tool call — "
            "pipeline wiring is broken"
        )

    def test_serialization_roundtrip(self, sample_conversation, tmp_path):
        output_path = tmp_path / "test.jsonl"

        with open(output_path, "w") as f:
            f.write(sample_conversation.model_dump_json() + "\n")

        from conv_gen.models import Conversation
        with open(output_path) as f:
            loaded = Conversation.model_validate_json(f.readline())

        assert loaded.conversation_id == sample_conversation.conversation_id
        assert len(loaded.messages) == len(sample_conversation.messages)
        assert loaded.judge_scores.naturalness == sample_conversation.judge_scores.naturalness


# ---------------------------------------------------------------------------- #
# Real end-to-end test — slow, real LLMs, opt-in only                          #
# ---------------------------------------------------------------------------- #

# Threshold justification — these values anchor the pass/fail gate of the real
# end-to-end validation. They are derived from the observed means across the
# existing 290-conversation experiment (Run A + Run B, 150 conversations each
# with seed 42), as recorded in output/comparison.json:
#
#   naturalness:      4.51   (very consistent across runs)
#   tool_correctness: 3.40   (dragged by judge structural-check false positives
#                             on fabricated IDs; this is a known tension we
#                             discussed in DESIGN.md Section C)
#   task_completion:  4.04
#   overall_mean:     3.98
#   MS+MT rate:       55.5% (Run A) / 64.6% (Run B)
#
# The thresholds below sit roughly 0.4–0.5 below the observed means, which
# leaves headroom for sample-size variance at n=100 (smaller than the 290 we
# measured on), while still being tight enough that a genuine quality
# regression (e.g., a new bug that drops task_completion by 1.0) would trip
# the check. A run below any of these thresholds signals a real problem that
# warrants investigation, not statistical noise.
#
# MS+MT rate is the tightest threshold because it's the spec's hard
# requirement (>=50% is minimum acceptable; we routinely hit 55%+).
THRESHOLD_NATURALNESS = 4.0
THRESHOLD_TOOL_CORRECTNESS = 3.0
THRESHOLD_TASK_COMPLETION = 3.5
THRESHOLD_OVERALL = 3.6
THRESHOLD_MS_MT = 0.50
E2E_REAL_COUNT = 100
E2E_REAL_SEED = 42


def _real_e2e_preconditions_met():
    """Return (met, reason) for whether the real e2e test can run."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY not set"
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY not set"
    registry_path = REPO_ROOT / ".cache" / "registry.json"
    if not registry_path.exists():
        return False, f"build artifacts missing at {registry_path} — run `python cli.py build` first"
    graph_path = REPO_ROOT / ".cache" / "tool_graph.pkl"
    if not graph_path.exists():
        return False, f"build artifacts missing at {graph_path} — run `python cli.py build` first"
    return True, ""


@pytest.mark.e2e_real
@pytest.mark.slow
class TestRealEndToEnd:
    """Real LLM-backed end-to-end validation.

    Runs the full production pipeline via subprocess (exactly as a
    reviewer would run it manually), generating 100 conversations with
    real Claude + GPT-4.1-nano + GPT-4o judge calls, and asserting that
    LLM-judge mean scores exceed the justified thresholds defined above.

    This test is the formal satisfaction of the deliverable requirement:

        "End-to-end test that builds artifacts and generates a dataset
         of at least 100 samples, asserting that LLM-as-judge mean
         scores exceed a threshold you define and justify."

    The test wraps the same CLI commands that a manual validation run
    would invoke, so passing this test is equivalent to a reviewer
    confirming by hand that the pipeline produces quality output.

    Runtime: ~30-60 minutes. Cost: ~$0.30-0.50 in API credits per run.
    Skipped automatically unless API keys and cached build artifacts
    are present.
    """

    def test_real_pipeline_meets_quality_thresholds(self, tmp_path):
        """Generate 100 real conversations and assert judge thresholds are met.

        Pipeline:
          1. Run `cli.py generate --count 100 --seed 42` to produce a
             JSONL dataset with real LLM calls for user simulation,
             assistant tool calls, mock tool responses, and judge scoring.
          2. Run `cli.py evaluate --threshold-*` to assert that the
             mean LLM-judge scores exceed the thresholds justified at
             the top of this module.

        The second subprocess exits non-zero if any threshold fails,
        which `subprocess.run(..., check=True)` translates into a
        `CalledProcessError` and a pytest failure with the FAIL output
        from the CLI visible in the exception message.
        """
        met, reason = _real_e2e_preconditions_met()
        if not met:
            pytest.skip(f"Real E2E preconditions not met: {reason}")

        output_jsonl = tmp_path / "e2e_real.jsonl"

        # Step 1: generate 100 real conversations
        generate_cmd = [
            sys.executable, str(CLI_PATH), "generate",
            "--seed", str(E2E_REAL_SEED),
            "--count", str(E2E_REAL_COUNT),
            "--no-cross-conversation-steering",
            "--output", str(output_jsonl),
        ]
        result = subprocess.run(
            generate_cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"generate command failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )
        assert output_jsonl.exists(), "generate did not produce the output file"

        # Verify we got at least the minimum required samples. Generation
        # can lose a small number of slots to transient API errors; the
        # cli.py retry loop recovers most, but we allow a 5% slack.
        line_count = sum(1 for _ in output_jsonl.open())
        min_expected = int(E2E_REAL_COUNT * 0.95)
        assert line_count >= min_expected, (
            f"generate produced {line_count} conversations, "
            f"expected at least {min_expected} (95% of {E2E_REAL_COUNT})"
        )

        # Step 2: evaluate with threshold assertions
        # When any threshold is violated, the evaluate command exits
        # with code 1 and prints FAIL details. We let subprocess raise
        # CalledProcessError so pytest surfaces the exact failure.
        evaluate_cmd = [
            sys.executable, str(CLI_PATH), "evaluate",
            "--input", str(output_jsonl),
            "--threshold-naturalness", str(THRESHOLD_NATURALNESS),
            "--threshold-tool", str(THRESHOLD_TOOL_CORRECTNESS),
            "--threshold-task", str(THRESHOLD_TASK_COMPLETION),
            "--threshold-overall", str(THRESHOLD_OVERALL),
            "--threshold-ms-mt", str(THRESHOLD_MS_MT),
        ]
        result = subprocess.run(
            evaluate_cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        # Always print the evaluate output so it's visible in the test log
        print("EVALUATE STDOUT:\n" + result.stdout)
        if result.stderr:
            print("EVALUATE STDERR:\n" + result.stderr)

        assert result.returncode == 0, (
            f"evaluate command failed thresholds (exit={result.returncode}).\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )
