"""End-to-end test: generate conversations with mocked LLM clients, assert quality."""

import json
import random
from unittest.mock import MagicMock

import pytest

from saplvl.agents.assistant import AssistantAgent
from saplvl.agents.orchestrator import ConversationOrchestrator
from saplvl.agents.tool_executor import ToolExecutorAgent
from saplvl.agents.user_simulator import UserSimulatorAgent
from saplvl.judgellm.judge import JudgeLLM
from saplvl.memory.steering import DiversityMetrics, DiversitySteering
from saplvl.models import JudgeScore
from saplvl.sampler.sampler import SamplingConstraints, ToolChainSampler
from saplvl.simulator.executor import SessionState, ToolSimulator


def _make_mock_anthropic():
    """Mock Anthropic client for Claude-based agents."""
    client = MagicMock()
    call_count = [0]

    user_messages = [
        "I need to plan a trip to Paris next month.",
        "My budget is around 200 per night.",
        "Yes, please book that one.",
        "Thanks! Can you also check the weather?",
        "Great, that's all I need.",
    ]

    def create_side_effect(**kwargs):
        call_count[0] += 1
        response = MagicMock()

        if "tools" not in kwargs:
            # User simulator or scenario generator call
            msg_idx = min(call_count[0] // 3, len(user_messages) - 1)
            block = MagicMock()
            block.type = "text"
            block.text = user_messages[msg_idx]
            response.content = [block]
        else:
            # Assistant call - alternate between tool use and text
            if call_count[0] % 3 == 0:
                text_block = MagicMock()
                text_block.type = "text"
                text_block.text = "Let me help you with that."

                tool_block = MagicMock()
                tool_block.type = "tool_use"

                tools = kwargs.get("tools", [])
                if tools:
                    tool_block.name = tools[0]["name"]
                    schema = tools[0].get("input_schema", {})
                    required = schema.get("required", [])
                    props = schema.get("properties", {})
                    args = {}
                    for r in required:
                        ptype = props.get(r, {}).get("type", "string")
                        if ptype == "string":
                            args[r] = "test_value"
                        elif ptype == "integer":
                            args[r] = 100
                        elif ptype == "number":
                            args[r] = 99.99
                        else:
                            args[r] = "test"
                    tool_block.input = args
                else:
                    tool_block.name = "test__test"
                    tool_block.input = {}

                response.content = [text_block, tool_block]
            else:
                text_block = MagicMock()
                text_block.type = "text"
                text_block.text = "I found some great options for you."
                response.content = [text_block]

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

        assert len(conversations) >= 100

        scored = [c for c in conversations if c.judge_scores]
        assert len(scored) > 0

        mean_nat = sum(c.judge_scores.naturalness for c in scored) / len(scored)
        mean_tool = sum(c.judge_scores.tool_correctness for c in scored) / len(scored)
        mean_task = sum(c.judge_scores.task_completion for c in scored) / len(scored)

        assert mean_nat > 3.0
        assert mean_tool > 3.0
        assert mean_task > 3.0

    def test_serialization_roundtrip(self, sample_conversation, tmp_path):
        output_path = tmp_path / "test.jsonl"

        with open(output_path, "w") as f:
            f.write(sample_conversation.model_dump_json() + "\n")

        from saplvl.models import Conversation
        with open(output_path) as f:
            loaded = Conversation.model_validate_json(f.readline())

        assert loaded.conversation_id == sample_conversation.conversation_id
        assert len(loaded.messages) == len(sample_conversation.messages)
        assert loaded.judge_scores.naturalness == sample_conversation.judge_scores.naturalness
