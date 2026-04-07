"""Integration test for the conversation orchestrator, including retry/repair loop."""

from unittest.mock import MagicMock

import pytest

from saplvl.agents.assistant import AssistantAgent
from saplvl.agents.orchestrator import ConversationOrchestrator
from saplvl.agents.tool_executor import ToolExecutorAgent
from saplvl.agents.user_simulator import UserSimulatorAgent
from saplvl.judgellm.judge import JudgeLLM
from saplvl.models import JudgeScore
from saplvl.simulator.executor import SessionState, ToolSimulator


def _make_user_agent():
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(type="text", text="Find me a hotel in Paris")]
    client.messages.create.return_value = response
    return UserSimulatorAgent(client)


def _make_assistant_agent_with_tool_calls(registry):
    """Create an assistant agent that returns tool calls on first call, text on second."""
    client = MagicMock()
    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        response = MagicMock()

        if call_count[0] % 2 == 1:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Let me search for hotels."
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.name = "HotelFinder__search_hotels"
            tool_block.input = {"city": "Paris"}
            response.content = [text_block, tool_block]
        else:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "I found a great hotel for you!"
            response.content = [text_block]

        return response

    client.messages.create.side_effect = side_effect
    return AssistantAgent(client, registry)


def _make_judge(score_sequence):
    """Create a mock judge that returns scores from the sequence."""
    client = MagicMock()
    idx = [0]

    def score_fn(conv):
        s = score_sequence[min(idx[0], len(score_sequence) - 1)]
        idx[0] += 1
        return s

    judge = JudgeLLM(client)
    judge.score = score_fn
    return judge


class TestConversationOrchestrator:
    def test_generates_conversation(self, sample_registry):
        user_agent = _make_user_agent()
        assistant_agent = _make_assistant_agent_with_tool_calls(sample_registry)
        simulator = ToolSimulator(sample_registry, rng=__import__("random").Random(42))
        session = SessionState()
        tool_executor = ToolExecutorAgent(simulator, session)

        judge = _make_judge([
            JudgeScore(naturalness=4.0, tool_correctness=4.0, task_completion=4.0)
        ])

        orchestrator = ConversationOrchestrator(
            user_agent=user_agent,
            assistant_agent=assistant_agent,
            tool_executor=tool_executor,
            judge=judge,
            max_turns=3,
        )

        chain = [("HotelFinder", "search_hotels"), ("HotelFinder", "book_hotel")]
        conv = orchestrator.generate_conversation(chain, "Find a hotel in Paris")

        assert conv.conversation_id.startswith("conv_")
        assert len(conv.messages) > 0
        assert conv.judge_scores is not None

    def test_retry_on_low_score(self, sample_registry):
        user_agent = _make_user_agent()
        assistant_agent = _make_assistant_agent_with_tool_calls(sample_registry)
        simulator = ToolSimulator(sample_registry, rng=__import__("random").Random(42))
        session = SessionState()
        tool_executor = ToolExecutorAgent(simulator, session)

        judge = _make_judge([
            JudgeScore(naturalness=2.0, tool_correctness=2.0, task_completion=2.0),
            JudgeScore(naturalness=4.5, tool_correctness=4.5, task_completion=4.5),
        ])

        orchestrator = ConversationOrchestrator(
            user_agent=user_agent,
            assistant_agent=assistant_agent,
            tool_executor=tool_executor,
            judge=judge,
            max_turns=3,
            max_retries=2,
            quality_threshold=3.5,
        )

        chain = [("HotelFinder", "search_hotels")]
        conv = orchestrator.generate_conversation(chain, "Find a hotel")
        assert conv.judge_scores is not None
        assert conv.judge_scores.mean_score >= 3.5

    def test_keeps_best_after_max_retries(self, sample_registry):
        user_agent = _make_user_agent()
        assistant_agent = _make_assistant_agent_with_tool_calls(sample_registry)
        simulator = ToolSimulator(sample_registry, rng=__import__("random").Random(42))
        session = SessionState()
        tool_executor = ToolExecutorAgent(simulator, session)

        judge = _make_judge([
            JudgeScore(naturalness=2.0, tool_correctness=2.0, task_completion=2.0),
            JudgeScore(naturalness=2.5, tool_correctness=2.5, task_completion=2.5),
            JudgeScore(naturalness=2.2, tool_correctness=2.2, task_completion=2.2),
        ])

        orchestrator = ConversationOrchestrator(
            user_agent=user_agent,
            assistant_agent=assistant_agent,
            tool_executor=tool_executor,
            judge=judge,
            max_turns=3,
            max_retries=2,
            quality_threshold=4.0,
        )

        chain = [("HotelFinder", "search_hotels")]
        conv = orchestrator.generate_conversation(chain, "Find a hotel")
        assert conv.judge_scores.mean_score == pytest.approx(2.5)

    def test_repair_hints_generated(self):
        low_natural = JudgeScore(naturalness=2.0, tool_correctness=4.0, task_completion=4.0)
        hints = ConversationOrchestrator._build_repair_hints(low_natural)
        assert "natural" in hints.lower() or "conversational" in hints.lower()

        low_tool = JudgeScore(naturalness=4.0, tool_correctness=2.0, task_completion=4.0)
        hints = ConversationOrchestrator._build_repair_hints(low_tool)
        assert "parameter" in hints.lower() or "schema" in hints.lower()

        low_task = JudgeScore(naturalness=4.0, tool_correctness=4.0, task_completion=2.0)
        hints = ConversationOrchestrator._build_repair_hints(low_task)
        assert "request" in hints.lower() or "complete" in hints.lower()

    def test_no_judge_still_works(self, sample_registry):
        user_agent = _make_user_agent()
        assistant_agent = _make_assistant_agent_with_tool_calls(sample_registry)
        simulator = ToolSimulator(sample_registry, rng=__import__("random").Random(42))
        session = SessionState()
        tool_executor = ToolExecutorAgent(simulator, session)

        orchestrator = ConversationOrchestrator(
            user_agent=user_agent,
            assistant_agent=assistant_agent,
            tool_executor=tool_executor,
            judge=None,
            max_turns=3,
        )

        chain = [("HotelFinder", "search_hotels")]
        conv = orchestrator.generate_conversation(chain, "Find a hotel")
        assert conv.judge_scores is None
        assert len(conv.messages) > 0
