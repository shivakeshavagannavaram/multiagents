"""Tests for individual agents."""

import json
from unittest.mock import MagicMock

import pytest

from saplvl.agents.assistant import AssistantAgent
from saplvl.agents.tool_executor import ToolExecutorAgent
from saplvl.agents.user_simulator import UserSimulatorAgent
from saplvl.memory.context import ConversationContext
from saplvl.models import Message, ToolCall
from saplvl.simulator.executor import SessionState, ToolSimulator


class TestUserSimulatorAgent:
    def test_initial_message(self, mock_anthropic_client):
        agent = UserSimulatorAgent(mock_anthropic_client)
        ctx = ConversationContext()
        msg = agent.run(ctx, scenario="Find a hotel in Paris")
        assert msg.role == "user"
        assert msg.content is not None
        mock_anthropic_client.messages.create.assert_called_once()

    def test_followup_message(self, mock_anthropic_client):
        agent = UserSimulatorAgent(mock_anthropic_client)
        ctx = ConversationContext()
        ctx.add_message(Message(role="user", content="Find hotels"))
        ctx.add_message(Message(role="assistant", content="Which city?"))
        msg = agent.run(ctx, scenario="Find a hotel in Paris")
        assert msg.role == "user"


class TestAssistantAgent:
    def test_text_response(self, mock_anthropic_client, sample_registry):
        agent = AssistantAgent(mock_anthropic_client, sample_registry)
        ctx = ConversationContext()
        ctx.add_message(Message(role="user", content="Hello"))
        msg = agent.run(ctx, available_tools=[("HotelFinder", "search_hotels")])
        assert msg.role == "assistant"

    def test_tool_call_response(self, sample_registry):
        client = MagicMock()
        # Simulate a Claude tool_use response
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me search for hotels."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "HotelFinder__search_hotels"
        tool_block.input = {"city": "Paris", "check_in": "2026-04-11"}

        response = MagicMock()
        response.content = [text_block, tool_block]
        client.messages.create.return_value = response

        agent = AssistantAgent(client, sample_registry)
        ctx = ConversationContext()
        ctx.add_message(Message(role="user", content="Find hotels in Paris"))
        msg = agent.run(ctx, available_tools=[("HotelFinder", "search_hotels")])

        assert msg.role == "assistant"
        assert msg.content is not None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].tool_name == "HotelFinder"

    def test_build_tool_definitions(self, mock_anthropic_client, sample_registry):
        agent = AssistantAgent(mock_anthropic_client, sample_registry)
        tools = agent._build_tool_definitions([
            ("HotelFinder", "search_hotels"),
            ("FlightSearch", "search_flights"),
        ])
        assert len(tools) == 2
        assert all("name" in t and "input_schema" in t for t in tools)

    def test_sanitize_name(self):
        assert AssistantAgent._sanitize_name("Hotel Finder") == "Hotel_Finder"
        assert AssistantAgent._sanitize_name("123tool") == "t_123tool"


class TestToolExecutorAgent:
    def test_execute_tool_calls(self, sample_registry):
        simulator = ToolSimulator(sample_registry, rng=__import__("random").Random(42))
        session = SessionState()
        agent = ToolExecutorAgent(simulator, session)

        ctx = ConversationContext()
        tool_calls = [
            ToolCall(tool_name="HotelFinder", api_name="search_hotels", arguments={"city": "Paris"}),
        ]
        msg = agent.run(ctx, tool_calls=tool_calls)
        assert msg.role == "tool"
        assert msg.tool_outputs is not None
        assert len(msg.tool_outputs) == 1
        assert msg.tool_outputs[0].success

    def test_no_tool_calls(self, sample_registry):
        simulator = ToolSimulator(sample_registry)
        session = SessionState()
        agent = ToolExecutorAgent(simulator, session)

        ctx = ConversationContext()
        msg = agent.run(ctx, tool_calls=None)
        assert msg.role == "tool"
        assert "No tool calls" in msg.content
