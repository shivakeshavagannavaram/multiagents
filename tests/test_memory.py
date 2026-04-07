"""Tests for context management and diversity metrics."""

import pytest

from saplvl.memory.context import ConversationContext
from saplvl.memory.steering import DiversityMetrics, DiversitySteering
from saplvl.models import (
    Conversation,
    JudgeScore,
    Message,
    ToolCall,
    ToolOutput,
)


class TestConversationContext:
    def test_add_and_get_messages(self):
        ctx = ConversationContext()
        msg = Message(role="user", content="Hello")
        ctx.add_message(msg)
        assert len(ctx.messages) == 1
        assert ctx.messages[0].content == "Hello"

    def test_build_prompt_context(self):
        ctx = ConversationContext()
        ctx.add_message(Message(role="user", content="Find hotels"))
        ctx.add_message(Message(role="assistant", content="In which city?"))
        result = ctx.build_prompt_context()
        assert "USER: Find hotels" in result
        assert "ASSISTANT: In which city?" in result

    def test_tool_output_extraction(self):
        ctx = ConversationContext()
        tc = ToolCall(tool_name="T", api_name="search", arguments={})
        output = ToolOutput(
            tool_call=tc,
            response={"hotel_id": "htl_1", "name": "Test Hotel"},
        )
        ctx.add_tool_output(output)
        values = ctx.get_available_values()
        assert "hotel_id" in values
        assert values["hotel_id"] == "htl_1"

    def test_format_available_values_empty(self):
        ctx = ConversationContext()
        result = ctx.format_available_values()
        assert "No values" in result

    def test_get_last_user_message(self):
        ctx = ConversationContext()
        ctx.add_message(Message(role="user", content="First"))
        ctx.add_message(Message(role="assistant", content="Reply"))
        ctx.add_message(Message(role="user", content="Second"))
        assert ctx.get_last_user_message() == "Second"

    def test_truncation(self):
        ctx = ConversationContext(max_history_chars=100)
        for i in range(20):
            ctx.add_message(Message(role="user", content=f"Message number {i} " * 10))
        result = ctx.build_prompt_context()
        assert len(result) <= 200  # Some overhead for truncation message


class TestDiversitySteering:
    def test_disabled_steering(self):
        steering = DiversitySteering(enabled=False)
        assert steering.get_steering_guidance(["tool1"]) == ""
        assert steering.get_exclude_tools() == []

    def test_record_and_stats(self):
        steering = DiversitySteering(enabled=True)
        steering.memory = None  # Disable mem0 for unit test

        conv = Conversation(
            conversation_id="test",
            tool_calls=[
                ToolCall(tool_name="T1", api_name="a1", arguments={}),
                ToolCall(tool_name="T2", api_name="a2", arguments={}),
            ],
            metadata={"categories": {"T1/a1": "Travel", "T2/a2": "Food"}, "categories_list": ["Travel", "Food"]},
        )
        steering.record_conversation(conv)
        stats = steering.get_usage_stats()
        assert stats["total_conversations"] == 1
        assert len(stats["tool_usage"]) > 0

    def test_steering_guidance_after_many_conversations(self):
        steering = DiversitySteering(enabled=True)
        steering.memory = None

        # Generate many conversations heavily using T1, with a few using T2
        for _ in range(10):
            conv = Conversation(
                tool_calls=[ToolCall(tool_name="T1", api_name="a1", arguments={})],
                metadata={"categories": {"T1/a1": "Travel"}, "categories_list": ["Travel"]},
            )
            steering.record_conversation(conv)

        # Add a few with a different tool so T1 becomes overrepresented relative to mean
        for _ in range(2):
            conv = Conversation(
                tool_calls=[ToolCall(tool_name="T2", api_name="a2", arguments={})],
                metadata={"categories": {"T2/a2": "Food"}, "categories_list": ["Food"]},
            )
            steering.record_conversation(conv)

        guidance = steering.get_steering_guidance(["T1/a1"])
        assert len(guidance) > 0  # Should provide some guidance about overuse or underrepresented domains


class TestDiversityMetrics:
    def _make_conv(self, tools, categories=None):
        tool_calls = [
            ToolCall(tool_name=t.split("/")[0], api_name=t.split("/")[1], arguments={})
            for t in tools
        ]
        return Conversation(
            tool_calls=tool_calls,
            metadata={"categories_list": categories or []},
        )

    def test_entropy_single_combination(self):
        convs = [self._make_conv(["T/a"]) for _ in range(10)]
        entropy = DiversityMetrics.tool_combination_entropy(convs)
        assert entropy == 0.0  # All same combination

    def test_entropy_diverse_combinations(self):
        convs = [
            self._make_conv(["T1/a1"]),
            self._make_conv(["T2/a2"]),
            self._make_conv(["T3/a3"]),
            self._make_conv(["T4/a4"]),
        ]
        entropy = DiversityMetrics.tool_combination_entropy(convs)
        assert entropy == 2.0  # log2(4) = 2.0

    def test_uniformity_single_category(self):
        convs = [self._make_conv(["T/a"], ["Travel"]) for _ in range(10)]
        uniformity = DiversityMetrics.domain_coverage_uniformity(convs)
        assert uniformity == 0.0  # Only one category

    def test_uniformity_balanced(self):
        convs = [
            self._make_conv(["T1/a1"], ["Travel"]),
            self._make_conv(["T2/a2"], ["Food"]),
            self._make_conv(["T3/a3"], ["Weather"]),
            self._make_conv(["T4/a4"], ["Finance"]),
        ]
        uniformity = DiversityMetrics.domain_coverage_uniformity(convs)
        assert uniformity == 1.0  # Perfectly uniform

    def test_unique_tool_ratio(self):
        convs = [
            self._make_conv(["T1/a1"]),
            self._make_conv(["T1/a1"]),
            self._make_conv(["T2/a2"]),
        ]
        ratio = DiversityMetrics.unique_tool_ratio(convs)
        assert abs(ratio - 2 / 3) < 0.01

    def test_summary(self):
        convs = [self._make_conv(["T1/a1"]), self._make_conv(["T2/a2"])]
        result = DiversityMetrics.summary(convs)
        assert "tool_combination_entropy" in result
        assert "domain_coverage_uniformity" in result
        assert result["total_conversations"] == 2

    def test_empty_list(self):
        assert DiversityMetrics.tool_combination_entropy([]) == 0.0
        assert DiversityMetrics.domain_coverage_uniformity([]) == 0.0
        assert DiversityMetrics.unique_tool_ratio([]) == 0.0
