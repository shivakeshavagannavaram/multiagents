"""Tests for context management and diversity metrics."""

import pytest

from conv_gen.memory.context import ConversationContext
from conv_gen.memory.steering import DiversityMetrics, DiversitySteering
from conv_gen.models import (
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

    # ----- Primary metrics: tool usage entropy ------------------------------

    def test_tool_usage_entropy_uniform(self):
        """Uniform distribution of 4 tools with equal counts → normalized entropy = 1.0."""
        convs = [
            self._make_conv(["T1/a", "T2/a", "T3/a", "T4/a"]),
            self._make_conv(["T1/a", "T2/a", "T3/a", "T4/a"]),
        ]
        entropy = DiversityMetrics.tool_usage_entropy(convs)
        assert abs(entropy - 1.0) < 1e-9

    def test_tool_usage_entropy_monopoly(self):
        """Single tool called 20 times → entropy = 0.0 (one tool dominates)."""
        convs = [self._make_conv(["T1/a"] * 20)]
        entropy = DiversityMetrics.tool_usage_entropy(convs)
        assert entropy == 0.0

    def test_tool_usage_entropy_skewed(self):
        """80/20 split: one tool at 16/20, four tools at 1/20 each.

        Should be low (more concentrated) but > 0 (multiple tools exist).
        """
        convs = [
            self._make_conv(
                ["T1/a"] * 16 + ["T2/a", "T3/a", "T4/a", "T5/a"]
            )
        ]
        entropy = DiversityMetrics.tool_usage_entropy(convs)
        assert 0.0 < entropy < 0.7

    def test_tool_usage_entropy_empty(self):
        assert DiversityMetrics.tool_usage_entropy([]) == 0.0

    def test_tool_usage_entropy_single_tool_many_calls(self):
        """Single tool, many calls → still 0 (only one element in distribution)."""
        convs = [self._make_conv(["T1/a"] * 5), self._make_conv(["T1/a"] * 3)]
        assert DiversityMetrics.tool_usage_entropy(convs) == 0.0

    # ----- Primary metrics: unique tool coverage ----------------------------

    def test_unique_tools_used_counts_distinct_pairs(self):
        convs = [
            self._make_conv(["T1/a", "T1/b", "T2/a"]),
            self._make_conv(["T1/a", "T3/a"]),
        ]
        # Distinct (tool, api) pairs: (T1,a), (T1,b), (T2,a), (T3,a) = 4
        assert DiversityMetrics.unique_tools_used(convs) == 4

    def test_unique_tool_coverage_ratio(self):
        """3 unique tools against a 10-endpoint registry = 0.3 coverage."""
        convs = [
            self._make_conv(["T1/a"]),
            self._make_conv(["T2/a"]),
            self._make_conv(["T3/a"]),
        ]
        ratio = DiversityMetrics.unique_tool_coverage_ratio(convs, registry_size=10)
        assert abs(ratio - 0.3) < 1e-9

    def test_unique_tool_coverage_ratio_zero_registry(self):
        convs = [self._make_conv(["T1/a"])]
        assert DiversityMetrics.unique_tool_coverage_ratio(convs, registry_size=0) == 0.0

    # ----- Secondary metrics ------------------------------------------------

    def test_top_n_tool_concentration_dominated(self):
        """One tool gets 90% of calls → top-1 concentration = 0.9."""
        convs = [self._make_conv(["T1/a"] * 9 + ["T2/a"])]
        conc = DiversityMetrics.top_n_tool_concentration(convs, n=1)
        assert abs(conc - 0.9) < 1e-9

    def test_top_n_tool_concentration_spread(self):
        """5 tools used equally → top-5 concentration = 1.0 (all calls covered)."""
        convs = [self._make_conv(["T1/a", "T2/a", "T3/a", "T4/a", "T5/a"])]
        conc = DiversityMetrics.top_n_tool_concentration(convs, n=5)
        assert abs(conc - 1.0) < 1e-9

    def test_tool_combination_entropy_single_combination(self):
        """Chain-level entropy: all same chain → 0.0."""
        convs = [self._make_conv(["T/a"]) for _ in range(10)]
        assert DiversityMetrics.tool_combination_entropy(convs) == 0.0

    def test_tool_combination_entropy_diverse(self):
        """Chain-level entropy: 4 distinct chains → log2(4) = 2.0."""
        convs = [
            self._make_conv(["T1/a1"]),
            self._make_conv(["T2/a2"]),
            self._make_conv(["T3/a3"]),
            self._make_conv(["T4/a4"]),
        ]
        assert DiversityMetrics.tool_combination_entropy(convs) == 2.0

    def test_domain_coverage_uniformity_single_category(self):
        convs = [self._make_conv(["T/a"], ["Travel"]) for _ in range(10)]
        assert DiversityMetrics.domain_coverage_uniformity(convs) == 0.0

    def test_domain_coverage_uniformity_balanced(self):
        """4 categories, 1 conv each → normalized entropy = 1.0."""
        convs = [
            self._make_conv(["T1/a1"], ["Travel"]),
            self._make_conv(["T2/a2"], ["Food"]),
            self._make_conv(["T3/a3"], ["Weather"]),
            self._make_conv(["T4/a4"], ["Finance"]),
        ]
        assert abs(DiversityMetrics.domain_coverage_uniformity(convs) - 1.0) < 1e-9

    def test_unique_chain_ratio(self):
        """2 unique chain signatures over 3 conversations = 2/3."""
        convs = [
            self._make_conv(["T1/a1"]),
            self._make_conv(["T1/a1"]),
            self._make_conv(["T2/a2"]),
        ]
        ratio = DiversityMetrics.unique_chain_ratio(convs)
        assert abs(ratio - 2 / 3) < 1e-9

    # ----- Summary structure ------------------------------------------------

    def test_summary_structure(self):
        convs = [self._make_conv(["T1/a1"]), self._make_conv(["T2/a2"])]
        result = DiversityMetrics.summary(convs, registry_size=100)
        # Nested structure
        assert "primary" in result
        assert "secondary" in result
        assert "totals" in result
        # Primary metrics
        assert "tool_usage_entropy" in result["primary"]
        assert "unique_tools_used" in result["primary"]
        assert "unique_tool_coverage_ratio" in result["primary"]
        # Secondary metrics
        assert "top_5_tool_concentration" in result["secondary"]
        assert "tool_combination_entropy" in result["secondary"]
        assert "domain_coverage_uniformity" in result["secondary"]
        assert "unique_chain_ratio" in result["secondary"]
        # Totals
        assert result["totals"]["total_conversations"] == 2
        assert result["totals"]["registry_size"] == 100

    def test_summary_without_registry(self):
        """Summary omits coverage_ratio when registry_size not provided."""
        convs = [self._make_conv(["T1/a1"])]
        result = DiversityMetrics.summary(convs)
        assert "unique_tool_coverage_ratio" not in result["primary"]
        assert "unique_tools_used" in result["primary"]

    def test_empty_list(self):
        assert DiversityMetrics.tool_usage_entropy([]) == 0.0
        assert DiversityMetrics.unique_tools_used([]) == 0
        assert DiversityMetrics.unique_tool_coverage_ratio([], 100) == 0.0
        assert DiversityMetrics.top_n_tool_concentration([], n=5) == 0.0
        assert DiversityMetrics.tool_combination_entropy([]) == 0.0
        assert DiversityMetrics.domain_coverage_uniformity([]) == 0.0
        assert DiversityMetrics.unique_chain_ratio([]) == 0.0


class TestQualityMetrics:
    def _make_conv(self, tools, scores=None):
        from conv_gen.memory.steering import QualityMetrics  # noqa: F401
        tool_calls = [
            ToolCall(tool_name=t.split("/")[0], api_name=t.split("/")[1], arguments={})
            for t in tools
        ]
        return Conversation(
            tool_calls=tool_calls,
            judge_scores=scores,
        )

    def test_mean_scores_empty(self):
        from conv_gen.memory.steering import QualityMetrics
        result = QualityMetrics.mean_scores([])
        assert result["scored_count"] == 0
        assert result["naturalness"] == 0.0

    def test_mean_scores_basic(self):
        from conv_gen.memory.steering import QualityMetrics
        convs = [
            self._make_conv(["T1/a"], JudgeScore(naturalness=5, tool_correctness=4, task_completion=3)),
            self._make_conv(["T1/a"], JudgeScore(naturalness=3, tool_correctness=4, task_completion=5)),
        ]
        result = QualityMetrics.mean_scores(convs)
        assert result["scored_count"] == 2
        assert result["naturalness"] == 4.0
        assert result["tool_correctness"] == 4.0
        assert result["task_completion"] == 4.0
        assert result["overall_mean"] == 4.0

    def test_ms_mt_rate(self):
        from conv_gen.memory.steering import QualityMetrics
        convs = [
            self._make_conv(["T1/a", "T2/a", "T3/a"]),  # MS+MT ✓
            self._make_conv(["T1/a", "T1/a"]),  # 2 calls, 1 tool — no
            self._make_conv(["T1/a"]),  # 1 call — no
            self._make_conv(["T1/a", "T2/a", "T3/a", "T4/a"]),  # MS+MT ✓
        ]
        rate = QualityMetrics.ms_mt_rate(convs)
        assert rate == 0.5

    def test_ms_mt_rate_empty(self):
        from conv_gen.memory.steering import QualityMetrics
        assert QualityMetrics.ms_mt_rate([]) == 0.0
