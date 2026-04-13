"""Tests for the JSONL wire format used for the generated dataset."""

import json

import pytest

from conv_gen.models import (
    Conversation,
    JudgeScore,
    Message,
    ToolCall,
    ToolOutput,
)
from conv_gen.output_format import (
    from_any_json,
    from_wire_dict,
    read_jsonl,
    to_wire_dict,
    to_wire_json,
    write_jsonl,
)


def _make_conv_with_tools():
    """Build a small Conversation mirroring the deliverable spec example."""
    call1 = ToolCall(
        tool_name="hotels",
        api_name="search",
        arguments={"city": "Paris", "max_price": 200, "currency": "EUR"},
    )
    call2 = ToolCall(
        tool_name="hotels",
        api_name="book",
        arguments={"hotel_id": "htl_881", "check_in": "2026-04-11"},
    )
    return Conversation(
        conversation_id="conv_0042",
        messages=[
            Message(role="user", content="Find me a hotel in Paris for next weekend"),
            Message(role="assistant", content="What's your budget range?"),
            Message(role="user", content="Under 200 euros per night"),
            Message(role="assistant", content=None, tool_calls=[call1]),
            Message(
                role="tool",
                content=None,
                tool_outputs=[
                    ToolOutput(
                        tool_call=call1,
                        response={
                            "results": [
                                {"id": "htl_881", "name": "Hotel du Marais", "price": 175}
                            ]
                        },
                        success=True,
                    )
                ],
            ),
            Message(role="assistant", content=None, tool_calls=[call2]),
            Message(
                role="tool",
                content=None,
                tool_outputs=[
                    ToolOutput(
                        tool_call=call2,
                        response={"booking_id": "bk_3391", "status": "confirmed"},
                        success=True,
                    )
                ],
            ),
            Message(
                role="assistant",
                content="I've booked Hotel du Marais for Apr 11. Confirmation: bk_3391.",
            ),
        ],
        tool_calls=[call1, call2],
        tool_outputs=[
            ToolOutput(
                tool_call=call1,
                response={"results": [{"id": "htl_881", "name": "Hotel du Marais", "price": 175}]},
                success=True,
            ),
            ToolOutput(
                tool_call=call2,
                response={"booking_id": "bk_3391", "status": "confirmed"},
                success=True,
            ),
        ],
        judge_scores=JudgeScore(
            naturalness=4.2, tool_correctness=4.8, task_completion=5.0,
            reasoning="clear chain",
        ),
        metadata={"seed": 42, "tools_used": ["hotels/search", "hotels/book"], "num_turns": 7},
    )


class TestToWireDict:
    def test_conversation_id_preserved(self):
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        assert wire["conversation_id"] == "conv_0042"

    def test_no_top_level_tool_arrays(self):
        """Wire format must not emit the top-level aggregated arrays."""
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        assert "tool_calls" not in wire
        assert "tool_outputs" not in wire

    def test_user_message_shape(self):
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        user_msgs = [m for m in wire["messages"] if m["role"] == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[0]["content"] == "Find me a hotel in Paris for next weekend"
        # User messages only carry role and content in the wire format
        assert set(user_msgs[0].keys()) == {"role", "content"}

    def test_assistant_text_message_shape(self):
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        # The first assistant message is a text clarification question
        clarification = wire["messages"][1]
        assert clarification["role"] == "assistant"
        assert clarification["content"] == "What's your budget range?"
        assert "tool_calls" not in clarification

    def test_assistant_tool_call_message_shape(self):
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        # Find the assistant message that carries the hotels/search call
        tool_call_msgs = [
            m for m in wire["messages"]
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        assert len(tool_call_msgs) == 2
        first = tool_call_msgs[0]
        assert first["content"] is None
        assert len(first["tool_calls"]) == 1
        tc = first["tool_calls"][0]
        # Endpoint is the combined tool_name/api_name string
        assert tc["endpoint"] == "hotels/search"
        # Arguments are preserved as a dict
        assert tc["arguments"] == {
            "city": "Paris", "max_price": 200, "currency": "EUR"
        }

    def test_tool_message_content_is_response_dict(self):
        """Tool messages store the response dict directly in `content`."""
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        tool_msgs = [m for m in wire["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        # First tool response — the hotels/search result
        assert tool_msgs[0]["content"] == {
            "results": [{"id": "htl_881", "name": "Hotel du Marais", "price": 175}]
        }
        # Second tool response — the booking confirmation
        assert tool_msgs[1]["content"] == {
            "booking_id": "bk_3391", "status": "confirmed"
        }

    def test_judge_scores_shape(self):
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        assert "judge_scores" in wire
        scores = wire["judge_scores"]
        assert scores["naturalness"] == 4.2
        assert scores["tool_correctness"] == 4.8
        assert scores["task_completion"] == 5.0
        assert scores["reasoning"] == "clear chain"

    def test_judge_scores_omitted_when_reasoning_empty(self):
        """Reasoning is omitted from the wire dict when it's the empty string
        to keep the wire format compact for unscored intermediate outputs."""
        conv = Conversation(
            messages=[Message(role="user", content="hi")],
            judge_scores=JudgeScore(
                naturalness=3.0, tool_correctness=3.0, task_completion=3.0,
                reasoning="",
            ),
        )
        wire = to_wire_dict(conv)
        # reasoning should be absent when empty
        assert "reasoning" not in wire["judge_scores"]

    def test_metadata_preserved(self):
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        assert wire["metadata"]["seed"] == 42
        assert wire["metadata"]["num_turns"] == 7

    def test_metadata_omitted_when_empty(self):
        conv = Conversation(messages=[Message(role="user", content="hi")])
        wire = to_wire_dict(conv)
        assert "metadata" not in wire

    def test_judge_scores_omitted_when_none(self):
        conv = Conversation(messages=[Message(role="user", content="hi")])
        wire = to_wire_dict(conv)
        assert "judge_scores" not in wire

    def test_parallel_tool_calls_expand_to_multiple_tool_messages(self):
        """N parallel tool calls → N consecutive tool messages in the wire format."""
        call_a = ToolCall(tool_name="a", api_name="get", arguments={})
        call_b = ToolCall(tool_name="b", api_name="get", arguments={})
        conv = Conversation(
            messages=[
                Message(role="user", content="do both"),
                Message(role="assistant", content=None, tool_calls=[call_a, call_b]),
                Message(
                    role="tool",
                    content=None,
                    tool_outputs=[
                        ToolOutput(tool_call=call_a, response={"a_result": 1}),
                        ToolOutput(tool_call=call_b, response={"b_result": 2}),
                    ],
                ),
            ],
        )
        wire = to_wire_dict(conv)
        # Expected: user, assistant (with 2 tool_calls), tool, tool
        assert [m["role"] for m in wire["messages"]] == [
            "user", "assistant", "tool", "tool"
        ]
        # Assistant has both calls
        assert len(wire["messages"][1]["tool_calls"]) == 2
        # Two consecutive tool messages, each holding one response
        assert wire["messages"][2]["content"] == {"a_result": 1}
        assert wire["messages"][3]["content"] == {"b_result": 2}


class TestFromWireDict:
    def test_round_trip_matches_original(self):
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        recovered = from_wire_dict(wire)

        assert recovered.conversation_id == conv.conversation_id
        assert len(recovered.tool_calls) == len(conv.tool_calls)
        assert recovered.tool_calls[0].tool_name == "hotels"
        assert recovered.tool_calls[0].api_name == "search"
        assert recovered.tool_calls[0].arguments == {
            "city": "Paris", "max_price": 200, "currency": "EUR",
        }
        assert recovered.judge_scores is not None
        assert recovered.judge_scores.naturalness == 4.2
        assert recovered.metadata["seed"] == 42

    def test_endpoint_split_preserves_slashes_in_api_name(self):
        """If an API name itself contains a slash, only split on the first one."""
        wire = {
            "conversation_id": "conv_test",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "endpoint": "Service/path/to/endpoint",
                            "arguments": {"x": 1},
                        }
                    ],
                },
                {"role": "tool", "content": {"ok": True}},
            ],
        }
        conv = from_wire_dict(wire)
        assert conv.tool_calls[0].tool_name == "Service"
        assert conv.tool_calls[0].api_name == "path/to/endpoint"

    def test_tool_response_becomes_tool_output(self):
        """Tool message content dict becomes a ToolOutput.response on load."""
        wire = {
            "conversation_id": "conv_test",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"endpoint": "t/a", "arguments": {}}],
                },
                {"role": "tool", "content": {"data": "value"}},
            ],
        }
        conv = from_wire_dict(wire)
        assert len(conv.tool_outputs) == 1
        assert conv.tool_outputs[0].response == {"data": "value"}
        # The back-reference points to the right tool call
        assert conv.tool_outputs[0].tool_call.tool_name == "t"
        assert conv.tool_outputs[0].tool_call.api_name == "a"

    def test_non_dict_tool_content_is_wrapped(self):
        """If a tool response is a string (not a dict), wrap it."""
        wire = {
            "conversation_id": "conv_test",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"endpoint": "t/a", "arguments": {}}],
                },
                {"role": "tool", "content": "plain string response"},
            ],
        }
        conv = from_wire_dict(wire)
        assert conv.tool_outputs[0].response == {"result": "plain string response"}

    def test_missing_judge_scores_and_metadata(self):
        wire = {
            "conversation_id": "conv_test",
            "messages": [{"role": "user", "content": "hi"}],
        }
        conv = from_wire_dict(wire)
        assert conv.judge_scores is None
        assert conv.metadata == {}


class TestAutoDetection:
    """The loader should accept both wire format and legacy internal format."""

    def test_detects_wire_format(self):
        conv = _make_conv_with_tools()
        wire_line = to_wire_json(conv)
        loaded = from_any_json(wire_line)
        assert loaded.conversation_id == conv.conversation_id
        assert len(loaded.tool_calls) == 2

    def test_detects_legacy_format(self):
        """Legacy format has top-level tool_calls/tool_outputs arrays."""
        conv = _make_conv_with_tools()
        legacy_line = conv.model_dump_json()  # old Pydantic-native dump
        loaded = from_any_json(legacy_line)
        assert loaded.conversation_id == conv.conversation_id
        assert len(loaded.tool_calls) == 2

    def test_real_run_file_is_loadable(self, tmp_path):
        """The existing pre-deliverable run_a.jsonl file loads unchanged.

        Skipped automatically if the file isn't present (e.g., in a
        fresh checkout without cached runs).
        """
        from pathlib import Path
        run_a = Path("output/run_no_steering.jsonl")
        if not run_a.exists():
            pytest.skip("output/run_no_steering.jsonl not present")
        convs = read_jsonl(run_a)
        assert len(convs) > 0
        # All records should have parsed into internal Conversation objects
        for c in convs[:5]:
            assert isinstance(c, Conversation)
            assert c.conversation_id.startswith("conv_")


class TestJsonlHelpers:
    def test_write_and_read_round_trip(self, tmp_path):
        path = tmp_path / "test.jsonl"
        convs = [_make_conv_with_tools()]
        write_jsonl(path, convs)

        # File exists and contains exactly one line
        assert path.exists()
        content = path.read_text()
        assert content.count("\n") == 1

        # And it's a wire-format record
        parsed = json.loads(content.strip())
        assert "tool_calls" not in parsed  # top-level absent
        assert parsed["conversation_id"] == "conv_0042"

        # read_jsonl recovers the internal Conversation
        loaded = read_jsonl(path)
        assert len(loaded) == 1
        assert loaded[0].conversation_id == "conv_0042"
        assert len(loaded[0].tool_calls) == 2


class TestMatchesSpecExample:
    """Structural checks against the exact shape given in the deliverable spec."""

    def test_tool_call_has_endpoint_and_arguments(self):
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        # Find an assistant message with tool_calls
        tool_call_msgs = [
            m for m in wire["messages"]
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        assert tool_call_msgs, "expected at least one tool call assistant message"
        first_call = tool_call_msgs[0]["tool_calls"][0]
        # Exact two keys required by the spec example
        assert set(first_call.keys()) == {"endpoint", "arguments"}
        # endpoint is a single string, not a {tool_name, api_name} pair
        assert isinstance(first_call["endpoint"], str)
        assert isinstance(first_call["arguments"], dict)

    def test_tool_message_has_content_dict_only(self):
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        tool_msgs = [m for m in wire["messages"] if m["role"] == "tool"]
        assert tool_msgs
        # Tool messages carry only role + content, and content is the
        # response dict — no separate tool_outputs/tool_call fields.
        for m in tool_msgs:
            assert set(m.keys()) == {"role", "content"}
            assert isinstance(m["content"], dict)

    def test_judge_scores_has_three_dimensions(self):
        conv = _make_conv_with_tools()
        wire = to_wire_dict(conv)
        scores = wire["judge_scores"]
        for dim in ("naturalness", "tool_correctness", "task_completion"):
            assert dim in scores
            assert isinstance(scores[dim], (int, float))
            assert 1.0 <= scores[dim] <= 5.0
