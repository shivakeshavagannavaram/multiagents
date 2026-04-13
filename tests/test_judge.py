"""Tests for the LLM-as-judge module."""

import json
from unittest.mock import MagicMock

import pytest

from conv_gen.judgellm.judge import JudgeLLM
from conv_gen.models import Conversation, JudgeScore, Message, ToolCall, ToolOutput


class TestJudgeLLM:
    def test_score_conversation(self, mock_openai_client, sample_conversation):
        judge = JudgeLLM(mock_openai_client)
        score = judge.score(sample_conversation)
        assert isinstance(score, JudgeScore)
        assert 1.0 <= score.naturalness <= 5.0
        assert 1.0 <= score.tool_correctness <= 5.0
        assert 1.0 <= score.task_completion <= 5.0

    def test_score_calls_openai(self, mock_openai_client, sample_conversation):
        judge = JudgeLLM(mock_openai_client)
        judge.score(sample_conversation)
        mock_openai_client.chat.completions.create.assert_called_once()

        call_kwargs = mock_openai_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}

    def test_batch_score(self, mock_openai_client, sample_conversation):
        judge = JudgeLLM(mock_openai_client)
        scores = judge.batch_score([sample_conversation, sample_conversation])
        assert len(scores) == 2

    def test_clamp_values(self):
        assert JudgeLLM._clamp(0.5) == 1.0
        assert JudgeLLM._clamp(6.0) == 5.0
        assert JudgeLLM._clamp(3.5) == 3.5
        assert JudgeLLM._clamp("invalid") == 3.0

    def test_format_conversation(self, sample_conversation):
        formatted = JudgeLLM._format_conversation(sample_conversation)
        assert "USER:" in formatted
        assert "ASSISTANT" in formatted
        assert "TOOL CALL" in formatted or "TOOL RESPONSE" in formatted

    def test_handles_scoring_failure(self, sample_conversation):
        """When the judge API fails, batch_score returns a fallback score
        below the default quality_threshold (3.0) so the orchestrator's
        retry loop kicks in on the next attempt. The fallback was
        intentionally lowered from 3.0 to 2.0 so failed scoring doesn't
        look like acceptable output.
        """
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")

        judge = JudgeLLM(client)
        scores = judge.batch_score([sample_conversation])
        assert len(scores) == 1
        # Fallback scores are 2.0 on all dimensions — below the default
        # 3.0 threshold so the orchestrator treats failure as "quality too low".
        assert scores[0].naturalness == 2.0
        assert scores[0].tool_correctness == 2.0
        assert scores[0].task_completion == 2.0
        # The reasoning field carries the exception message so humans
        # debugging a run can see what went wrong.
        assert "API error" in scores[0].reasoning


class TestNormalizeFieldName:
    """The normalizer needs to handle camelCase / PascalCase / snake_case /
    kebab-case so the chain-break check can match fields that tools name
    differently."""

    def test_camel_case_to_snake(self):
        assert JudgeLLM._normalize_field_name("userId") == "user_id"
        assert JudgeLLM._normalize_field_name("conversationId") == "conversation_id"
        assert JudgeLLM._normalize_field_name("hotelPrice") == "hotel_price"

    def test_pascal_case(self):
        assert JudgeLLM._normalize_field_name("UserId") == "user_id"
        assert JudgeLLM._normalize_field_name("HTMLParser") == "html_parser"
        assert JudgeLLM._normalize_field_name("UserID") == "user_id"

    def test_snake_case_passthrough(self):
        assert JudgeLLM._normalize_field_name("user_id") == "user_id"
        assert JudgeLLM._normalize_field_name("hotel_id") == "hotel_id"

    def test_kebab_case(self):
        assert JudgeLLM._normalize_field_name("user-id") == "user_id"
        assert JudgeLLM._normalize_field_name("conversation-id") == "conversation_id"

    def test_vendor_prefix_is_preserved_not_stripped(self):
        """Vendor-specific field names stay distinct after normalization.

        This is important: intercomUserId is semantically different from
        a generic user_id, because the identifier namespace is different.
        If we collapsed them, the chain-break check would give false
        positives when matching across services.
        """
        assert JudgeLLM._normalize_field_name("intercomUserId") == "intercom_user_id"
        assert JudgeLLM._normalize_field_name("intercom_user_id") == "intercom_user_id"
        # And that's different from plain user_id
        assert JudgeLLM._normalize_field_name("intercomUserId") != JudgeLLM._normalize_field_name("userId")

    def test_empty_and_leading_trailing_underscores(self):
        assert JudgeLLM._normalize_field_name("") == ""
        assert JudgeLLM._normalize_field_name("   ") == ""
        assert JudgeLLM._normalize_field_name("_x_") == "x"


class TestCollectFieldValues:
    """The field-value collector builds the {field_name → values} map
    the chain-break check consumes."""

    def test_flat_dict(self):
        out = {}
        JudgeLLM._collect_field_values(
            {"hotel_id": "htl_a8f2", "name": "Hotel X", "price": 175}, out
        )
        assert out["hotel_id"] == {"htl_a8f2"}
        assert out["name"] == {"Hotel X"}
        assert out["price"] == {"175"}  # numeric → string-coerced

    def test_nested_dict_walked(self):
        out = {}
        JudgeLLM._collect_field_values(
            {"outer": {"innerId": "abc"}, "top_id": "xyz"}, out
        )
        assert out["inner_id"] == {"abc"}
        assert out["top_id"] == {"xyz"}

    def test_list_of_dicts_flattened(self):
        """A list of dicts (like a search result array) contributes its
        fields to the top-level map, so later calls can reference any of
        them."""
        out = {}
        JudgeLLM._collect_field_values(
            {"results": [
                {"flight_id": "fl_001", "price": 200},
                {"flight_id": "fl_002", "price": 300},
            ]},
            out,
        )
        assert out["flight_id"] == {"fl_001", "fl_002"}
        assert out["price"] == {"200", "300"}

    def test_camel_case_normalized_on_collection(self):
        out = {}
        JudgeLLM._collect_field_values({"bookingId": "bk_999"}, out)
        assert "booking_id" in out
        assert out["booking_id"] == {"bk_999"}

    def test_empty_strings_skipped(self):
        out = {}
        JudgeLLM._collect_field_values({"id": "", "name": "Thing"}, out)
        assert "id" not in out
        assert out["name"] == {"Thing"}

    def test_booleans_skipped(self):
        out = {}
        JudgeLLM._collect_field_values({"active": True, "user_id": "u_1"}, out)
        assert "active" not in out
        assert out["user_id"] == {"u_1"}


class TestChainBreakCheck:
    """The main check — field-name-aware hallucination detection.

    These tests exercise _apply_structural_checks directly, bypassing
    the LLM call, so we can pin down the check's behavior on carefully
    crafted conversations.
    """

    def _judge(self):
        return JudgeLLM(MagicMock())

    def _base_scores(self):
        return JudgeScore(naturalness=5.0, tool_correctness=5.0, task_completion=5.0)

    def test_legitimate_fabrication_is_not_flagged(self):
        """Situation B: no prior response contained the field the assistant
        fabricates. Rule #11(b) says this is correct behavior. The check
        must NOT fire and tool_correctness must NOT be capped.
        """
        judge = self._judge()
        # Call 1 searches; its response has hotel_id.
        # Call 2 creates a reservation with admin_id — a field the prior
        # response DIDN'T have. Fabricated per rule #11.
        call1 = ToolCall(tool_name="hotels", api_name="search", arguments={"city": "Paris"})
        call2 = ToolCall(
            tool_name="admin",
            api_name="create_reservation",
            arguments={"admin_id": "admin_sarah_csm", "city": "Paris"},
        )
        conv = Conversation(
            conversation_id="conv_fab",
            messages=[
                Message(role="user", content="Book me a Paris hotel"),
                Message(role="assistant", content=None, tool_calls=[call1]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(
                        tool_call=call1,
                        response={"hotel_id": "htl_a8f2", "name": "Hotel X"},
                    )
                ]),
                Message(role="assistant", content=None, tool_calls=[call2]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(tool_call=call2, response={"reservation_id": "rv_1"}),
                ]),
            ],
            tool_calls=[call1, call2],
            tool_outputs=[
                ToolOutput(tool_call=call1, response={"hotel_id": "htl_a8f2", "name": "Hotel X"}),
                ToolOutput(tool_call=call2, response={"reservation_id": "rv_1"}),
            ],
            metadata={"tools_planned": 2},
        )
        result = judge._apply_structural_checks(conv, self._base_scores())
        # tool_correctness should remain 5.0 — no chain break happened
        assert result.tool_correctness == 5.0
        assert "Chain break" not in result.reasoning

    def test_chain_break_is_flagged(self):
        """Situation A: prior response produced hotel_id='htl_a8f2', but
        the assistant calls the next tool with hotel_id='htl_wrong'.
        The check MUST fire and cap tool_correctness at 2.5.
        """
        judge = self._judge()
        call1 = ToolCall(tool_name="hotels", api_name="search", arguments={"city": "Paris"})
        call2 = ToolCall(
            tool_name="hotels",
            api_name="book",
            arguments={"hotel_id": "htl_wrong", "check_in": "2026-04-11"},
        )
        conv = Conversation(
            conversation_id="conv_break",
            messages=[
                Message(role="user", content="Book a hotel"),
                Message(role="assistant", content=None, tool_calls=[call1]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(
                        tool_call=call1,
                        response={"hotel_id": "htl_a8f2", "name": "Hotel X"},
                    )
                ]),
                Message(role="assistant", content=None, tool_calls=[call2]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(tool_call=call2, response={"booking_id": "bk_1"}),
                ]),
            ],
            tool_calls=[call1, call2],
            tool_outputs=[
                ToolOutput(tool_call=call1, response={"hotel_id": "htl_a8f2", "name": "Hotel X"}),
                ToolOutput(tool_call=call2, response={"booking_id": "bk_1"}),
            ],
            metadata={"tools_planned": 2},
        )
        result = judge._apply_structural_checks(conv, self._base_scores())
        # Chain break → capped at 2.5
        assert result.tool_correctness == 2.5
        assert "Chain break" in result.reasoning

    def test_chain_propagation_accepted(self):
        """The happy path: the assistant correctly uses a prior response's
        hotel_id in the next call. No flag.
        """
        judge = self._judge()
        call1 = ToolCall(tool_name="hotels", api_name="search", arguments={"city": "Paris"})
        call2 = ToolCall(
            tool_name="hotels",
            api_name="book",
            arguments={"hotel_id": "htl_a8f2", "check_in": "2026-04-11"},
        )
        conv = Conversation(
            conversation_id="conv_good",
            messages=[
                Message(role="user", content="Book a hotel"),
                Message(role="assistant", content=None, tool_calls=[call1]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(
                        tool_call=call1,
                        response={"hotel_id": "htl_a8f2"},
                    )
                ]),
                Message(role="assistant", content=None, tool_calls=[call2]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(tool_call=call2, response={"booking_id": "bk_1"}),
                ]),
            ],
            tool_calls=[call1, call2],
            tool_outputs=[
                ToolOutput(tool_call=call1, response={"hotel_id": "htl_a8f2"}),
                ToolOutput(tool_call=call2, response={"booking_id": "bk_1"}),
            ],
            metadata={"tools_planned": 2},
        )
        result = judge._apply_structural_checks(conv, self._base_scores())
        assert result.tool_correctness == 5.0
        assert "Chain break" not in result.reasoning

    def test_user_provided_value_accepted(self):
        """If the user explicitly provides an ID in their message, the
        assistant can use it in a later tool call even without a prior
        response containing that field. This also should not flag."""
        judge = self._judge()
        call1 = ToolCall(
            tool_name="lookup", api_name="get",
            arguments={"q": "test"},
        )
        call2 = ToolCall(
            tool_name="actions", api_name="update",
            arguments={"ticket_id": "TKT-99999"},
        )
        conv = Conversation(
            conversation_id="conv_user_id",
            messages=[
                Message(role="user", content="Update ticket TKT-99999 please"),
                Message(role="assistant", content=None, tool_calls=[call1]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(tool_call=call1, response={"status": "ok"}),
                ]),
                Message(role="assistant", content=None, tool_calls=[call2]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(tool_call=call2, response={"updated": True}),
                ]),
            ],
            tool_calls=[call1, call2],
            tool_outputs=[
                ToolOutput(tool_call=call1, response={"status": "ok"}),
                ToolOutput(tool_call=call2, response={"updated": True}),
            ],
            metadata={"tools_planned": 2},
        )
        result = judge._apply_structural_checks(conv, self._base_scores())
        # prior response had "status" but no "ticket_id" — user said
        # TKT-99999 — so this is a legitimate user-provided ID, not a
        # chain break.
        assert result.tool_correctness == 5.0

    def test_first_call_never_flagged(self):
        """Even if the first call's argument has no origin, we don't flag
        it — the check only applies to call 2 onward. Rule #11 allows
        first-call fabrication unconditionally."""
        judge = self._judge()
        call1 = ToolCall(
            tool_name="intercom", api_name="reply",
            arguments={"intercomUserId": "customer_delayed_shipment"},
        )
        conv = Conversation(
            conversation_id="conv_1",
            messages=[
                Message(role="user", content="Reply to that customer"),
                Message(role="assistant", content=None, tool_calls=[call1]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(tool_call=call1, response={"sent": True}),
                ]),
            ],
            tool_calls=[call1],
            tool_outputs=[ToolOutput(tool_call=call1, response={"sent": True})],
            metadata={"tools_planned": 1},
        )
        result = judge._apply_structural_checks(conv, self._base_scores())
        assert result.tool_correctness == 5.0

    def test_camel_case_field_matches_snake_case_prior(self):
        """Field normalization must work: a prior response producing
        ``hotel_id`` should match a subsequent call arg named ``hotelId``
        and allow the value to chain cleanly."""
        judge = self._judge()
        call1 = ToolCall(tool_name="hotels", api_name="search", arguments={"city": "Paris"})
        call2 = ToolCall(
            tool_name="hotels",
            api_name="book",
            arguments={"hotelId": "htl_a8f2"},  # camelCase key, same value
        )
        conv = Conversation(
            conversation_id="conv_camel",
            messages=[
                Message(role="user", content="Book a hotel"),
                Message(role="assistant", content=None, tool_calls=[call1]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(
                        tool_call=call1,
                        response={"hotel_id": "htl_a8f2"},  # snake_case key
                    )
                ]),
                Message(role="assistant", content=None, tool_calls=[call2]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(tool_call=call2, response={"booking_id": "bk_1"}),
                ]),
            ],
            tool_calls=[call1, call2],
            tool_outputs=[
                ToolOutput(tool_call=call1, response={"hotel_id": "htl_a8f2"}),
                ToolOutput(tool_call=call2, response={"booking_id": "bk_1"}),
            ],
            metadata={"tools_planned": 2},
        )
        result = judge._apply_structural_checks(conv, self._base_scores())
        # No chain break — hotelId normalizes to hotel_id, matches prior response's hotel_id
        assert result.tool_correctness == 5.0

    def test_default_and_empty_values_skipped(self):
        """The check explicitly skips 'default' and '' values, which
        come from the credential-stripping fallback path. They should
        never trigger a chain-break penalty."""
        judge = self._judge()
        call1 = ToolCall(tool_name="t1", api_name="a", arguments={})
        call2 = ToolCall(
            tool_name="t2",
            api_name="a",
            arguments={"api_key": "default", "session_id": ""},
        )
        conv = Conversation(
            conversation_id="conv_default",
            messages=[
                Message(role="user", content="go"),
                Message(role="assistant", content=None, tool_calls=[call1]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(tool_call=call1, response={"session_id": "real_sid"}),
                ]),
                Message(role="assistant", content=None, tool_calls=[call2]),
                Message(role="tool", content=None, tool_outputs=[
                    ToolOutput(tool_call=call2, response={"ok": True}),
                ]),
            ],
            tool_calls=[call1, call2],
            tool_outputs=[
                ToolOutput(tool_call=call1, response={"session_id": "real_sid"}),
                ToolOutput(tool_call=call2, response={"ok": True}),
            ],
            metadata={"tools_planned": 2},
        )
        result = judge._apply_structural_checks(conv, self._base_scores())
        # 'default' and '' are explicit skips. The real session_id field
        # from prior response goes unmatched but that's fine because
        # '' is skipped.
        assert result.tool_correctness == 5.0
