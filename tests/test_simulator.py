"""Tests for the tool simulator module."""

import json
from unittest.mock import MagicMock

import pytest

from saplvl.models import ToolCall
from saplvl.simulator.executor import SessionState, ToolSimulator


class TestSessionState:
    def test_set_and_get(self):
        state = SessionState()
        state.set("hotel_id", "htl_123")
        assert state.get("hotel_id") == "htl_123"

    def test_get_missing(self):
        state = SessionState()
        assert state.get("missing") is None

    def test_get_all(self):
        state = SessionState()
        state.set("a", 1)
        state.set("b", 2)
        assert state.get_all() == {"a": 1, "b": 2}

    def test_add_response_extracts_ids(self):
        state = SessionState()
        state.add_response({
            "hotel_id": "htl_456",
            "name": "Grand Hotel",
            "booking_id": "bk_789",
        })
        assert state.get("hotel_id") == "htl_456"
        assert state.get("name") == "Grand Hotel"
        assert state.get("booking_id") == "bk_789"

    def test_add_response_extracts_from_nested(self):
        state = SessionState()
        state.add_response({
            "results": [{"id": "item_1", "name": "Test Item"}]
        })
        assert state.get("results.id") is not None or state.get("id") is not None

    def test_format_for_prompt_empty(self):
        state = SessionState()
        result = state.format_for_prompt()
        assert "No previous" in result

    def test_format_for_prompt_with_values(self):
        state = SessionState()
        state.set("hotel_id", "htl_123")
        result = state.format_for_prompt()
        assert "hotel_id" in result
        assert "htl_123" in result


class TestToolSimulator:
    def test_execute_unknown_endpoint(self, sample_registry):
        simulator = ToolSimulator(sample_registry)
        tc = ToolCall(tool_name="Unknown", api_name="unknown_ep", arguments={})
        session = SessionState()
        output = simulator.execute(tc, session)
        assert not output.success
        assert "error" in output.response

    def test_schema_fallback_search(self, sample_registry):
        simulator = ToolSimulator(sample_registry, rng=__import__("random").Random(42))
        tc = ToolCall(
            tool_name="HotelFinder",
            api_name="search_hotels",
            arguments={"city": "Paris", "max_price": 200},
        )
        session = SessionState()
        output = simulator.execute(tc, session)
        assert output.success
        assert "results" in output.response

    def test_schema_fallback_booking(self, sample_registry):
        simulator = ToolSimulator(sample_registry, rng=__import__("random").Random(42))
        tc = ToolCall(
            tool_name="HotelFinder",
            api_name="book_hotel",
            arguments={"hotel_id": "htl_123", "check_in": "2026-04-11"},
        )
        session = SessionState()
        output = simulator.execute(tc, session)
        assert output.success
        assert output.response.get("status") == "confirmed"

    def test_session_state_chains(self, sample_registry):
        simulator = ToolSimulator(sample_registry, rng=__import__("random").Random(42))
        session = SessionState()

        # First call: search
        tc1 = ToolCall(
            tool_name="HotelFinder",
            api_name="search_hotels",
            arguments={"city": "Paris"},
        )
        output1 = simulator.execute(tc1, session)
        assert output1.success

        # Session should have values from the response
        all_vals = session.get_all()
        assert len(all_vals) > 0

    def test_llm_mock(self, sample_registry, mock_openai_client):
        # Override the mock to return valid JSON
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            '{"results": [{"id": "htl_001", "name": "Test Hotel", "price": 150}]}'

        simulator = ToolSimulator(sample_registry, openai_client=mock_openai_client)
        tc = ToolCall(
            tool_name="HotelFinder",
            api_name="search_hotels",
            arguments={"city": "Paris"},
        )
        session = SessionState()
        output = simulator.execute(tc, session)
        assert output.success
        assert "results" in output.response
        mock_openai_client.chat.completions.create.assert_called_once()
