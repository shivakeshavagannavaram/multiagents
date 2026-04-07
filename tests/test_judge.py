"""Tests for the LLM-as-judge module."""

import json
from unittest.mock import MagicMock

import pytest

from saplvl.judgellm.judge import JudgeLLM
from saplvl.models import Conversation, JudgeScore, Message, ToolCall, ToolOutput


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
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")

        judge = JudgeLLM(client)
        scores = judge.batch_score([sample_conversation])
        assert len(scores) == 1
        assert scores[0].naturalness == 3.0  # Default fallback
