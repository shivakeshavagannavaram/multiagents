"""LLM-as-judge for conversation quality scoring."""

from __future__ import annotations

import json
import logging

import openai

from saplvl.models import Conversation, JudgeScore

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert evaluator of AI assistant conversations that involve tool use.

Score the following conversation on three dimensions, each on a scale of 1-5.

## Scoring Rubric

### Naturalness (1-5)
How natural and human-like does the conversation feel?
- 5: Indistinguishable from a real user-assistant interaction; natural flow, appropriate tone
- 4: Mostly natural with minor awkwardness
- 3: Functional but somewhat robotic or formulaic
- 2: Clearly artificial; unnatural phrasing or flow
- 1: Incoherent or nonsensical conversation

### Tool Correctness (1-5)
Are tools selected and called correctly?
- 5: All tool calls use correct endpoints, valid arguments, and proper sequencing; arguments reference real values from previous responses
- 4: Minor issues (e.g., one optional parameter missing) but fundamentally correct
- 3: Tools are appropriate but some arguments are incorrect or hallucinated
- 2: Wrong tools selected or significant argument errors
- 1: Tool calls are invalid, nonsensical, or completely wrong

### Task Completion (1-5)
Does the conversation successfully complete the user's goal?
- 5: User's request fully addressed with clear, helpful final response
- 4: Task mostly completed with minor gaps
- 3: Partial completion — some steps done but user goal not fully met
- 2: Significant parts of the request unaddressed
- 1: Task abandoned or completely wrong result

## Conversation to Evaluate

{conversation}

## Tools Used
{tools_info}

Respond with ONLY a JSON object in this exact format:
{{
    "naturalness": <float 1-5>,
    "tool_correctness": <float 1-5>,
    "task_completion": <float 1-5>,
    "reasoning": "<brief explanation of scores>"
}}"""


class JudgeLLM:
    """Scores conversations using OpenAI as an LLM-as-judge.

    Evaluates on three dimensions:
    - naturalness: How natural the conversation feels
    - tool_correctness: Correctness of tool selection and arguments
    - task_completion: Whether the user's goal was achieved
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "gpt-4o",
    ):
        self.client = client
        self.model = model

    def score(self, conversation: Conversation) -> JudgeScore:
        """Score a single conversation."""
        conv_text = self._format_conversation(conversation)
        tools_info = self._format_tools(conversation)

        prompt = JUDGE_PROMPT.format(
            conversation=conv_text,
            tools_info=tools_info,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=500,
        )

        raw = json.loads(response.choices[0].message.content)

        return JudgeScore(
            naturalness=self._clamp(raw.get("naturalness", 3.0)),
            tool_correctness=self._clamp(raw.get("tool_correctness", 3.0)),
            task_completion=self._clamp(raw.get("task_completion", 3.0)),
            reasoning=raw.get("reasoning", ""),
        )

    def batch_score(self, conversations: list[Conversation]) -> list[JudgeScore]:
        """Score multiple conversations sequentially."""
        scores = []
        for i, conv in enumerate(conversations):
            try:
                score = self.score(conv)
                scores.append(score)
                logger.info(
                    "Scored %d/%d: naturalness=%.1f, tool_correctness=%.1f, task_completion=%.1f",
                    i + 1, len(conversations),
                    score.naturalness, score.tool_correctness, score.task_completion,
                )
            except Exception as e:
                logger.warning("Failed to score conversation %d: %s", i, e)
                scores.append(JudgeScore(
                    naturalness=3.0, tool_correctness=3.0, task_completion=3.0,
                    reasoning=f"Scoring failed: {e}",
                ))
        return scores

    @staticmethod
    def _format_conversation(conversation: Conversation) -> str:
        """Format conversation messages for the judge prompt."""
        lines = []
        for msg in conversation.messages:
            role = msg.role.upper()
            if msg.content:
                lines.append(f"{role}: {msg.content}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(
                        f"{role} [TOOL CALL]: {tc.tool_name}/{tc.api_name}"
                        f"({json.dumps(tc.arguments)})"
                    )
            if msg.tool_outputs:
                for to in msg.tool_outputs:
                    lines.append(
                        f"TOOL RESPONSE [{to.tool_call.tool_name}/{to.tool_call.api_name}]: "
                        f"{json.dumps(to.response)}"
                    )
        return "\n".join(lines)

    @staticmethod
    def _format_tools(conversation: Conversation) -> str:
        """Format tool usage summary."""
        if not conversation.tool_calls:
            return "No tools were used."

        lines = [f"Tools called ({len(conversation.tool_calls)} total):"]
        for tc in conversation.tool_calls:
            lines.append(f"  - {tc.tool_name}/{tc.api_name}: args={json.dumps(tc.arguments)}")
        return "\n".join(lines)

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp a score to [1.0, 5.0]."""
        try:
            return max(1.0, min(5.0, float(value)))
        except (TypeError, ValueError):
            return 3.0
