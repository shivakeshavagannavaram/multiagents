"""User simulator agent that generates realistic user messages."""

from __future__ import annotations

import logging

import anthropic

from saplvl.agents.base import BaseAgent
from saplvl.memory.context import ConversationContext
from saplvl.models import Message

logger = logging.getLogger(__name__)

INITIAL_USER_PROMPT = """You are simulating a real user making a request to an AI assistant.

Scenario: {scenario}

Generate the user's FIRST message based on this scenario.

Guidelines:
- Be natural and conversational — write like a real person would
- You may be slightly vague or ambiguous (don't provide every detail upfront)
- Don't mention specific API names or technical endpoints
- Keep it to 1-3 sentences
{steering_guidance}

Respond with ONLY the user's message, nothing else."""

FOLLOWUP_USER_PROMPT = """You are simulating a real user in an ongoing conversation with an AI assistant.

Scenario: {scenario}

Conversation so far:
{conversation_history}

Generate the user's NEXT message.

Guidelines:
- If the assistant asked a clarifying question, answer it naturally
- If the assistant completed a step, you may ask for more or provide additional details
- If the task is mostly done, express satisfaction or ask a follow-up question
- Be natural — real users sometimes add new requirements mid-conversation
- Keep it to 1-3 sentences
{steering_guidance}

Respond with ONLY the user's message, nothing else."""


class UserSimulatorAgent(BaseAgent):
    """Generates realistic user messages using Claude.

    For the first turn, generates an initial request based on the scenario.
    For subsequent turns, generates follow-ups based on conversation history.
    Sometimes generates ambiguous requests to trigger disambiguation.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.client = client
        self.model = model

    def run(
        self,
        context: ConversationContext,
        scenario: str = "",
        steering_guidance: str = "",
        **kwargs,
    ) -> Message:
        """Generate the next user message."""
        turn_number = context.get_turn_count()

        if turn_number == 0:
            prompt = INITIAL_USER_PROMPT.format(
                scenario=scenario,
                steering_guidance=steering_guidance,
            )
        else:
            prompt = FOLLOWUP_USER_PROMPT.format(
                scenario=scenario,
                conversation_history=context.build_prompt_context(),
                steering_guidance=steering_guidance,
            )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        user_text = response.content[0].text.strip()
        logger.debug("UserSimulator generated: %s", user_text[:100])

        return Message(role="user", content=user_text)
