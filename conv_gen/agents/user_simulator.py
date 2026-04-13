"""User simulator agent that generates realistic user messages."""

from __future__ import annotations

import logging

import anthropic

from conv_gen.agents.base import BaseAgent
from conv_gen.memory.context import ConversationContext
from conv_gen.models import Message

logger = logging.getLogger(__name__)


def build_user_visible_history(messages: list[Message]) -> str:
    """User-visible history only — no tool calls, no tool outputs."""
    lines = []
    for msg in messages:
        if msg.role == "user" and msg.content:
            lines.append(f"You: {msg.content}")
        elif msg.role == "assistant" and msg.content:
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines) if lines else "No messages yet."


def build_completion_guidance(
    tools_remaining_count: int,
    tools_completed_count: int,
    conversation_type: str = "",
) -> str:
    """Concrete-count status line telling the user to continue or wrap up."""
    total = tools_remaining_count + tools_completed_count

    if tools_remaining_count <= 0:
        return (
            "STATUS: ALL TASKS COMPLETE.\n"
            "Say a natural thank you and goodbye in one sentence.\n"
            "Do NOT ask for anything else.\n"
            "Do NOT introduce new topics."
        )

    if conversation_type in ("quick_lookup", "single"):
        return (
            f"STATUS: IN PROGRESS ({tools_completed_count}/{total} tasks done).\n"
            "React naturally to the assistant's last message.\n"
            "Do NOT wrap up or say goodbye yet."
        )

    return (
        f"STATUS: IN PROGRESS ({tools_completed_count}/{total} tasks done).\n"
        f"{tools_remaining_count} task(s) still to complete.\n"
        "Keep the conversation going naturally.\n"
        "Do NOT say goodbye until every task is done."
    )


INITIAL_USER_PROMPT = """You are simulating a real person making a request to an AI assistant.

SCENARIO:
{scenario}

YOUR TASK:
Generate the user's FIRST message to start this conversation.

RULES — follow all of these:
1. Ask for ONE thing only — a single clear request
2. Do NOT mention API names, tool names, or technical endpoints
3. Do NOT provide API keys, tokens, or account IDs unless the assistant asks
4. Do NOT ask multiple questions at once
5. Keep it to 1-2 sentences maximum
6. Write naturally — vary tone (casual, professional, rushed, detailed) to feel like different people
{steering_guidance}

Respond with ONLY the user's first message. Nothing else."""

FOLLOWUP_USER_PROMPT = """You are simulating a real person in an ongoing conversation with an AI assistant.

SCENARIO:
{scenario}

CONVERSATION SO FAR (what you have seen and said):
{user_visible_history}

TURN: {turn_number}

COMPLETION STATUS:
{plan_status}

YOUR TASK:
Generate your next message as the user.

GROUNDING RULES — these are critical:
1. You ONLY know what the assistant explicitly told you in their messages
2. If the assistant did not state a value, you do NOT know that value
3. NEVER use IDs, names, prices, or numbers the assistant did not tell you
4. If you need a value the assistant has not provided — ASK for it

CONVERSATION RULES:
5. Ask ONE thing at a time
6. If the assistant asked you a question — answer with just the value, no filler
7. Do NOT repeat or echo values back to the assistant
8. Do NOT say goodbye or wrap up unless completion status says ALL TASKS COMPLETE
9. Keep to 1 sentence. Maximum 2 if providing a specific value
{steering_guidance}

Respond with ONLY your next message. Nothing else."""


class UserSimulatorAgent(BaseAgent):
    """Generates realistic user messages using Claude."""

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
        plan_status: str = "",
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
                user_visible_history=build_user_visible_history(context.messages),
                turn_number=turn_number + 1,
                plan_status=plan_status or "React naturally to what the assistant said.",
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
