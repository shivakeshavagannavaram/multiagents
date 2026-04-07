"""Within-conversation context management."""

from __future__ import annotations

import json
from typing import Any

from saplvl.models import Message, ToolOutput


class ConversationContext:
    """Manages conversation history and tool outputs for within-conversation grounding.

    Ensures that later tool calls can reference values from earlier outputs,
    and that agent prompts include relevant conversation history.
    """

    def __init__(self, max_history_chars: int = 16000):
        self._messages: list[Message] = []
        self._tool_outputs: list[ToolOutput] = []
        self._available_values: dict[str, Any] = {}
        self.max_history_chars = max_history_chars

    def add_message(self, msg: Message) -> None:
        self._messages.append(msg)

    def add_tool_output(self, output: ToolOutput) -> None:
        self._tool_outputs.append(output)
        self._extract_values(output.response)

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def tool_outputs(self) -> list[ToolOutput]:
        return list(self._tool_outputs)

    def build_prompt_context(self) -> str:
        """Format conversation history for injection into agent prompts.

        Truncates oldest messages if the total exceeds max_history_chars.
        """
        parts = []
        for msg in self._messages:
            role = msg.role.upper()
            content = msg.content or ""

            if msg.tool_calls:
                calls_str = ", ".join(
                    f"{tc.tool_name}/{tc.api_name}({json.dumps(tc.arguments)})"
                    for tc in msg.tool_calls
                )
                content += f"\n[Tool calls: {calls_str}]"

            if msg.tool_outputs:
                for to in msg.tool_outputs:
                    content += f"\n[Tool response from {to.tool_call.tool_name}/{to.tool_call.api_name}: {json.dumps(to.response)}]"

            parts.append(f"{role}: {content}")

        full_history = "\n\n".join(parts)

        # Truncate from the beginning if too long, keeping recent context
        if len(full_history) > self.max_history_chars:
            full_history = "... (earlier conversation truncated) ...\n\n" + \
                full_history[-self.max_history_chars:]

        return full_history

    def get_available_values(self) -> dict[str, Any]:
        """Return all extractable values from tool outputs for grounding."""
        return dict(self._available_values)

    def format_available_values(self) -> str:
        """Format available values as a string for prompts."""
        if not self._available_values:
            return "No values available from previous tool calls."
        lines = ["Values from previous tool calls:"]
        for key, value in self._available_values.items():
            lines.append(f"  - {key}: {json.dumps(value)}")
        return "\n".join(lines)

    def _extract_values(self, data: dict[str, Any], prefix: str = "") -> None:
        """Extract reusable values from a tool response."""
        for key, value in data.items():
            if isinstance(value, dict):
                self._extract_values(value, f"{prefix}{key}.")
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    self._extract_values(value[0], f"{prefix}{key}[0].")
                elif value:
                    self._available_values[f"{prefix}{key}"] = value[0]
            elif value is not None:
                self._available_values[f"{prefix}{key}"] = value

    def get_last_user_message(self) -> str | None:
        """Get the most recent user message content."""
        for msg in reversed(self._messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return None

    def get_turn_count(self) -> int:
        return len(self._messages)
