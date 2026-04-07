"""Assistant agent that generates responses with tool calls using structured output."""

from __future__ import annotations

import json
import logging
import re

import anthropic

from saplvl.agents.base import BaseAgent
from saplvl.ingestor.registry import ToolRegistry
from saplvl.memory.context import ConversationContext
from saplvl.models import Message, ToolCall

logger = logging.getLogger(__name__)

ASSISTANT_SYSTEM_PROMPT = """You are a helpful AI assistant that can use tools to accomplish tasks.

When a user makes a request, you should:
1. If the request is ambiguous or missing required information, ask a clarifying question FIRST
2. Otherwise, use the available tools to fulfill the request
3. After receiving tool results, summarize them for the user in a natural way
4. If multiple tools are needed, use them in a logical sequence

{available_values}

Important:
- Use actual values from previous tool results (don't make up IDs or data)
- Be concise but helpful in your responses
- Ask clarifying questions when intent is genuinely ambiguous or required parameters are missing"""


class AssistantAgent(BaseAgent):
    """Generates assistant responses with tool calls using Claude's native tool_use.

    This agent uses structured output (Claude tool_use) which produces validated
    JSON arguments against the tool's input schema.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        registry: ToolRegistry,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.client = client
        self.registry = registry
        self.model = model

    def run(
        self,
        context: ConversationContext,
        available_tools: list[tuple[str, str]] | None = None,
        **kwargs,
    ) -> Message:
        """Generate an assistant response, potentially with tool calls.

        Uses Claude's native tool_use feature for structured output.
        """
        tools = self._build_tool_definitions(available_tools or [])
        messages = self._build_messages(context)

        system = ASSISTANT_SYSTEM_PROMPT.format(
            available_values=context.format_available_values(),
        )

        api_kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": messages,
        }
        if tools:
            api_kwargs["tools"] = tools

        response = self.client.messages.create(**api_kwargs)

        return self._parse_response(response)

    def _build_tool_definitions(self, available_tools: list[tuple[str, str]]) -> list[dict]:
        """Convert API endpoints into Claude tool definitions."""
        tools = []
        for tool_name, api_name in available_tools:
            endpoint = self.registry.get_endpoint(tool_name, api_name)
            if not endpoint:
                continue

            properties = {}
            required = []

            for param in endpoint.required_parameters:
                properties[param.name] = {
                    "type": self._map_type(param.type),
                    "description": param.description or param.name,
                }
                required.append(param.name)

            for param in endpoint.optional_parameters:
                properties[param.name] = {
                    "type": self._map_type(param.type),
                    "description": param.description or param.name,
                }

            input_schema = {
                "type": "object",
                "properties": properties,
                "required": required,
            }

            tool_id = f"{self._sanitize_name(tool_name)}__{self._sanitize_name(api_name)}"

            tools.append({
                "name": tool_id,
                "description": (
                    f"[{tool_name}/{api_name}] "
                    f"{endpoint.description or 'No description available.'}"
                ),
                "input_schema": input_schema,
            })

        return tools

    def _build_messages(self, context: ConversationContext) -> list[dict]:
        """Convert conversation context into Claude message format.

        Claude requires:
        1. Alternating user/assistant messages
        2. Each tool_use must have a tool_result immediately after (in next user msg)
        3. tool_use IDs must be unique
        """
        # First pass: pair up assistant tool_calls with their tool results
        raw = list(context.messages)
        result_messages = []
        tool_use_counter = 0

        i = 0
        while i < len(raw):
            msg = raw[i]

            if msg.role == "user":
                self._append_user(result_messages, msg.content or "")

            elif msg.role == "assistant":
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})

                tool_use_ids = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_use_counter += 1
                        tool_id = f"{self._sanitize_name(tc.tool_name)}__{self._sanitize_name(tc.api_name)}"
                        use_id = f"toolu_{tool_use_counter:04d}"
                        content_blocks.append({
                            "type": "tool_use",
                            "id": use_id,
                            "name": tool_id,
                            "input": tc.arguments,
                        })
                        tool_use_ids.append(use_id)

                if not content_blocks:
                    content_blocks.append({"type": "text", "text": "I'll help you with that."})

                result_messages.append({"role": "assistant", "content": content_blocks})

                # If this assistant message had tool calls, we must provide tool_results next
                if tool_use_ids:
                    tool_results = []
                    # Look ahead for the tool message
                    if i + 1 < len(raw) and raw[i + 1].role == "tool" and raw[i + 1].tool_outputs:
                        tool_msg = raw[i + 1]
                        for idx, to in enumerate(tool_msg.tool_outputs):
                            uid = tool_use_ids[idx] if idx < len(tool_use_ids) else tool_use_ids[-1]
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": uid,
                                "content": json.dumps(to.response),
                            })
                        i += 1  # Skip the tool message since we consumed it
                    else:
                        # No tool results found — inject placeholders
                        for uid in tool_use_ids:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": uid,
                                "content": '{"status": "completed"}',
                            })

                    result_messages.append({"role": "user", "content": tool_results})

            elif msg.role == "tool":
                # Orphaned tool message (not consumed above) — skip it
                pass

            i += 1

        # Ensure starts with user
        if not result_messages or result_messages[0]["role"] != "user":
            result_messages.insert(0, {"role": "user", "content": "Hello, I need help with something."})

        # Merge consecutive same-role messages
        cleaned = []
        for msg in result_messages:
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                prev = cleaned[-1]
                self._merge_content(prev, msg)
            else:
                cleaned.append(msg)

        return cleaned

    def _append_user(self, messages: list[dict], content: str) -> None:
        """Append a user text message, merging with previous user msg if needed."""
        if messages and messages[-1]["role"] == "user":
            prev = messages[-1]
            if isinstance(prev["content"], str):
                prev["content"] = prev["content"] + "\n" + content
            else:
                prev["content"].append({"type": "text", "text": content})
        else:
            messages.append({"role": "user", "content": content})

    @staticmethod
    def _merge_content(prev: dict, msg: dict) -> None:
        """Merge msg content into prev (same role)."""
        if isinstance(prev["content"], str) and isinstance(msg["content"], str):
            prev["content"] = prev["content"] + "\n" + msg["content"]
        else:
            if isinstance(prev["content"], str):
                prev["content"] = [{"type": "text", "text": prev["content"]}]
            if isinstance(msg["content"], str):
                prev["content"].append({"type": "text", "text": msg["content"]})
            elif isinstance(msg["content"], list):
                prev["content"].extend(msg["content"])

    def _parse_response(self, response) -> Message:
        """Parse Claude's response into a Message model."""
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                parts = block.name.split("__", 1)
                if len(parts) == 2:
                    tool_name, api_name = parts
                else:
                    tool_name, api_name = block.name, block.name

                tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        api_name=api_name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        content = "\n".join(text_parts) if text_parts else None

        return Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )

    @staticmethod
    def _map_type(param_type: str) -> str:
        mapping = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "object": "object",
            "array": "array",
        }
        return mapping.get(param_type.lower(), "string")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize a name for use as a Claude tool identifier."""
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if sanitized and not sanitized[0].isalpha():
            sanitized = "t_" + sanitized
        return sanitized[:60]
