from __future__ import annotations

import json
import logging
import re

import anthropic

from conv_gen.agents.base import BaseAgent
from conv_gen.ingestor.registry import ToolRegistry
from conv_gen.memory.context import ConversationContext
from conv_gen.models import Message, ToolCall

logger = logging.getLogger(__name__)


# Credential stripping — auth params are hidden from Claude's tool schema
_AUTH_SUFFIX_TOKENS = frozenset({
    "token", "secret", "password", "credential", "credentials",
    "apikey", "api_key", "access_token", "auth", "authorization",
    "client_secret", "client_id", "app_key", "app_secret", "app_id",
    "bearer",
})
_AUTH_SUBSTRINGS = ("token", "secret", "password", "credential")


def _is_auth_param(name: str, description: str = "") -> bool:
    """True if a parameter name is an auth credential that should be stripped."""
    if not name:
        return False
    n = name.lower().replace("-", "_").strip("_")

    if n in _AUTH_SUFFIX_TOKENS:
        return True

    for tok in _AUTH_SUFFIX_TOKENS:
        if n.endswith("_" + tok) or n.startswith(tok + "_"):
            return True

    # "key" alone not matched — collides with "keyword"
    for sub in _AUTH_SUBSTRINGS:
        if sub in n:
            return True

    return False


ASSISTANT_SYSTEM_PROMPT = """You are a helpful AI assistant that uses tools to accomplish tasks.

{available_values}

Rules:
1. ALWAYS use tools when available. Never answer from your own knowledge when a tool can provide the answer
2. Use EXACT values the user provides (coordinates, dates, names). Never approximate
3. When chaining tool calls, use EXACT values from previous tool responses — not invented values
4. After receiving tool results, summarize the 2-3 most important values briefly
5. Never ask for API keys, tokens, passwords, or auth credentials. Use "default" for auth parameters
6. If a required parameter has a default or example value, use it without asking
7. If the user's intent is unclear, ask ONE short clarifying question
8. Do NOT restate the user's request before calling tools
9. NEVER mention tool names, API names, or endpoints to the user
10. Do NOT say "Is there anything else?" when done
11. NEVER ask the user for system-generated IDs (fixture_id, event_id, post_id, session_id, message_id, booking_id, holiday_id, user_id, team_id, channel_id, etc.). Real users don't know these values. Handle them this way:
    a) If a PRIOR tool response contains the ID you need, use it EXACTLY — never reinvent one when it's available from context.
    b) If no prior tool has produced the ID, CREATE one yourself and proceed with the tool call. Do NOT ask the user. Generate a plausible identifier from what the user said — e.g. "man_utd_vs_liv_2024" for a match, "sarah_post_books_2024" for a post, "hol_christmas_2024" for a holiday, "evt_recent_order_delay" for an event. This is expected and correct — treat your own generated ID as valid input for the tool call.
    c) Once a tool has returned a response, any subsequent tool call needing the same entity should use the ID that APPEARS IN THE RESPONSE (not your original fabrication), maintaining consistency across the chain.
    d) Ask the user ONLY for values a real person would naturally know: names, cities, dates, topics, preferences, email addresses, product descriptions, URLs, team matchups, item categories.

"""


class AssistantAgent(BaseAgent):
    """Generates assistant responses using Claude's native tool_use (structured output)."""

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
        plan_status: str = "",
        **kwargs,
    ) -> Message:
        """Generate an assistant response via Claude tool_use (structured output)."""
        tools, name_map, param_maps = self._build_tool_definitions(available_tools or [])
        messages = self._build_messages(context)

        plan_instruction = ""
        if plan_status:
            plan_instruction += f"\n\nCurrent status: {plan_status}"
        if available_tools:
            tool_list_str = ", ".join(f"{t}/{a}" for t, a in available_tools)
            plan_instruction += (
                f"\n\nTools remaining in this workflow: {tool_list_str}"
                "\nCall each remaining tool in order to complete the user's task."
            )

        system = ASSISTANT_SYSTEM_PROMPT.format(
            available_values=context.format_available_values(),
        ) + plan_instruction

        api_kwargs = {
            "model": self.model,
            "max_tokens": 2048,
            "system": system,
            "messages": messages,
        }
        if tools:
            api_kwargs["tools"] = tools

        response = self.client.messages.create(**api_kwargs)

        return self._parse_response(response, name_map, param_maps)

    def _build_tool_definitions(
        self, available_tools: list[tuple[str, str]]
    ) -> tuple[list[dict], dict[str, tuple[str, str]], dict[str, dict[str, str]]]:
        """Convert endpoints to Claude tool defs with sanitized names; returns (tools, name_map, param_maps)."""
        tools = []
        name_map: dict[str, tuple[str, str]] = {}
        param_maps: dict[str, dict[str, str]] = {}

        for tool_name, api_name in available_tools:
            endpoint = self.registry.get_endpoint(tool_name, api_name)
            if not endpoint:
                continue

            properties = {}
            required = []
            param_map: dict[str, str] = {}

            for param in endpoint.required_parameters:
                if _is_auth_param(param.name, param.description or ""):
                    continue
                safe_name = self._sanitize_param_name(param.name)
                # Disambiguate collisions with a numeric suffix
                if safe_name in properties:
                    suffix = 2
                    while f"{safe_name}_{suffix}" in properties:
                        suffix += 1
                    safe_name = f"{safe_name}_{suffix}"
                param_map[safe_name] = param.name
                desc = param.description or param.name
                if param.default:
                    desc += f" (default: {param.default})"
                elif param.example_value:
                    desc += f" (e.g. {param.example_value})"
                properties[safe_name] = {
                    "type": self._map_type(param.type),
                    "description": desc,
                }
                required.append(safe_name)

            for param in endpoint.optional_parameters:
                if _is_auth_param(param.name, param.description or ""):
                    continue
                safe_name = self._sanitize_param_name(param.name)
                if safe_name in properties:
                    suffix = 2
                    while f"{safe_name}_{suffix}" in properties:
                        suffix += 1
                    safe_name = f"{safe_name}_{suffix}"
                param_map[safe_name] = param.name
                desc = param.description or param.name
                if param.default:
                    desc += f" (default: {param.default})"
                properties[safe_name] = {
                    "type": self._map_type(param.type),
                    "description": desc,
                }

            input_schema = {
                "type": "object",
                "properties": properties,
                "required": required,
            }

            tool_id = f"{self._sanitize_name(tool_name)}__{self._sanitize_name(api_name)}"
            name_map[tool_id] = (tool_name, api_name)
            param_maps[tool_id] = param_map

            tools.append({
                "name": tool_id,
                "description": endpoint.description or "No description available.",
                "input_schema": input_schema,
            })

        return tools, name_map, param_maps

    def _build_messages(self, context: ConversationContext) -> list[dict]:
        """Convert ConversationContext to Claude API message format."""
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

                # Skip empty assistant messages (no ghost messages in replay)
                if not content_blocks:
                    i += 1
                    continue

                result_messages.append({"role": "assistant", "content": content_blocks})

                if tool_use_ids:
                    tool_results = []
                    if i + 1 < len(raw) and raw[i + 1].role == "tool" and raw[i + 1].tool_outputs:
                        tool_msg = raw[i + 1]
                        for idx, to in enumerate(tool_msg.tool_outputs):
                            uid = tool_use_ids[idx] if idx < len(tool_use_ids) else tool_use_ids[-1]
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": uid,
                                "content": json.dumps(to.response),
                            })
                        i += 1
                    else:
                        # Missing results → explicit error (API requires a tool_result per tool_use)
                        for uid in tool_use_ids:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": uid,
                                "content": '{"error": "tool result unavailable"}',
                                "is_error": True,
                            })

                    result_messages.append({"role": "user", "content": tool_results})

            elif msg.role == "tool":
                # Orphan tool message — skip
                pass

            i += 1

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

    def _parse_response(
        self,
        response,
        name_map: dict[str, tuple[str, str]] | None = None,
        param_maps: dict[str, dict[str, str]] | None = None,
    ) -> Message:
        """Parse Claude response into a Message, reversing sanitized tool/param names."""
        name_map = name_map or {}
        param_maps = param_maps or {}
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                if block.name in name_map:
                    tool_name, api_name = name_map[block.name]
                else:
                    parts = block.name.split("__", 1)
                    if len(parts) == 2:
                        tool_name, api_name = parts
                    else:
                        tool_name, api_name = block.name, block.name

                raw_args = block.input if isinstance(block.input, dict) else {}

                # Reverse-map sanitized param keys back to registry-canonical names
                param_map = param_maps.get(block.name, {})
                if param_map:
                    arguments = {
                        param_map.get(k, k): v for k, v in raw_args.items()
                    }
                else:
                    arguments = raw_args

                tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        api_name=api_name,
                        arguments=arguments,
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

    @staticmethod
    def _sanitize_param_name(name: str) -> str:
        """Sanitize a parameter name to match Anthropic's `^[a-zA-Z0-9_.-]{1,64}$` pattern."""
        if not name:
            return "param"
        sanitized = re.sub(r"[^a-zA-Z0-9_.\-]", "_", name)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        if not sanitized:
            return "param"
        return sanitized[:64]
