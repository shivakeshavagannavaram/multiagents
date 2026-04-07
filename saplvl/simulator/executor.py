"""Offline tool execution with mock response generation."""

from __future__ import annotations

import json
import logging
import random
import string
import uuid
from typing import Any

import openai

from saplvl.ingestor.registry import ToolRegistry
from saplvl.models import ToolCall, ToolOutput

logger = logging.getLogger(__name__)


class SessionState:
    """Tracks generated values across tool calls within a conversation.

    Ensures chaining consistency: IDs from step N are available in step N+1.
    """

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._history: list[dict[str, Any]] = []

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def get_all(self) -> dict[str, Any]:
        return dict(self._store)

    def add_response(self, response: dict[str, Any]) -> None:
        """Record a tool response and extract reusable values."""
        self._history.append(response)
        self._extract_values(response)

    def _extract_values(self, data: dict[str, Any], prefix: str = "") -> None:
        """Recursively extract ID-like and entity-like values from responses."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._extract_values(value, full_key)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    # Store first item's fields for downstream use
                    self._extract_values(value[0], full_key)
                    self.set(f"{key}_count", len(value))
                elif value:
                    self.set(key, value[0])
            elif value is not None:
                # Store any scalar value that looks useful
                if any(suffix in key.lower() for suffix in (
                    "id", "name", "code", "token", "number", "key",
                    "email", "url", "status", "type",
                )):
                    self.set(key, value)

    def format_for_prompt(self) -> str:
        """Format available values for inclusion in prompts."""
        if not self._store:
            return "No previous tool outputs available."

        lines = ["Available values from previous tool calls:"]
        for key, value in self._store.items():
            lines.append(f"  - {key}: {value}")
        return "\n".join(lines)


class ToolSimulator:
    """Generates mock tool responses for offline conversation generation.

    Uses OpenAI for realistic mock generation with a schema-derived fallback.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        openai_client: openai.OpenAI | None = None,
        model: str = "gpt-4o-mini",
        rng: random.Random | None = None,
    ):
        self.registry = registry
        self.openai_client = openai_client
        self.model = model
        self.rng = rng or random.Random()

    def execute(self, tool_call: ToolCall, session: SessionState) -> ToolOutput:
        """Generate a mock response for a tool call.

        1. Look up endpoint schema from registry.
        2. Substitute session state values into arguments where applicable.
        3. Generate response via LLM or schema fallback.
        4. Record response in session state.
        """
        endpoint = self.registry.get_endpoint(tool_call.tool_name, tool_call.api_name)

        if endpoint is None:
            logger.warning(
                "Unknown endpoint: %s/%s", tool_call.tool_name, tool_call.api_name
            )
            return ToolOutput(
                tool_call=tool_call,
                response={"error": f"Unknown endpoint: {tool_call.tool_name}/{tool_call.api_name}"},
                success=False,
            )

        # Try LLM mock first, fallback to schema-derived
        try:
            if self.openai_client:
                response = self._llm_mock(tool_call, session)
            else:
                response = self._schema_fallback(tool_call, session)
        except Exception as e:
            logger.warning("Mock generation failed for %s/%s: %s", tool_call.tool_name, tool_call.api_name, e)
            response = self._schema_fallback(tool_call, session)

        session.add_response(response)

        return ToolOutput(
            tool_call=tool_call,
            response=response,
            success=True,
        )

    def _llm_mock(self, tool_call: ToolCall, session: SessionState) -> dict[str, Any]:
        """Generate a realistic mock response using OpenAI."""
        endpoint = self.registry.get_endpoint(tool_call.tool_name, tool_call.api_name)
        endpoint_desc = endpoint.description if endpoint else ""

        params_desc = ""
        if endpoint:
            params_desc = "\n".join(
                f"  - {p.name} ({p.type}): {p.description}"
                for p in endpoint.all_parameters
            )

        session_context = session.format_for_prompt()

        prompt = f"""Generate a realistic mock JSON response for this API call.

API: {tool_call.tool_name}/{tool_call.api_name}
Description: {endpoint_desc}
Method: {endpoint.method if endpoint else 'GET'}
Parameters:
{params_desc}

Arguments provided: {json.dumps(tool_call.arguments)}

{session_context}

Requirements:
- Return a realistic JSON response that this API would return
- Include relevant IDs, names, and data fields
- If the arguments reference values from previous calls, use those exact values
- Keep the response concise but complete
- Return ONLY valid JSON, no explanation"""

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=500,
        )

        return json.loads(response.choices[0].message.content)

    def _schema_fallback(self, tool_call: ToolCall, session: SessionState) -> dict[str, Any]:
        """Generate a mock response based on endpoint schema and heuristics."""
        endpoint = self.registry.get_endpoint(tool_call.tool_name, tool_call.api_name)
        response: dict[str, Any] = {}

        api_name = tool_call.api_name.lower()

        # Generate appropriate response structure based on API patterns
        if any(word in api_name for word in ("search", "find", "list", "get_all")):
            item = self._generate_entity(tool_call, session)
            response["results"] = [item]
            response["total_count"] = self.rng.randint(1, 50)
            response["status"] = "success"

        elif any(word in api_name for word in ("book", "create", "register", "order", "add")):
            entity = api_name.split("_")[-1] if "_" in api_name else "item"
            response[f"{entity}_id"] = f"{entity[:3]}_{uuid.uuid4().hex[:4]}"
            response["status"] = "confirmed"
            response["created_at"] = "2026-04-06T10:00:00Z"
            # Echo back relevant arguments
            for key, value in tool_call.arguments.items():
                response[key] = value

        elif any(word in api_name for word in ("get", "fetch", "retrieve", "detail")):
            response = self._generate_entity(tool_call, session)

        elif any(word in api_name for word in ("update", "modify", "edit")):
            response["status"] = "updated"
            response["updated_at"] = "2026-04-06T10:00:00Z"
            for key, value in tool_call.arguments.items():
                response[key] = value

        elif any(word in api_name for word in ("delete", "remove", "cancel")):
            response["status"] = "deleted"
            response["message"] = "Successfully deleted"

        else:
            response = self._generate_entity(tool_call, session)
            response["status"] = "success"

        return response

    def _generate_entity(self, tool_call: ToolCall, session: SessionState) -> dict[str, Any]:
        """Generate a mock entity based on the tool call context."""
        entity: dict[str, Any] = {}

        api_name = tool_call.api_name.lower()
        entity_type = api_name.split("_")[-1] if "_" in api_name else "item"
        entity["id"] = f"{entity_type[:3]}_{uuid.uuid4().hex[:4]}"

        # Use session values where applicable
        for key, value in tool_call.arguments.items():
            if key.endswith("_id") or key == "id":
                entity[key] = value
            elif key in ("name", "city", "country", "type", "category"):
                entity[key] = value

        # Add common fields based on entity type
        entity["name"] = self._random_name(entity_type)

        if any(word in api_name for word in ("hotel", "property", "accommodation")):
            entity["price"] = self.rng.randint(50, 500)
            entity["rating"] = round(self.rng.uniform(3.0, 5.0), 1)
            entity["currency"] = "USD"

        elif any(word in api_name for word in ("flight", "airline")):
            entity["price"] = self.rng.randint(100, 2000)
            entity["airline"] = self.rng.choice(["United", "Delta", "American", "Southwest"])
            entity["duration"] = f"{self.rng.randint(1, 12)}h {self.rng.randint(0, 59)}m"

        elif any(word in api_name for word in ("restaurant", "food", "dining")):
            entity["cuisine"] = self.rng.choice(["Italian", "Japanese", "Mexican", "French", "Indian"])
            entity["rating"] = round(self.rng.uniform(3.5, 5.0), 1)
            entity["price_range"] = self.rng.choice(["$", "$$", "$$$", "$$$$"])

        elif any(word in api_name for word in ("weather", "forecast")):
            entity["temperature"] = self.rng.randint(-10, 40)
            entity["condition"] = self.rng.choice(["Sunny", "Cloudy", "Rainy", "Snowy", "Windy"])
            entity["humidity"] = self.rng.randint(20, 90)

        return entity

    def _random_name(self, entity_type: str) -> str:
        """Generate a plausible name for an entity."""
        prefixes = {
            "hotel": ["Grand", "Royal", "Sunset", "Ocean View", "Mountain"],
            "flight": ["FL", "UA", "DL", "AA", "SW"],
            "restaurant": ["Café", "Bistro", "Ristorante", "The", "Le"],
            "user": ["John", "Alice", "Carlos", "Yuki", "Priya"],
        }
        prefix_list = prefixes.get(entity_type, ["Item"])
        suffix = "".join(self.rng.choices(string.ascii_uppercase, k=3))
        return f"{self.rng.choice(prefix_list)} {suffix}"
