"""Offline tool execution with mock response generation."""

from __future__ import annotations

import json
import logging
import random
import string
import uuid
from typing import Any

import openai

from conv_gen.ingestor.registry import ToolRegistry
from conv_gen.models import ToolCall, ToolOutput

logger = logging.getLogger(__name__)


class SessionState:
    """Per-conversation value store so step N+1 can reuse IDs from step N."""

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._history: list[dict[str, Any]] = []

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        norm = self._normalize(key)
        if norm != key:
            self._store[norm] = value

    def get(self, key: str, default: Any = None) -> Any:
        val = self._store.get(key)
        if val is not None:
            return val
        return self._store.get(self._normalize(key), default)

    @staticmethod
    def _normalize(name: str) -> str:
        """camelCase → snake_case, lowercase."""
        import re
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
        return s.lower().strip('_')

    def get_all(self) -> dict[str, Any]:
        return dict(self._store)

    def add_response(self, response: dict[str, Any]) -> None:
        self._history.append(response)
        self._extract_values(response)

    def _extract_values(self, data: dict[str, Any], prefix: str = "") -> None:
        """Flatten every non-empty scalar out of the response dict."""
        if not isinstance(data, dict):
            return
        for key, value in data.items():
            if isinstance(value, dict):
                self._extract_values(value, key)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    self._extract_values(value[0], key)
                    self.set(f"{key}_count", len(value))
                elif value:
                    self.set(key, value[0])
            elif value is not None and value != "":
                self.set(key, value)

    def format_for_prompt(self) -> str:
        """Render session values as a prompt-ready block."""
        if not self._store:
            return "No previous tool outputs available."

        lines = ["Available values from previous tool calls:"]
        for key, value in self._store.items():
            lines.append(f"  - {key}: {value}")
        return "\n".join(lines)


class ToolSimulator:
    """Offline mock tool execution. LLM-backed by default, schema-fallback otherwise."""

    def __init__(
        self,
        registry: ToolRegistry,
        openai_client: openai.OpenAI | None = None,
        use_llm_mocks: bool = True,
        model: str = "gpt-4.1-nano",
        rng: random.Random | None = None,
    ):
        self.registry = registry
        self.openai_client = openai_client
        self.use_llm_mocks = use_llm_mocks and openai_client is not None
        self.model = model
        self.rng = rng or random.Random()

    def execute(
        self,
        tool_call: ToolCall,
        session: SessionState,
        scenario: str = "",
        chain_context: str = "",
    ) -> ToolOutput:
        """Generate a mock response for one tool call and record it in session state."""
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

        if self.use_llm_mocks:
            try:
                response = self._llm_mock(tool_call, session, scenario, chain_context)
            except Exception as e:
                logger.warning("LLM mock failed for %s/%s: %s, falling back to schema", tool_call.tool_name, tool_call.api_name, e)
                response = self._schema_fallback(tool_call, session)
        else:
            response = self._schema_fallback(tool_call, session)

        if not isinstance(response, dict):
            response = {"result": response, "status": "success"}

        session.add_response(response)

        return ToolOutput(
            tool_call=tool_call,
            response=response,
            success=True,
        )

    def _llm_mock(self, tool_call: ToolCall, session: SessionState, scenario: str = "", chain_context: str = "") -> dict[str, Any]:
        """Nano-generated mock response, grounded on session values and the endpoint schema."""
        endpoint = self.registry.get_endpoint(tool_call.tool_name, tool_call.api_name)
        endpoint_desc = endpoint.description if endpoint else ""

        params_desc = ""
        if endpoint:
            params_desc = "\n".join(
                f"  - {p.name} ({p.type}): {p.description}"
                for p in endpoint.all_parameters
            )

        schema_hint = ""
        if endpoint and endpoint.response_schema:
            schema_str = json.dumps(endpoint.response_schema, indent=2)
            if len(schema_str) > 1500:
                schema_str = schema_str[:1500] + "\n  ... (truncated)"
            schema_hint = f"\nExpected response schema:\n{schema_str}"

        session_context = session.format_for_prompt()

        scenario_hint = ""
        if scenario:
            scenario_hint = f"\nConversation context: {scenario}\n"

        chain_hint = ""
        if chain_context:
            chain_hint = f"\nWorkflow context: {chain_context}\n"

        prompt = f"""Generate a realistic mock JSON response for this API call.
{scenario_hint}{chain_hint}
API: {tool_call.tool_name}/{tool_call.api_name}
Description: {endpoint_desc}
Method: {endpoint.method if endpoint else 'GET'}
Parameters:
{params_desc}

Arguments provided: {json.dumps(tool_call.arguments)}
{schema_hint}

{session_context}

Requirements:
- Return a realistic JSON response with plausible real-world values (real city names, realistic prices, proper IDs)
- Match the response schema structure if provided above
- Use correct data types (numbers for prices/temperatures/coordinates, not strings)
- If the arguments reference values from previous calls, reuse those EXACT values
- Generate unique IDs for new entities (e.g. "htl_a8f2", "bk_3391")
- If the response contains a list, return 2-3 items with DISTINCT values in each
- Do NOT echo input parameters back in the response — especially API keys, tokens, or auth credentials. The response is what the API returns, not what was sent
- Return ONLY valid JSON, no explanation"""

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=800,
        )

        return json.loads(response.choices[0].message.content)

    def _schema_fallback(self, tool_call: ToolCall, session: SessionState) -> dict[str, Any]:
        """Mock response from schema → from params → from API-name heuristics."""
        endpoint = self.registry.get_endpoint(tool_call.tool_name, tool_call.api_name)

        if endpoint and endpoint.response_schema:
            return self._generate_from_schema(
                endpoint.response_schema, tool_call, session
            )

        if endpoint and endpoint.all_parameters:
            return self._infer_from_params(endpoint, tool_call, session)

        response: dict[str, Any] = {}
        api_name = tool_call.api_name.lower()

        if any(word in api_name for word in ("search", "find", "list", "get_all")):
            response["results"] = [{"id": f"item_{uuid.uuid4().hex[:6]}"}]
            response["total_count"] = self.rng.randint(1, 50)
            response["status"] = "success"

        elif any(word in api_name for word in ("book", "create", "register", "order", "add")):
            entity = api_name.split("_")[-1] if "_" in api_name else "item"
            response[f"{entity}_id"] = f"{entity[:3]}_{uuid.uuid4().hex[:4]}"
            response["status"] = "confirmed"

        elif any(word in api_name for word in ("delete", "remove", "cancel")):
            response["status"] = "deleted"
            response["message"] = "Successfully deleted"

        else:
            response["status"] = "success"
            response["message"] = "Operation completed"

        return response

    def _infer_from_params(
        self,
        endpoint: Any,
        tool_call: ToolCall,
        session: SessionState,
    ) -> dict[str, Any]:
        """Echo input params + synthesize plausible output fields from the API name."""
        response: dict[str, Any] = {}
        api_name = endpoint.name.lower()
        is_search = any(w in api_name for w in ("search", "find", "list", "get_all", "query"))
        is_create = any(w in api_name for w in ("book", "create", "register", "order", "add", "post"))
        is_get = any(w in api_name for w in ("get", "fetch", "retrieve", "detail", "info"))

        entity: dict[str, Any] = {}

        for param in endpoint.all_parameters:
            name = param.name
            if name in tool_call.arguments:
                entity[name] = tool_call.arguments[name]
            elif session.get(name) is not None:
                entity[name] = session.get(name)
            else:
                entity[name] = self._generate_param_value(param)

        for param in endpoint.all_parameters:
            name = param.name.lower()
            if name.endswith("_id") or name == "id":
                entity_type = name.replace("_id", "") if name != "id" else "item"
                entity[f"{entity_type}_name"] = f"Sample {entity_type.replace('_', ' ').title()} {self.rng.randint(1, 99)}"
            elif name in ("city", "country", "location"):
                entity["latitude"] = round(self.rng.uniform(-90, 90), 4)
                entity["longitude"] = round(self.rng.uniform(-180, 180), 4)

        if not any(k.endswith("_id") or k == "id" for k in entity):
            entity["id"] = f"res_{uuid.uuid4().hex[:6]}"

        if is_search:
            response["results"] = [entity]
            response["total_count"] = self.rng.randint(1, 25)
            response["status"] = "success"
        elif is_create:
            response.update(entity)
            response["status"] = "confirmed"
            day = self.rng.randint(1, 28)
            month = self.rng.randint(1, 12)
            hour = self.rng.randint(8, 20)
            minute = self.rng.randint(0, 59)
            response["created_at"] = f"2026-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z"
            entity_type = api_name.split("_")[-1] if "_" in api_name else "item"
            response[f"{entity_type}_id"] = f"{entity_type[:3]}_{uuid.uuid4().hex[:4]}"
        elif is_get:
            response.update(entity)
        else:
            response.update(entity)
            response["status"] = "success"

        return response

    def _generate_param_value(self, param) -> Any:
        if param.example_value is not None:
            return param.example_value
        if param.default is not None:
            return param.default
        return self._mock_scalar(param.name, param.type)

    def _generate_from_schema(
        self,
        schema: dict[str, Any],
        tool_call: ToolCall,
        session: SessionState,
    ) -> dict[str, Any]:
        return self._mock_schema_value(schema, tool_call, session)

    def _mock_schema_value(
        self,
        schema: Any,
        tool_call: ToolCall,
        session: SessionState,
        depth: int = 0,
        field_name: str | None = None,
    ) -> Any:
        """Mock a value for a schema field: args → session → endpoint defaults → generated."""
        if depth > 4 or not isinstance(schema, dict):
            return "sample_value"

        schema_type = schema.get("type", "object")

        if "properties" in schema:
            result = {}
            for key, prop in schema["properties"].items():
                resolved = self._resolve_from_context(key, tool_call, session)
                if resolved is not None:
                    result[key] = resolved
                elif isinstance(prop, dict):
                    result[key] = self._mock_schema_value(
                        prop, tool_call, session, depth + 1, field_name=key
                    )
                else:
                    result[key] = self._mock_scalar(key, str(prop))
            return result

        # Simplified format: {"bio": "str", "name": "str", "fans_num": "str"}
        schema_keywords = {"type", "description", "default", "enum", "format",
                           "items", "required", "properties", "additionalProperties"}
        non_keyword_keys = [k for k in schema if k not in schema_keywords]
        if non_keyword_keys:
            result = {}
            for key, type_hint in schema.items():
                if key in ("type", "description", "default", "enum", "format",
                           "items", "required", "properties"):
                    continue
                resolved = self._resolve_from_context(key, tool_call, session)
                if resolved is not None:
                    result[key] = resolved
                elif isinstance(type_hint, dict):
                    result[key] = self._mock_schema_value(
                        type_hint, tool_call, session, depth + 1, field_name=key
                    )
                elif isinstance(type_hint, list) and type_hint:
                    if isinstance(type_hint[0], dict):
                        count = self.rng.randint(1, 3)
                        result[key] = [
                            self._mock_schema_value(
                                type_hint[0], tool_call, session, depth + 1, field_name=key
                            ) for _ in range(count)
                        ]
                    else:
                        result[key] = type_hint
                elif isinstance(type_hint, str):
                    result[key] = self._mock_scalar(key, type_hint)
                else:
                    result[key] = type_hint
            if result:
                return result

        if schema_type == "array":
            items = schema.get("items", {})
            count = self.rng.randint(1, 3)
            return [self._mock_schema_value(items, tool_call, session, depth + 1, field_name=field_name)
                    for _ in range(count)]
        elif schema_type in ("integer", "int"):
            return self._mock_integer(field_name or "")
        elif schema_type == "number":
            return self._mock_number(field_name or "")
        elif schema_type == "boolean":
            return self.rng.choice([True, False])
        elif schema_type in ("string", "str"):
            if field_name:
                return self._mock_string(field_name.lower(), field_name)
            return f"value_{uuid.uuid4().hex[:4]}"
        else:
            return {"status": "success"}

    def _resolve_from_context(
        self, field_name: str, tool_call: ToolCall, session: SessionState
    ) -> Any | None:
        """Resolve a schema field from args, session, or endpoint defaults (with fuzzy matching)."""
        norm = self._normalize_for_match(field_name)

        if field_name in tool_call.arguments:
            return tool_call.arguments[field_name]

        for arg_name, arg_val in tool_call.arguments.items():
            arg_norm = self._normalize_for_match(arg_name)
            if norm.endswith(arg_norm) or arg_norm.endswith(norm):
                return arg_val
            if len(arg_norm) >= 3 and arg_norm in norm:
                return arg_val
            if len(norm) >= 3 and norm in arg_norm:
                return arg_val

        session_val = session.get(field_name)
        if session_val is not None:
            return session_val
        for skey, sval in session.get_all().items():
            sk_norm = self._normalize_for_match(skey)
            if norm == sk_norm:
                return sval
            if len(norm) >= 4 and (norm.endswith(sk_norm) or sk_norm.endswith(norm)):
                return sval

        endpoint = self.registry.get_endpoint(tool_call.tool_name, tool_call.api_name)
        if endpoint:
            for param in endpoint.all_parameters:
                if param.name == field_name:
                    if param.example_value is not None and param.example_value != "":
                        return param.example_value
                    if param.default is not None and param.default != "":
                        return param.default

        return None

    @staticmethod
    def _normalize_for_match(name: str) -> str:
        """camelCase → snake_case, lowercased."""
        import re
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        return s.lower().strip('_')

    def _mock_scalar(self, field_name: str, type_hint: str) -> Any:
        """Type-aware scalar generator using field-name patterns for ranges."""
        hint = type_hint.lower().strip()
        name = field_name.lower()

        if hint in ("bool", "boolean"):
            return self.rng.choice([True, False])

        if "list" in hint or "array" in hint or hint == "[]":
            return []

        if hint in ("int", "integer"):
            return self._mock_integer(name)

        if hint in ("float", "number"):
            return self._mock_number(name)

        return self._mock_string(name, field_name)

    def _mock_integer(self, name: str) -> int:
        """Integer with a field-name-driven range."""
        if any(w in name for w in ("count", "total", "num", "quantity", "size")):
            return self.rng.randint(1, 50)
        if "year" in name:
            return self.rng.randint(2020, 2026)
        if "age" in name:
            return self.rng.randint(18, 65)
        if any(w in name for w in ("page",)):
            return self.rng.randint(1, 10)
        if any(w in name for w in ("rating", "score", "stars")):
            return self.rng.randint(1, 5)
        if any(w in name for w in ("price", "cost", "amount", "fee")):
            return self.rng.randint(10, 999)
        if any(w in name for w in ("duration", "minutes", "length")):
            return self.rng.randint(1, 180)
        if any(w in name for w in ("percent", "percentage")):
            return self.rng.randint(0, 100)
        if any(w in name for w in ("rank", "position")):
            return self.rng.randint(1, 100)
        if any(w in name for w in ("season", "episode", "round")):
            return self.rng.randint(1, 20)
        if any(w in name for w in ("height", "weight", "width")):
            return self.rng.randint(50, 200)
        if any(w in name for w in ("goal", "point", "assist", "win", "loss")):
            return self.rng.randint(0, 50)
        if "appearance" in name:
            return self.rng.randint(1, 500)
        if "card" in name:
            return self.rng.randint(0, 15)
        if "port" in name:
            return self.rng.randint(80, 9999)
        return self.rng.randint(1, 100)

    def _mock_number(self, name: str) -> float:
        """Float with a field-name-driven range."""
        if any(w in name for w in ("price", "cost", "amount", "fee", "salary")):
            return round(self.rng.uniform(10, 999), 2)
        if any(w in name for w in ("rating", "score", "stars")):
            return round(self.rng.uniform(1.0, 5.0), 1)
        if any(w in name for w in ("percent", "percentage", "ratio")):
            return round(self.rng.uniform(0, 100), 1)
        if "latitude" in name or name == "lat":
            return round(self.rng.uniform(-90, 90), 6)
        if "longitude" in name or name in ("lng", "lon"):
            return round(self.rng.uniform(-180, 180), 6)
        if any(w in name for w in ("temperature", "temp")):
            return round(self.rng.uniform(-10, 40), 1)
        if any(w in name for w in ("distance", "miles", "km")):
            return round(self.rng.uniform(0.1, 500), 1)
        if "high" in name or "max" in name:
            return round(self.rng.uniform(100, 500), 2)
        if "low" in name or "min" in name:
            return round(self.rng.uniform(1, 99), 2)
        if "volume" in name:
            return round(self.rng.uniform(1000, 1e7), 0)
        return round(self.rng.uniform(1, 500), 2)

    def _mock_string(self, name: str, original_name: str) -> str:
        """String pattern-matched from field name. Structurally correct, no hardcoded vocab."""
        if name.endswith("_id") or name == "id":
            prefix = name.replace("_id", "")[:4] if name != "id" else "item"
            return f"{prefix}_{uuid.uuid4().hex[:6]}"

        if "name" in name:
            readable = original_name.replace("_name", "").replace("Name", "").strip()
            if readable:
                return f"Sample {readable.replace('_', ' ').title()} {self.rng.randint(1, 99)}"
            return f"Item {self.rng.randint(1, 999)}"

        if "date" in name or name in ("created_at", "updated_at", "timestamp", "created", "updated"):
            day = self.rng.randint(1, 28)
            month = self.rng.randint(1, 12)
            return f"2026-{month:02d}-{day:02d}"
        if "time" in name:
            return f"{self.rng.randint(0,23):02d}:{self.rng.randint(0,59):02d}:00"

        if "email" in name:
            return f"user{self.rng.randint(1,999)}@example.com"
        if "phone" in name or "mobile" in name:
            return f"+1-{self.rng.randint(200,999)}-{self.rng.randint(100,999)}-{self.rng.randint(1000,9999)}"

        if any(w in name for w in ("url", "link", "image", "avatar", "logo", "photo", "thumbnail")):
            return f"https://api.example.com/{original_name}/{uuid.uuid4().hex[:8]}"

        if "status" in name:
            return self.rng.choice(["active", "confirmed", "completed", "available"])
        if "state" in name:
            return self.rng.choice(["active", "pending", "processing", "done"])

        if "city" in name:
            return f"City {self.rng.randint(1, 50)}"
        if "country" in name:
            return f"Country {self.rng.randint(1, 20)}"
        if "address" in name:
            return f"{self.rng.randint(1,999)} Street {self.rng.randint(1,99)}"
        if "zip" in name or "postal" in name:
            return f"{self.rng.randint(10000, 99999)}"

        if "currency" in name or "code" in name:
            return "".join(self.rng.choices(string.ascii_uppercase, k=3))
        if "symbol" in name:
            return "".join(self.rng.choices(string.ascii_uppercase, k=self.rng.randint(3, 5)))

        if "lang" in name or "locale" in name:
            return self.rng.choice(["en", "fr", "de", "ja", "es"])
        if "timezone" in name or name == "tz":
            return f"UTC{self.rng.choice(['+', '-'])}{self.rng.randint(0,12)}"

        if any(w in name for w in ("description", "desc", "summary", "bio", "about")):
            readable = original_name.replace("_", " ").title()
            return f"Sample {readable} content for this item."
        if any(w in name for w in ("title", "headline", "subject", "label")):
            readable = original_name.replace("_", " ").title()
            return f"Sample {readable} {self.rng.randint(1, 99)}"

        if "type" in name or "kind" in name or "category" in name or "genre" in name:
            return f"type_{self.rng.randint(1, 10)}"

        if "format" in name:
            return self.rng.choice(["json", "xml", "csv"])
        if "method" in name:
            return self.rng.choice(["GET", "POST", "PUT"])

        if "gender" in name or "sex" in name:
            return self.rng.choice(["male", "female", "other"])

        if "color" in name or "colour" in name:
            return f"#{self.rng.randint(0, 0xFFFFFF):06x}"

        if "version" in name:
            return f"{self.rng.randint(1,5)}.{self.rng.randint(0,9)}.{self.rng.randint(0,9)}"

        readable = original_name.replace("_", " ").title()
        return f"Sample {readable}"

