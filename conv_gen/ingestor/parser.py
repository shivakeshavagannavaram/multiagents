"""Parse raw ToolBench JSON into clean data models."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from conv_gen.models import APIEndpoint, Tool, ToolParameter

logger = logging.getLogger(__name__)


class ToolBenchParser:
    """Defensive parser for raw ToolBench JSON into clean Tool models."""

    TYPE_MAP = {
        "string": "string",
        "str": "string",
        "text": "string",
        "int": "integer",
        "integer": "integer",
        "number": "number",
        "float": "number",
        "double": "number",
        "boolean": "boolean",
        "bool": "boolean",
        "object": "object",
        "dict": "object",
        "array": "array",
        "list": "array",
    }

    VALID_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}

    def parse_file(self, path: Path) -> list[Tool]:
        with open(path) as f:
            raw_tools = json.load(f)
        return self.parse_tools(raw_tools)

    def parse_tools(self, raw_tools: list[dict]) -> list[Tool]:
        tools = []
        for raw in raw_tools:
            try:
                tool = self._parse_tool(raw)
                if tool.api_list:
                    tools.append(tool)
            except Exception as e:
                logger.warning("Failed to parse tool '%s': %s", raw.get("tool_name", "?"), e)
        logger.info("Parsed %d tools from %d raw entries", len(tools), len(raw_tools))
        return tools

    def _parse_tool(self, raw: dict) -> Tool:
        tool_name = raw.get("tool_name", "").strip()
        if not tool_name:
            raise ValueError("Tool has no name")

        api_list = []
        for raw_api in raw.get("api_list", []):
            try:
                endpoint = self._parse_endpoint(raw_api)
                api_list.append(endpoint)
            except Exception as e:
                logger.debug("Skipping endpoint in tool '%s': %s", tool_name, e)

        tool_description = self._clean_description(raw.get("tool_description", ""))
        if not tool_description and api_list:
            tool_description = self._synthesize_tool_description(tool_name, api_list)

        return Tool(
            tool_name=tool_name,
            standardized_name=raw.get("standardized_name", self._standardize_name(tool_name)),
            tool_description=tool_description,
            category=raw.get("category", "").strip(),
            api_list=api_list,
        )

    def _parse_endpoint(self, raw: dict) -> APIEndpoint:
        name = raw.get("name", "").strip()
        if not name:
            raise ValueError("Endpoint has no name")

        response_schema = self._parse_response_schema(raw.get("schema"))

        method = self._normalize_method(raw.get("method", "GET"))
        required_params = self._parse_parameters(raw.get("required_parameters", []))
        optional_params = self._parse_parameters(raw.get("optional_parameters", []))

        if not response_schema:
            response_schema = self._infer_response_schema(
                name, method, required_params + optional_params
            )

        return APIEndpoint(
            name=name,
            url=raw.get("url", ""),
            description=self._clean_description(raw.get("description", "")),
            method=method,
            required_parameters=required_params,
            optional_parameters=optional_params,
            response_schema=response_schema,
        )

    def _parse_parameters(self, raw_params) -> list[ToolParameter]:
        """Accept list of dicts, JSON strings, or mixed formats."""
        if isinstance(raw_params, str):
            try:
                raw_params = json.loads(raw_params)
            except json.JSONDecodeError:
                return []

        if not isinstance(raw_params, list):
            return []

        params = []
        for raw_p in raw_params:
            if isinstance(raw_p, str):
                try:
                    raw_p = json.loads(raw_p)
                except json.JSONDecodeError:
                    continue

            if not isinstance(raw_p, dict):
                continue

            name = raw_p.get("name", "").strip()
            if not name:
                continue

            param_type = self._normalize_type(raw_p.get("type"))

            description = self._clean_description(str(raw_p.get("description", "")))
            if not description:
                description = self._humanize_param_name(name)

            params.append(
                ToolParameter(
                    name=name,
                    type=param_type,
                    description=description,
                    default=raw_p.get("default"),
                    example_value=raw_p.get("example_value"),
                )
            )
        return params

    def _normalize_type(self, raw_type: Any) -> str:
        """Map messy type strings ('DATE (YYYY-MM-DD)', 'ENUM', None, ...) to canonical types."""
        if raw_type is None:
            return "string"

        type_str = str(raw_type).strip().lower()
        type_str = re.sub(r"\s*\(.*\)", "", type_str).strip()

        if type_str in self.TYPE_MAP:
            return self.TYPE_MAP[type_str]

        if "date" in type_str or "time" in type_str:
            return "string"
        if "enum" in type_str:
            return "string"
        if "comma" in type_str:
            return "string"

        return "string"

    def _normalize_method(self, method: str) -> str:
        if not method:
            return "GET"
        method = method.upper().strip()
        return method if method in self.VALID_METHODS else "GET"

    def _parse_response_schema(self, raw_schema: Any) -> dict[str, Any] | None:
        """Return the schema as-is if present (full JSON Schema or simplified type hints)."""
        if not raw_schema:
            return None

        if isinstance(raw_schema, str):
            if not raw_schema.strip():
                return None
            try:
                raw_schema = json.loads(raw_schema)
            except json.JSONDecodeError:
                return None

        if isinstance(raw_schema, dict) and raw_schema:
            return raw_schema

        return None

    @staticmethod
    def _infer_response_schema(
        api_name: str, method: str, params: list[ToolParameter]
    ) -> dict[str, Any]:
        """Synthesize a plausible response schema from API name, method, and input params."""
        properties: dict[str, Any] = {}
        api_lower = api_name.lower()

        is_search = any(w in api_lower for w in ("search", "find", "list", "query", "get_all"))
        is_create = any(w in api_lower for w in ("book", "create", "register", "order", "add"))
        is_delete = any(w in api_lower for w in ("delete", "remove", "cancel"))
        is_mutation = method in ("POST", "PUT", "PATCH", "DELETE")

        entity_props: dict[str, Any] = {}
        for param in params:
            entity_props[param.name] = {"type": param.type}

        for param in params:
            name = param.name.lower()
            if name.endswith("_id") or name == "id":
                entity_type = name.replace("_id", "") if name != "id" else "item"
                entity_props[f"{entity_type}_name"] = {"type": "string"}

        if "id" not in entity_props and not any(k.endswith("_id") for k in entity_props):
            entity_props["id"] = {"type": "string"}

        if is_search:
            properties["results"] = {
                "type": "array",
                "items": {"type": "object", "properties": entity_props},
            }
            properties["total_count"] = {"type": "integer"}
            properties["status"] = {"type": "string"}
        elif is_create:
            parts = api_lower.replace("_", " ").split()
            entity = parts[-1] if len(parts) > 1 else "item"
            properties[f"{entity}_id"] = {"type": "string"}
            properties["status"] = {"type": "string"}
            properties["created_at"] = {"type": "string"}
            properties.update(entity_props)
        elif is_delete:
            properties["status"] = {"type": "string"}
            properties["message"] = {"type": "string"}
        elif is_mutation:
            properties["status"] = {"type": "string"}
            properties.update(entity_props)
        else:
            properties.update(entity_props)
            properties["status"] = {"type": "string"}

        return {"type": "object", "properties": properties}

    @staticmethod
    def _clean_description(desc: str) -> str:
        """Strip HTML, ToolBench boilerplate, and whitespace."""
        if not desc:
            return ""
        desc = re.sub(r"<[^>]+>", "", desc)
        desc = re.sub(
            r'This is the subfunction for tool "[^"]+",?\s*you can use this tool\.?\s*',
            '', desc
        )
        desc = re.sub(
            r'The description of this function is:\s*"?',
            '', desc
        )
        desc = desc.rstrip('"').strip()
        desc = re.sub(r"\s+", " ", desc).strip()
        if desc in ("-", "N/A", "n/a", "None", "none", "null", "..."):
            return ""
        return desc

    @staticmethod
    def _standardize_name(name: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
        return name.strip("_")

    @staticmethod
    def _humanize_param_name(name: str) -> str:
        return name.replace("_", " ").replace("-", " ").capitalize()

    @staticmethod
    def _synthesize_tool_description(tool_name: str, api_list: list[APIEndpoint]) -> str:
        """Build a description from the tool's API list when none is supplied."""
        api_descs = [ep.description for ep in api_list if ep.description]
        if not api_descs:
            return f"Tool providing {len(api_list)} API endpoint(s)"

        summary_parts = api_descs[:3]
        summary = "; ".join(summary_parts)
        if len(api_descs) > 3:
            summary += f" (and {len(api_descs) - 3} more)"
        return summary
