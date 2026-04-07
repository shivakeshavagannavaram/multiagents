"""Parse raw ToolBench JSON into clean data models."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from saplvl.models import APIEndpoint, Tool, ToolParameter

logger = logging.getLogger(__name__)


class ToolBenchParser:
    """Parses raw ToolBench JSON into uniform Tool models.

    Handles common inconsistencies:
    - Missing parameter types (defaults to 'string')
    - Missing descriptions
    - HTML in descriptions
    - Inconsistent method names
    - Parameters as JSON strings instead of dicts
    """

    VALID_TYPES = {"string", "integer", "number", "boolean", "object", "array"}
    VALID_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}

    def parse_file(self, path: Path) -> list[Tool]:
        """Parse a JSON file containing tool definitions."""
        with open(path) as f:
            raw_tools = json.load(f)
        return self.parse_tools(raw_tools)

    def parse_tools(self, raw_tools: list[dict]) -> list[Tool]:
        """Parse a list of raw tool dicts into Tool models."""
        tools = []
        for raw in raw_tools:
            try:
                tool = self._parse_tool(raw)
                if tool.api_list:  # Skip tools with no endpoints
                    tools.append(tool)
            except Exception as e:
                logger.warning("Failed to parse tool '%s': %s", raw.get("tool_name", "?"), e)
        logger.info("Parsed %d tools from %d raw entries", len(tools), len(raw_tools))
        return tools

    def _parse_tool(self, raw: dict) -> Tool:
        """Parse a single tool definition."""
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

        return Tool(
            tool_name=tool_name,
            standardized_name=raw.get("standardized_name", self._standardize_name(tool_name)),
            tool_description=self._clean_description(raw.get("tool_description", "")),
            category=raw.get("category", "").strip(),
            api_list=api_list,
        )

    def _parse_endpoint(self, raw: dict) -> APIEndpoint:
        """Parse a single API endpoint."""
        name = raw.get("name", "").strip()
        if not name:
            raise ValueError("Endpoint has no name")

        return APIEndpoint(
            name=name,
            url=raw.get("url", ""),
            description=self._clean_description(raw.get("description", "")),
            method=self._normalize_method(raw.get("method", "GET")),
            required_parameters=self._parse_parameters(raw.get("required_parameters", [])),
            optional_parameters=self._parse_parameters(raw.get("optional_parameters", [])),
        )

    def _parse_parameters(self, raw_params) -> list[ToolParameter]:
        """Parse parameters from various formats."""
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

            param_type = str(raw_p.get("type", "string")).lower().strip()
            if param_type not in self.VALID_TYPES:
                param_type = "string"

            params.append(
                ToolParameter(
                    name=name,
                    type=param_type,
                    description=self._clean_description(str(raw_p.get("description", ""))),
                    default=raw_p.get("default"),
                    example_value=raw_p.get("example_value"),
                )
            )
        return params

    def _normalize_method(self, method: str) -> str:
        """Normalize HTTP method to uppercase."""
        method = method.upper().strip()
        return method if method in self.VALID_METHODS else "GET"

    @staticmethod
    def _clean_description(desc: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        if not desc:
            return ""
        desc = re.sub(r"<[^>]+>", "", desc)
        desc = re.sub(r"\s+", " ", desc).strip()
        return desc

    @staticmethod
    def _standardize_name(name: str) -> str:
        """Convert a tool name to a standardized identifier."""
        name = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
        return name.strip("_")
