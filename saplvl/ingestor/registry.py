"""In-memory tool registry with indexed lookups."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from saplvl.models import APIEndpoint, Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Indexed collection of parsed tools with fast lookups."""

    def __init__(self, tools: list[Tool]):
        self._tools = tools
        self._by_name: dict[str, Tool] = {}
        self._by_category: dict[str, list[Tool]] = {}
        self._endpoints: dict[tuple[str, str], APIEndpoint] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        for tool in self._tools:
            self._by_name[tool.tool_name] = tool

            cat = tool.category or "uncategorized"
            if cat not in self._by_category:
                self._by_category[cat] = []
            self._by_category[cat].append(tool)

            for endpoint in tool.api_list:
                self._endpoints[(tool.tool_name, endpoint.name)] = endpoint

    def get_tool(self, name: str) -> Tool | None:
        return self._by_name.get(name)

    def get_tools_by_category(self, category: str) -> list[Tool]:
        return self._by_category.get(category, [])

    def get_endpoint(self, tool_name: str, api_name: str) -> APIEndpoint | None:
        return self._endpoints.get((tool_name, api_name))

    def list_categories(self) -> list[str]:
        return sorted(self._by_category.keys())

    def all_tools(self) -> list[Tool]:
        return list(self._tools)

    def all_endpoint_keys(self) -> list[tuple[str, str]]:
        return list(self._endpoints.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def save(self, path: Path) -> None:
        """Save registry to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [tool.model_dump() for tool in self._tools]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved registry with %d tools to %s", len(self._tools), path)

    @classmethod
    def load(cls, path: Path) -> ToolRegistry:
        """Load registry from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        tools = [Tool.model_validate(d) for d in data]
        logger.info("Loaded registry with %d tools from %s", len(tools), path)
        return cls(tools)

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "num_tools": len(self._tools),
            "num_endpoints": len(self._endpoints),
            "num_categories": len(self._by_category),
            "categories": self.list_categories(),
        }
