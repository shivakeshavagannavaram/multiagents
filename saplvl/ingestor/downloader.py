"""Download ToolBench data from HuggingFace."""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

TOOLBENCH_DATASET = "tuandunghcmut/toolbench-v1"


class ToolBenchDownloader:
    """Downloads and caches ToolBench tool definitions from HuggingFace.

    The dataset has rows with {id, conversations} where the system message
    contains tool definitions in this format:

        You have access of the following tools:
        1.tool_name: tool description
        ...
        Specifically, you have access to the following APIs:
        [{name, description, parameters: {type, properties, required, optional}}]
    """

    def __init__(self, cache_dir: str = ".cache/toolbench"):
        self.cache_dir = Path(cache_dir)
        self.tools_file = self.cache_dir / "tools.json"

    def is_cached(self) -> bool:
        # Check that the cache actually has tools (not an empty [])
        if not self.tools_file.exists():
            return False
        try:
            with open(self.tools_file) as f:
                data = json.load(f)
            return len(data) > 0
        except Exception:
            return False

    def download(self, force: bool = False) -> Path:
        """Download ToolBench data and save parsed tools to cache.

        Returns path to the cached tools JSON file.
        """
        if self.is_cached() and not force:
            logger.info("ToolBench data already cached at %s", self.tools_file)
            return self.tools_file

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading ToolBench dataset from HuggingFace...")

        dataset = load_dataset(TOOLBENCH_DATASET, split="train")
        tools_raw = self._extract_tools_from_conversations(dataset)

        logger.info("Extracted %d tool definitions", len(tools_raw))

        with open(self.tools_file, "w") as f:
            json.dump(tools_raw, f, indent=2)

        return self.tools_file

    def _extract_tools_from_conversations(self, dataset) -> list[dict]:
        """Extract tool definitions from system messages in conversations.

        Each row has a system message containing:
        1. A tool list section: "You have access of the following tools:\n1.tool_name: description\n..."
        2. An API list section: "following APIs: [{name, description, parameters}]"

        We parse both to build complete tool definitions.
        """
        tools_map: dict[str, dict] = {}
        errors = 0

        for i, row in enumerate(dataset):
            if i % 10000 == 0:
                logger.info("Processing row %d/%d, found %d tools so far", i, len(dataset), len(tools_map))

            try:
                conversations = row.get("conversations", {})
                values = conversations.get("value", [])
                froms = conversations.get("from", [])

                # Find the system message
                system_msg = None
                for role, value in zip(froms, values):
                    if role == "system":
                        system_msg = value
                        break

                if not system_msg:
                    continue

                # Extract tool descriptions from the tool list section
                tool_descs = self._parse_tool_list(system_msg)

                # Extract API definitions from the API list section
                apis = self._parse_api_list(system_msg)

                if not apis:
                    continue

                # Group APIs by tool name and merge with tool descriptions
                for api in apis:
                    api_name = api.get("name", "")
                    if not api_name:
                        continue

                    # Extract tool name from API name pattern: "api_name_for_tool_name"
                    tool_name = self._extract_tool_name(api_name, api.get("description", ""))

                    if not tool_name:
                        continue

                    if tool_name not in tools_map:
                        tools_map[tool_name] = {
                            "tool_name": tool_name,
                            "tool_description": tool_descs.get(tool_name, ""),
                            "category": "",
                            "standardized_name": tool_name,
                            "api_list": [],
                        }

                    # Convert parameters to our format
                    params = api.get("parameters", {})
                    properties = params.get("properties", {})
                    required_names = set(params.get("required", []))
                    optional_names = set(params.get("optional", []))

                    required_params = []
                    optional_params = []

                    for pname, pinfo in properties.items():
                        param = {
                            "name": pname,
                            "type": pinfo.get("type", "string"),
                            "description": pinfo.get("description", ""),
                            "default": pinfo.get("default"),
                            "example_value": pinfo.get("example_value"),
                        }
                        if pname in required_names:
                            required_params.append(param)
                        else:
                            optional_params.append(param)

                    # Clean the API name (remove _for_tool_name suffix)
                    clean_api_name = self._clean_api_name(api_name, tool_name)

                    endpoint = {
                        "name": clean_api_name,
                        "url": "",
                        "description": self._clean_api_description(api.get("description", "")),
                        "method": "GET",
                        "required_parameters": required_params,
                        "optional_parameters": optional_params,
                    }

                    # Avoid duplicate endpoints
                    existing_names = {e["name"] for e in tools_map[tool_name]["api_list"]}
                    if clean_api_name not in existing_names:
                        tools_map[tool_name]["api_list"].append(endpoint)

            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.debug("Error parsing row %d: %s", i, e)

        if errors > 0:
            logger.info("Encountered %d parse errors (non-fatal)", errors)

        # Filter out tools with no endpoints
        result = [t for t in tools_map.values() if t["api_list"]]
        return result

    @staticmethod
    def _parse_tool_list(system_msg: str) -> dict[str, str]:
        """Parse the tool list section to get tool name -> description mapping.

        Format: "1.tool_name: description\n2.tool_name2: description2\n..."
        """
        tool_descs = {}
        pattern = r'\d+\.(\w+):\s*(.+?)(?=\n\d+\.|\nSpecifically|\n\n|$)'
        matches = re.finditer(pattern, system_msg, re.DOTALL)
        for match in matches:
            name = match.group(1).strip()
            desc = match.group(2).strip()
            tool_descs[name] = desc
        return tool_descs

    @staticmethod
    def _parse_api_list(system_msg: str) -> list[dict]:
        """Extract the API definitions list from the system message.

        The APIs are in a Python-style list of dicts after "following APIs:".
        Uses ast.literal_eval since the data uses single quotes.
        """
        # Find the API list
        start = system_msg.find("[{")
        if start < 0:
            return []

        end = system_msg.rfind("}]")
        if end < 0:
            return []

        api_str = system_msg[start:end + 2]

        try:
            return ast.literal_eval(api_str)
        except (ValueError, SyntaxError):
            # Try fixing common issues
            try:
                # Sometimes there are unescaped quotes in descriptions
                api_str = api_str.replace("\\'", "'")
                return ast.literal_eval(api_str)
            except (ValueError, SyntaxError):
                return []

    @staticmethod
    def _extract_tool_name(api_name: str, description: str) -> str:
        """Extract the tool name from an API name.

        Pattern: "function_name_for_tool_name"
        Or from description: 'subfunction for tool "tool_name"'
        """
        # Try description pattern first (more reliable)
        desc_match = re.search(r'tool\s+"([^"]+)"', description)
        if desc_match:
            return desc_match.group(1)

        # Fall back to API name pattern
        name_match = re.search(r'_for_(\w+)$', api_name)
        if name_match:
            return name_match.group(1)

        return ""

    @staticmethod
    def _clean_api_name(api_name: str, tool_name: str) -> str:
        """Remove the _for_tool_name suffix from API names."""
        suffix = f"_for_{tool_name}"
        if api_name.endswith(suffix):
            return api_name[:-len(suffix)]
        return api_name

    @staticmethod
    def _clean_api_description(description: str) -> str:
        """Clean up API descriptions by removing boilerplate."""
        # Remove "This is the subfunction for tool X, you can use this tool."
        description = re.sub(
            r'This is the subfunction for tool "[^"]+",?\s*you can use this tool\.?\s*',
            '', description
        )
        # Remove "The description of this function is: "
        description = re.sub(
            r'The description of this function is:\s*"?',
            '', description
        )
        # Clean trailing quote
        description = description.rstrip('"').strip()
        return description
