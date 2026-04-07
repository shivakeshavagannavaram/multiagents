"""Tests for the ingestor module: parser and registry."""

import json
import tempfile
from pathlib import Path

import pytest

from saplvl.ingestor.parser import ToolBenchParser
from saplvl.ingestor.registry import ToolRegistry
from saplvl.models import Tool


class TestToolBenchParser:
    def setup_method(self):
        self.parser = ToolBenchParser()

    def test_parse_valid_tool(self):
        raw = [
            {
                "tool_name": "TestTool",
                "standardized_name": "test_tool",
                "tool_description": "A test tool",
                "category": "Testing",
                "api_list": [
                    {
                        "name": "get_data",
                        "url": "https://api.test.com/data",
                        "description": "Get some data",
                        "method": "GET",
                        "required_parameters": [
                            {"name": "query", "type": "string", "description": "Search query"}
                        ],
                        "optional_parameters": [],
                    }
                ],
            }
        ]
        tools = self.parser.parse_tools(raw)
        assert len(tools) == 1
        assert tools[0].tool_name == "TestTool"
        assert len(tools[0].api_list) == 1
        assert tools[0].api_list[0].name == "get_data"
        assert len(tools[0].api_list[0].required_parameters) == 1

    def test_parse_missing_param_type_defaults_to_string(self):
        raw = [
            {
                "tool_name": "T1",
                "category": "Test",
                "api_list": [
                    {
                        "name": "ep1",
                        "required_parameters": [{"name": "p1"}],
                        "optional_parameters": [],
                    }
                ],
            }
        ]
        tools = self.parser.parse_tools(raw)
        assert tools[0].api_list[0].required_parameters[0].type == "string"

    def test_parse_missing_description_defaults_to_empty(self):
        raw = [
            {
                "tool_name": "T2",
                "category": "Test",
                "api_list": [{"name": "ep2", "required_parameters": [], "optional_parameters": []}],
            }
        ]
        tools = self.parser.parse_tools(raw)
        assert tools[0].api_list[0].description == ""

    def test_parse_invalid_method_defaults_to_get(self):
        raw = [
            {
                "tool_name": "T3",
                "category": "Test",
                "api_list": [
                    {"name": "ep3", "method": "INVALID", "required_parameters": [], "optional_parameters": []}
                ],
            }
        ]
        tools = self.parser.parse_tools(raw)
        assert tools[0].api_list[0].method == "GET"

    def test_parse_html_in_description(self):
        raw = [
            {
                "tool_name": "T4",
                "tool_description": "<p>Tool with <b>HTML</b> tags</p>",
                "category": "Test",
                "api_list": [
                    {"name": "ep4", "description": "<br>An endpoint<br/>", "required_parameters": [], "optional_parameters": []}
                ],
            }
        ]
        tools = self.parser.parse_tools(raw)
        assert "<" not in tools[0].tool_description
        assert "<" not in tools[0].api_list[0].description

    def test_parse_skips_tools_with_no_endpoints(self):
        raw = [
            {"tool_name": "Empty", "category": "Test", "api_list": []},
            {
                "tool_name": "NotEmpty",
                "category": "Test",
                "api_list": [{"name": "ep", "required_parameters": [], "optional_parameters": []}],
            },
        ]
        tools = self.parser.parse_tools(raw)
        assert len(tools) == 1
        assert tools[0].tool_name == "NotEmpty"

    def test_parse_skips_tools_with_no_name(self):
        raw = [{"tool_name": "", "category": "Test", "api_list": [{"name": "ep"}]}]
        tools = self.parser.parse_tools(raw)
        assert len(tools) == 0

    def test_parse_parameters_as_json_string(self):
        raw = [
            {
                "tool_name": "T5",
                "category": "Test",
                "api_list": [
                    {
                        "name": "ep5",
                        "required_parameters": json.dumps([{"name": "q", "type": "string"}]),
                        "optional_parameters": "[]",
                    }
                ],
            }
        ]
        tools = self.parser.parse_tools(raw)
        assert len(tools[0].api_list[0].required_parameters) == 1

    def test_parse_file(self, tmp_path):
        data = [
            {
                "tool_name": "FileTool",
                "category": "Test",
                "api_list": [{"name": "ep", "required_parameters": [], "optional_parameters": []}],
            }
        ]
        path = tmp_path / "tools.json"
        path.write_text(json.dumps(data))

        tools = self.parser.parse_file(path)
        assert len(tools) == 1


class TestToolRegistry:
    def test_get_tool(self, sample_registry):
        tool = sample_registry.get_tool("HotelFinder")
        assert tool is not None
        assert tool.tool_name == "HotelFinder"

    def test_get_tool_missing(self, sample_registry):
        assert sample_registry.get_tool("NonExistent") is None

    def test_get_tools_by_category(self, sample_registry):
        travel = sample_registry.get_tools_by_category("Travel")
        assert len(travel) == 2
        names = {t.tool_name for t in travel}
        assert "HotelFinder" in names
        assert "FlightSearch" in names

    def test_get_endpoint(self, sample_registry):
        ep = sample_registry.get_endpoint("HotelFinder", "search_hotels")
        assert ep is not None
        assert ep.name == "search_hotels"

    def test_list_categories(self, sample_registry):
        cats = sample_registry.list_categories()
        assert "Travel" in cats
        assert "Weather" in cats
        assert "Food" in cats
        assert "Finance" in cats

    def test_all_endpoint_keys(self, sample_registry):
        keys = sample_registry.all_endpoint_keys()
        assert ("HotelFinder", "search_hotels") in keys
        assert ("FlightSearch", "book_flight") in keys

    def test_len(self, sample_registry):
        assert len(sample_registry) == 5

    def test_save_and_load(self, sample_registry, tmp_path):
        path = tmp_path / "registry.json"
        sample_registry.save(path)
        loaded = ToolRegistry.load(path)
        assert len(loaded) == len(sample_registry)
        assert loaded.get_tool("HotelFinder") is not None

    def test_summary(self, sample_registry):
        s = sample_registry.summary()
        assert s["num_tools"] == 5
        assert s["num_categories"] == 4
        assert s["num_endpoints"] > 0
