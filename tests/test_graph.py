"""Tests for the Knowledge Graph builder."""

import pytest

from conv_gen.graph.builder import ToolGraphBuilder


class TestToolGraphBuilder:
    def test_endpoint_nodes_added(self, sample_graph, sample_registry):
        """All endpoints should have endpoint nodes in the KG."""
        expected_keys = sample_registry.all_endpoint_keys()
        for tool_name, api_name in expected_keys:
            ep_id = f"endpoint:{tool_name}/{api_name}"
            assert ep_id in sample_graph.nodes(), f"Missing endpoint node: {ep_id}"

    def test_category_nodes(self, sample_graph):
        """Category nodes should exist."""
        assert "category:Travel" in sample_graph.nodes()
        assert "category:Weather" in sample_graph.nodes()
        assert "category:Food" in sample_graph.nodes()
        assert "category:Finance" in sample_graph.nodes()

    def test_tool_nodes(self, sample_graph):
        """Tool nodes should exist."""
        assert "tool:HotelFinder" in sample_graph.nodes()
        assert "tool:FlightSearch" in sample_graph.nodes()

    def test_structural_edges(self, sample_graph):
        """Tool should have has_endpoint edges to its endpoints."""
        assert sample_graph.has_edge("tool:HotelFinder", "endpoint:HotelFinder/search_hotels")
        edge = sample_graph.edges["tool:HotelFinder", "endpoint:HotelFinder/search_hotels"]
        assert edge["edge_type"] == "has_endpoint"

    def test_belongs_to_edges(self, sample_graph):
        """Tools should have belongs_to edges to categories."""
        assert sample_graph.has_edge("tool:HotelFinder", "category:Travel")
        edge = sample_graph.edges["tool:HotelFinder", "category:Travel"]
        assert edge["edge_type"] == "belongs_to"

    def test_same_tool_edges(self, sample_graph):
        """search -> book within same tool should have same_tool edge."""
        assert sample_graph.has_edge(
            "endpoint:HotelFinder/search_hotels", "endpoint:HotelFinder/book_hotel"
        )
        edge = sample_graph.edges[
            "endpoint:HotelFinder/search_hotels", "endpoint:HotelFinder/book_hotel"
        ]
        assert edge["edge_type"] == "same_tool"

    def test_node_attributes(self, sample_graph):
        """Endpoint nodes should have expected attributes."""
        node_data = sample_graph.nodes["endpoint:HotelFinder/search_hotels"]
        assert node_data["node_type"] == "endpoint"
        assert node_data["category"] == "Travel"
        assert node_data["tool_name"] == "HotelFinder"
        assert node_data["schema_quality"] == "complete"

    def test_graph_is_directed(self, sample_graph):
        assert sample_graph.is_directed()

    def test_output_field_extraction(self):
        schema = {
            "type": "object",
            "properties": {
                "hotel_id": {"type": "string"},
                "hotel_name": {"type": "string"},
                "price": {"type": "number"},
            },
        }
        fields = ToolGraphBuilder._extract_output_fields(schema)
        assert "hotel_id" in fields
        assert "hotel_name" in fields
        assert fields["hotel_id"] == "string"

    def test_types_compatible(self):
        assert ToolGraphBuilder._types_compatible("string", "string")
        assert ToolGraphBuilder._types_compatible("string", "integer")
        assert not ToolGraphBuilder._types_compatible("boolean", "array")

    def test_normalize_field_name(self):
        assert ToolGraphBuilder._normalize_field_name("hotelId") == "hotel_id"
        assert ToolGraphBuilder._normalize_field_name("CheckInDate") == "check_in_date"
        assert ToolGraphBuilder._normalize_field_name("hotel_ID") == "hotel_id"

    def test_field_root(self):
        assert ToolGraphBuilder._field_root("hotel_id") == "hotel"
        assert ToolGraphBuilder._field_root("booking_confirmation_id") == "booking_confirmation"
        assert ToolGraphBuilder._field_root("city") == "city"

    def test_save_and_load(self, sample_graph, tmp_path):
        path = tmp_path / "graph.pkl"
        import pickle
        with open(path, "wb") as f:
            pickle.dump(sample_graph, f)

        loaded = ToolGraphBuilder.load(path)
        assert loaded.number_of_nodes() == sample_graph.number_of_nodes()
        assert loaded.number_of_edges() == sample_graph.number_of_edges()

    def test_export_json(self, sample_registry, tmp_path):
        builder = ToolGraphBuilder(sample_registry)
        builder.build()
        json_path = tmp_path / "kg.json"
        builder.export_json(json_path)
        assert json_path.exists()

        import json
        with open(json_path) as f:
            data = json.load(f)
        assert "nodes" in data
        assert "edges" in data
        assert data["summary"]["total_nodes"] > 0
