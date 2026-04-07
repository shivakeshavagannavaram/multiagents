"""Tests for the tool graph module."""

import pytest

from saplvl.graph.builder import ToolGraphBuilder


class TestToolGraphBuilder:
    def test_nodes_added(self, sample_graph, sample_registry):
        """All endpoints should be graph nodes."""
        expected_keys = sample_registry.all_endpoint_keys()
        for key in expected_keys:
            assert key in sample_graph.nodes()

    def test_category_edges(self, sample_graph):
        """Same-category tools should be connected."""
        # HotelFinder and FlightSearch are both Travel
        assert sample_graph.has_edge(
            ("HotelFinder", "search_hotels"), ("FlightSearch", "search_flights")
        )

    def test_parameter_compatibility_edges(self, sample_graph):
        """search -> book should have parameter compatibility."""
        assert sample_graph.has_edge(
            ("HotelFinder", "search_hotels"), ("HotelFinder", "book_hotel")
        )
        edge = sample_graph.edges[
            ("HotelFinder", "search_hotels"), ("HotelFinder", "book_hotel")
        ]
        assert edge["edge_type"] == "parameter_compatibility"

    def test_node_attributes(self, sample_graph):
        """Nodes should have expected attributes."""
        node_data = sample_graph.nodes[("HotelFinder", "search_hotels")]
        assert node_data["category"] == "Travel"
        assert node_data["tool_name"] == "HotelFinder"
        assert "input_params" in node_data
        assert "city" in node_data["input_params"]

    def test_graph_is_directed(self, sample_graph):
        assert sample_graph.is_directed()

    def test_infer_output_fields(self):
        fields = ToolGraphBuilder._infer_output_fields("search_hotels", "Search for available hotels")
        assert "hotel_id" in fields or "hotel" in fields

    def test_infer_output_fields_booking(self):
        fields = ToolGraphBuilder._infer_output_fields("book_room", "Book a hotel room")
        assert any("id" in f for f in fields)

    def test_param_name_overlap(self):
        outputs = {"hotel_id", "name"}
        inputs = {"hotel_id", "check_in"}
        overlap = ToolGraphBuilder._param_name_overlap(outputs, inputs)
        assert "hotel_id" in overlap

    def test_param_name_overlap_no_match(self):
        outputs = {"flight_id"}
        inputs = {"restaurant_id", "date"}
        overlap = ToolGraphBuilder._param_name_overlap(outputs, inputs)
        assert len(overlap) == 0

    def test_save_and_load(self, sample_graph, tmp_path):
        path = tmp_path / "graph.pkl"
        import pickle
        with open(path, "wb") as f:
            pickle.dump(sample_graph, f)

        loaded = ToolGraphBuilder.load(path)
        assert loaded.number_of_nodes() == sample_graph.number_of_nodes()
        assert loaded.number_of_edges() == sample_graph.number_of_edges()
