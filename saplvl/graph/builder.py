"""Build a Tool Graph from a ToolRegistry using NetworkX."""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from saplvl.ingestor.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Common REST patterns: parameter name suffixes that indicate entity IDs
ID_SUFFIXES = ("_id", "id", "_key", "_code", "_number", "_token")
ENTITY_FIELDS = ("name", "title", "email", "address", "url", "location", "city", "country")


class ToolGraphBuilder:
    """Constructs a directed graph capturing tool/API relationships.

    Node: (tool_name, api_name) tuple
    Node attributes: category, description, input_params, inferred_output_fields

    Edge types:
    - same_category: tools in the same ToolBench category
    - parameter_compatibility: output of A likely feeds input of B
    - semantic_similarity: description embeddings are similar
    """

    def __init__(
        self,
        registry: ToolRegistry,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.65,
    ):
        self.registry = registry
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        self.graph = nx.DiGraph()

    def build(self) -> nx.DiGraph:
        """Full build pipeline."""
        logger.info("Building tool graph...")
        self._add_nodes()
        logger.info("Added %d nodes", self.graph.number_of_nodes())

        self._add_category_edges()
        cat_edges = self.graph.number_of_edges()
        logger.info("Added %d category edges", cat_edges)

        self._add_parameter_compatibility_edges()
        param_edges = self.graph.number_of_edges() - cat_edges
        logger.info("Added %d parameter compatibility edges", param_edges)

        prev_edges = self.graph.number_of_edges()
        self._add_semantic_similarity_edges()
        sem_edges = self.graph.number_of_edges() - prev_edges
        logger.info("Added %d semantic similarity edges", sem_edges)

        logger.info(
            "Tool graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def _add_nodes(self) -> None:
        """Add one node per (tool_name, api_name) pair."""
        for tool in self.registry.all_tools():
            for endpoint in tool.api_list:
                node_id = (tool.tool_name, endpoint.name)
                input_params = {p.name: p.type for p in endpoint.all_parameters}
                inferred_outputs = self._infer_output_fields(endpoint.name, endpoint.description)

                self.graph.add_node(
                    node_id,
                    category=tool.category,
                    tool_name=tool.tool_name,
                    api_name=endpoint.name,
                    description=endpoint.description or tool.tool_description,
                    method=endpoint.method,
                    input_params=input_params,
                    inferred_outputs=inferred_outputs,
                )

    def _add_category_edges(self) -> None:
        """Add edges between all API nodes in the same category."""
        category_nodes: dict[str, list[tuple]] = {}
        for node, data in self.graph.nodes(data=True):
            cat = data.get("category", "")
            if cat:
                category_nodes.setdefault(cat, []).append(node)

        for cat, nodes in category_nodes.items():
            # Only connect within reasonable category sizes to avoid explosion
            if len(nodes) > 200:
                # Sample representative nodes for large categories
                continue
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1 :]:
                    if n1[0] != n2[0]:  # Don't connect same tool's APIs via category
                        self.graph.add_edge(n1, n2, edge_type="same_category", weight=1.0)
                        self.graph.add_edge(n2, n1, edge_type="same_category", weight=1.0)

    def _add_parameter_compatibility_edges(self) -> None:
        """Add directed edges where A's inferred outputs match B's inputs."""
        nodes = list(self.graph.nodes(data=True))

        for _, data_a in nodes:
            outputs_a = data_a.get("inferred_outputs", set())
            if not outputs_a:
                continue

            node_a = (data_a["tool_name"], data_a["api_name"])

            for _, data_b in nodes:
                node_b = (data_b["tool_name"], data_b["api_name"])
                if node_a == node_b:
                    continue

                inputs_b = set(data_b.get("input_params", {}).keys())
                # Check if any output field of A matches an input field of B
                overlap = self._param_name_overlap(outputs_a, inputs_b)
                if overlap:
                    self.graph.add_edge(
                        node_a,
                        node_b,
                        edge_type="parameter_compatibility",
                        weight=2.0,
                        shared_params=list(overlap),
                    )

    def _add_semantic_similarity_edges(self) -> None:
        """Add edges between semantically similar API descriptions."""
        nodes = list(self.graph.nodes())
        if len(nodes) < 2:
            return

        descriptions = [
            self.graph.nodes[n].get("description", "") for n in nodes
        ]

        logger.info("Computing semantic embeddings for %d nodes...", len(nodes))
        encoder = SentenceTransformer(self.embedding_model_name)
        embeddings = encoder.encode(descriptions, show_progress_bar=True, batch_size=256)
        embeddings = np.array(embeddings)

        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        # Compute similarities in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(nodes), batch_size):
            batch_end = min(i + batch_size, len(nodes))
            sim_matrix = embeddings[i:batch_end] @ embeddings.T

            for bi, row in enumerate(sim_matrix):
                global_i = i + bi
                for j in range(global_i + 1, len(nodes)):
                    if row[j] >= self.similarity_threshold:
                        n1, n2 = nodes[global_i], nodes[j]
                        # Don't add if already connected by stronger edge type
                        if not self.graph.has_edge(n1, n2):
                            self.graph.add_edge(
                                n1, n2,
                                edge_type="semantic_similarity",
                                weight=float(row[j]),
                            )
                            self.graph.add_edge(
                                n2, n1,
                                edge_type="semantic_similarity",
                                weight=float(row[j]),
                            )

    @staticmethod
    def _infer_output_fields(api_name: str, description: str) -> set[str]:
        """Infer likely output field names from an API's name and description.

        Heuristic: REST GET endpoints typically return the entity they name.
        E.g., 'get_user' likely returns a user_id, 'search_hotels' returns hotel_id.
        """
        fields = set()
        combined = f"{api_name} {description}".lower()

        # Extract entity names from API name patterns like get_X, search_X, list_X
        patterns = re.findall(r"(?:get|search|find|list|fetch|retrieve)_?(\w+)", combined)
        for entity in patterns:
            entity = entity.rstrip("s")  # Depluralize
            if entity:
                fields.add(f"{entity}_id")
                fields.add(entity)

        # Common output patterns
        if any(word in combined for word in ["book", "create", "register", "order"]):
            fields.add("booking_id")
            fields.add("order_id")
            fields.add("confirmation_id")

        if any(word in combined for word in ["search", "find", "list"]):
            fields.add("results")
            fields.add("items")

        return fields

    @staticmethod
    def _param_name_overlap(outputs: set[str], inputs: set[str]) -> set[str]:
        """Find parameter name matches between outputs and inputs.

        Uses substring matching for ID-like fields.
        """
        matches = set()
        for out_field in outputs:
            for in_field in inputs:
                # Exact match
                if out_field == in_field:
                    matches.add(in_field)
                    continue
                # ID suffix match: hotel_id matches hotel_id, property_id
                if out_field.endswith("_id") and in_field.endswith("_id"):
                    out_entity = out_field[:-3]
                    in_entity = in_field[:-3]
                    if out_entity == in_entity or out_entity in in_entity or in_entity in out_entity:
                        matches.add(in_field)
        return matches

    def save(self, path: Path) -> None:
        """Save the graph to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info("Saved tool graph to %s", path)

    @classmethod
    def load(cls, path: Path) -> nx.DiGraph:
        """Load a graph from disk."""
        with open(path, "rb") as f:
            graph = pickle.load(f)
        logger.info(
            "Loaded tool graph: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph
