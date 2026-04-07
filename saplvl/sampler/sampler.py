"""Tool chain sampling from the tool graph."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import networkx as nx

from saplvl.ingestor.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Edge type weights for neighbor selection
EDGE_WEIGHTS = {
    "parameter_compatibility": 3.0,
    "same_category": 2.0,
    "semantic_similarity": 1.0,
}


@dataclass
class SamplingConstraints:
    """Constraints for tool chain sampling."""

    min_tools: int = 2
    max_tools: int = 5
    min_steps: int = 3
    max_steps: int = 7
    categories: list[str] | None = None
    required_tools: list[str] | None = None
    allow_parallel: bool = True
    exclude_tools: list[str] = field(default_factory=list)


class ToolChainSampler:
    """Samples tool chains by walking the tool graph.

    Produces sequences of (tool_name, api_name) that form realistic
    multi-step, multi-tool chains for conversation generation.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        registry: ToolRegistry,
        rng: random.Random | None = None,
    ):
        self.graph = graph
        self.registry = registry
        self.rng = rng or random.Random()

    def sample_chain(self, constraints: SamplingConstraints | None = None) -> list[tuple[str, str]]:
        """Sample a tool chain that satisfies the given constraints.

        Algorithm:
        1. Pick a seed node (from constraints or random).
        2. Walk the graph, preferring parameter_compatibility edges.
        3. Stop when constraints are met.

        Returns list of (tool_name, api_name) in execution order.
        """
        constraints = constraints or SamplingConstraints()
        target_steps = self.rng.randint(constraints.min_steps, constraints.max_steps)

        seed = self._pick_seed(constraints)
        if seed is None:
            logger.warning("Could not find a suitable seed node")
            return []

        chain = [seed]
        visited = {seed}
        distinct_tools = {seed[0]}

        for _ in range(target_steps - 1):
            next_node = self._select_next(chain[-1], visited, constraints, distinct_tools)
            if next_node is None:
                break
            chain.append(next_node)
            visited.add(next_node)
            distinct_tools.add(next_node[0])

        # Validate against constraints
        if len(distinct_tools) < constraints.min_tools:
            # Try to inject a tool from a different category/tool
            chain = self._diversify_chain(chain, constraints, distinct_tools)

        return chain

    def sample_parallel_group(
        self, constraints: SamplingConstraints | None = None
    ) -> list[list[tuple[str, str]]]:
        """Sample a set of independent tool call groups.

        Returns a list of parallel groups, where each group is a sequential chain.
        E.g., [[search_flights], [search_hotels]] for parallel independent searches.
        """
        constraints = constraints or SamplingConstraints()
        seed = self._pick_seed(constraints)
        if seed is None:
            return []

        # Find neighbors that are independent (different tools, same category)
        neighbors = list(self.graph.successors(seed))
        independent = [
            n for n in neighbors
            if n[0] != seed[0] and self._is_independent(seed, n)
        ]

        if not independent:
            return [[seed]]

        # Pick 2-3 independent starting points
        num_parallel = min(self.rng.randint(2, 3), len(independent) + 1)
        selected = [seed] + self.rng.sample(independent, min(num_parallel - 1, len(independent)))

        groups = []
        for start in selected:
            sub_constraints = SamplingConstraints(
                min_tools=1, max_tools=2,
                min_steps=1, max_steps=3,
            )
            sub_chain = [start]
            visited = {start}
            for _ in range(sub_constraints.max_steps - 1):
                next_node = self._select_next(start, visited, sub_constraints, {start[0]})
                if next_node:
                    sub_chain.append(next_node)
                    visited.add(next_node)
            groups.append(sub_chain)

        return groups

    def _pick_seed(self, constraints: SamplingConstraints) -> tuple[str, str] | None:
        """Pick a starting node based on constraints."""
        candidates = list(self.graph.nodes())

        if constraints.categories:
            candidates = [
                n for n in candidates
                if self.graph.nodes[n].get("category") in constraints.categories
            ]

        if constraints.required_tools:
            required_candidates = [
                n for n in candidates
                if n[0] in constraints.required_tools
            ]
            if required_candidates:
                candidates = required_candidates

        if constraints.exclude_tools:
            candidates = [
                n for n in candidates
                if n[0] not in constraints.exclude_tools
            ]

        if not candidates:
            return None

        return self.rng.choice(candidates)

    def _select_next(
        self,
        current: tuple[str, str],
        visited: set[tuple[str, str]],
        constraints: SamplingConstraints,
        distinct_tools: set[str],
    ) -> tuple[str, str] | None:
        """Select the next node in the chain using weighted edge types."""
        neighbors = [n for n in self.graph.successors(current) if n not in visited]

        if constraints.exclude_tools:
            neighbors = [n for n in neighbors if n[0] not in constraints.exclude_tools]

        if not neighbors:
            # Fallback: try any node in the graph not yet visited
            all_nodes = [n for n in self.graph.nodes() if n not in visited]
            if constraints.exclude_tools:
                all_nodes = [n for n in all_nodes if n[0] not in constraints.exclude_tools]
            if not all_nodes:
                return None
            return self.rng.choice(all_nodes)

        # Weight neighbors by edge type
        weights = []
        for neighbor in neighbors:
            edge_data = self.graph.edges[current, neighbor]
            edge_type = edge_data.get("edge_type", "semantic_similarity")
            base_weight = EDGE_WEIGHTS.get(edge_type, 1.0)

            # Boost weight for new tools (to meet min_tools constraint)
            if neighbor[0] not in distinct_tools and len(distinct_tools) < constraints.min_tools:
                base_weight *= 2.0

            weights.append(base_weight)

        total = sum(weights)
        if total == 0:
            return self.rng.choice(neighbors)

        probs = [w / total for w in weights]
        return self.rng.choices(neighbors, weights=probs, k=1)[0]

    def _diversify_chain(
        self,
        chain: list[tuple[str, str]],
        constraints: SamplingConstraints,
        distinct_tools: set[str],
    ) -> list[tuple[str, str]]:
        """Add nodes from different tools to meet min_tools constraint."""
        needed = constraints.min_tools - len(distinct_tools)
        visited = set(chain)

        for _ in range(needed):
            # Find nodes from tools not yet in the chain
            candidates = [
                n for n in self.graph.nodes()
                if n[0] not in distinct_tools
                and n not in visited
                and n[0] not in constraints.exclude_tools
            ]
            if not candidates:
                break

            # Prefer nodes connected to the last node in the chain
            connected = [c for c in candidates if self.graph.has_edge(chain[-1], c)]
            if connected:
                pick = self.rng.choice(connected)
            else:
                pick = self.rng.choice(candidates)

            chain.append(pick)
            visited.add(pick)
            distinct_tools.add(pick[0])

        return chain

    def _is_independent(self, node_a: tuple, node_b: tuple) -> bool:
        """Check if two nodes are independent (can be executed in parallel)."""
        # Independent if there's no directed path between them of length 1
        # and they don't share parameter compatibility
        if self.graph.has_edge(node_a, node_b):
            edge = self.graph.edges[node_a, node_b]
            if edge.get("edge_type") == "parameter_compatibility":
                return False
        return True
