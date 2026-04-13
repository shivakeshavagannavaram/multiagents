"""Build a typed knowledge graph from a ToolRegistry using NetworkX."""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from conv_gen.ingestor.registry import ToolRegistry

logger = logging.getLogger(__name__)

HARD_EDGE_THRESHOLD = 0.7
SOFT_EDGE_THRESHOLD = 0.6

CROSS_TOOL_GENERIC_FRACTION = 0.005


class ToolGraphBuilder:
    """Constructs the typed KG over tools, endpoints, parameters, and output fields."""

    def __init__(
        self,
        registry: ToolRegistry,
        embedding_model: str = "all-MiniLM-L6-v2",
        semantic_threshold: float = 0.75,
        cross_tool_generic_fraction: float = CROSS_TOOL_GENERIC_FRACTION,
    ):
        self.registry = registry
        self.embedding_model_name = embedding_model
        self.semantic_threshold = semantic_threshold
        self.cross_tool_generic_fraction = cross_tool_generic_fraction
        self.graph = nx.DiGraph()
        self._field_idf: dict[str, float] = {}
        self._total_endpoints = 0
        self._cross_tool_generic_fields: frozenset[str] = frozenset()

    def build(self) -> nx.DiGraph:
        """Full KG build pipeline."""
        logger.info("Building Knowledge Graph...")
        self._add_structural_nodes()
        self._add_structural_edges()
        self._compute_field_idf()
        self._add_same_tool_edges()
        self._add_data_flow_edges()
        self._add_semantic_cross_category_edges()

        node_types: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("node_type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        edge_types: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            t = data.get("edge_type", "unknown")
            edge_types[t] = edge_types.get(t, 0) + 1

        hard = sum(1 for _, _, d in self.graph.edges(data=True)
                   if d.get("edge_type") == "feeds_into" and d.get("edge_class") == "hard")
        soft = sum(1 for _, _, d in self.graph.edges(data=True)
                   if d.get("edge_type") == "feeds_into" and d.get("edge_class") == "soft")

        logger.info("KG built: %d nodes, %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())
        logger.info("  Nodes: %s", node_types)
        logger.info("  Edges: %s", edge_types)
        logger.info("  feeds_into: %d hard, %d soft", hard, soft)
        return self.graph

    @staticmethod
    def _normalize_field_name(name: str) -> str:
        """Canonical snake_case: hotelId → hotel_id, CheckInDate → check_in_date."""
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
        # Lowercase everything
        s = s.lower()
        # Collapse multiple underscores
        s = re.sub(r'_+', '_', s)
        # Strip leading/trailing underscores
        s = s.strip('_')
        return s

    @staticmethod
    def _field_root(name: str) -> str:
        """Strip common suffixes: hotel_id → hotel, check_in_date → check_in."""
        normalized = ToolGraphBuilder._normalize_field_name(name)
        for suffix in ("_id", "_identifier", "_code", "_key", "_number",
                        "_token", "_date", "_time", "_name", "_type",
                        "_url", "_uri", "_link"):
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                return normalized[:-len(suffix)]
        return normalized

    def _compute_field_idf(self) -> None:
        """Compute per-field IDF and auto-discover generic fields (frequency + category uniformity)."""
        field_doc_count: Counter = Counter()
        endpoint_count = 0

        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "endpoint":
                endpoint_count += 1

        self._total_endpoints = endpoint_count

        endpoint_fields: dict[str, set[str]] = {}
        field_categories: dict[str, set[str]] = defaultdict(set)

        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") in ("parameter", "output_field"):
                ep = data.get("endpoint", "")
                norm_name = self._normalize_field_name(data["name"])
                endpoint_fields.setdefault(ep, set()).add(norm_name)
                if ep and ep in self.graph.nodes:
                    cat = self.graph.nodes[ep].get("category", "")
                    if cat:
                        field_categories[norm_name].add(cat)

        for ep, fields in endpoint_fields.items():
            for field_name in fields:
                field_doc_count[field_name] += 1

        for field_name, doc_count in field_doc_count.items():
            self._field_idf[field_name] = math.log(endpoint_count / doc_count) if doc_count > 0 else 0

        frequency_threshold = endpoint_count * self.cross_tool_generic_fraction
        high_freq_fields = {
            f for f, dc in field_doc_count.items()
            if dc >= frequency_threshold
        }

        total_categories = len({
            d.get("category", "") for _, d in self.graph.nodes(data=True)
            if d.get("node_type") == "endpoint" and d.get("category")
        })
        uniformity_threshold = total_categories * 0.5

        generic = set()
        bridge = set()
        for field_name in high_freq_fields:
            num_cats = len(field_categories.get(field_name, set()))
            if num_cats >= uniformity_threshold:
                generic.add(field_name)
            else:
                bridge.add(field_name)

        self._cross_tool_generic_fields = frozenset(generic)

        logger.info(
            "Field IDF computed: %d unique fields",
            len(field_doc_count),
        )
        logger.info(
            "  High-frequency fields: %d (>=%.1f%% of endpoints)",
            len(high_freq_fields), self.cross_tool_generic_fraction * 100,
        )
        logger.info(
            "  → Generic (uniform across >50%% of %d categories): %d — filtered from cross-tool edges",
            total_categories, len(generic),
        )
        logger.info(
            "  → Bridge (clustered in specific categories): %d — kept as cross-domain connectors",
            len(bridge),
        )

        generic_sorted = sorted(
            [(f, field_doc_count[f], len(field_categories.get(f, set())))
             for f in generic],
            key=lambda x: -x[1],
        )
        bridge_sorted = sorted(
            [(f, field_doc_count[f], len(field_categories.get(f, set())))
             for f in bridge],
            key=lambda x: -x[1],
        )
        logger.info(
            "  Generic examples: %s",
            [(f, dc, nc) for f, dc, nc in generic_sorted[:15]],
        )
        logger.info(
            "  Bridge examples: %s",
            [(f, dc, nc) for f, dc, nc in bridge_sorted[:15]],
        )

    def _is_generic_field(self, name: str) -> bool:
        norm = self._normalize_field_name(name)
        return norm in self._cross_tool_generic_fields

    def _get_field_specificity(self, name: str) -> float:
        """0–1 specificity score. Higher = rarer = more chain-signal."""
        norm = self._normalize_field_name(name)
        idf = self._field_idf.get(norm, 0)
        if self._total_endpoints <= 1:
            return 0.5
        max_idf = math.log(self._total_endpoints)
        return min(idf / max_idf, 1.0) if max_idf > 0 else 0.5

    def _add_structural_nodes(self) -> None:
        """Add category, tool, endpoint, parameter, and output_field nodes."""
        categories_seen = set()

        for tool in self.registry.all_tools():
            cat = tool.category or "uncategorized"

            if cat not in categories_seen:
                self.graph.add_node(f"category:{cat}", node_type="category", name=cat)
                categories_seen.add(cat)

            tool_id = f"tool:{tool.tool_name}"
            self.graph.add_node(
                tool_id, node_type="tool", name=tool.tool_name,
                category=cat, description=tool.tool_description,
                num_endpoints=len(tool.api_list),
            )

            for endpoint in tool.api_list:
                ep_id = f"endpoint:{tool.tool_name}/{endpoint.name}"
                schema_quality = self._assess_schema_quality(endpoint.response_schema)

                self.graph.add_node(
                    ep_id, node_type="endpoint", name=endpoint.name,
                    tool_name=tool.tool_name, category=cat,
                    description=endpoint.description or tool.tool_description,
                    method=endpoint.method, schema_quality=schema_quality,
                )

                for param in endpoint.all_parameters:
                    norm_name = self._normalize_field_name(param.name)
                    param_id = f"param:{tool.tool_name}/{endpoint.name}/{param.name}"
                    self.graph.add_node(
                        param_id, node_type="parameter",
                        name=param.name, normalized_name=norm_name,
                        param_type=param.type, description=param.description,
                        endpoint=ep_id,
                        is_required=param in endpoint.required_parameters,
                    )

                # Filter out echoed input params — they aren't real outputs.
                input_param_names = {
                    self._normalize_field_name(p.name)
                    for p in endpoint.all_parameters
                }
                output_fields = self._extract_output_fields(endpoint.response_schema)
                for field_name, field_type in output_fields.items():
                    norm_name = self._normalize_field_name(field_name)
                    if norm_name in input_param_names:
                        continue  # skip echoed input params
                    out_id = f"output:{tool.tool_name}/{endpoint.name}/{field_name}"
                    self.graph.add_node(
                        out_id, node_type="output_field",
                        name=field_name, normalized_name=norm_name,
                        field_type=field_type, endpoint=ep_id,
                        schema_quality=schema_quality,
                    )

        logger.info("Added %d structural nodes", self.graph.number_of_nodes())

    def _add_structural_edges(self) -> None:
        """Add belongs_to, has_endpoint, takes_input, produces edges."""
        for node, data in self.graph.nodes(data=True):
            nt = data.get("node_type")
            if nt == "tool":
                self.graph.add_edge(node, f"category:{data['category']}", edge_type="belongs_to")
            elif nt == "endpoint":
                self.graph.add_edge(f"tool:{data['tool_name']}", node, edge_type="has_endpoint")
            elif nt == "parameter":
                self.graph.add_edge(data["endpoint"], node, edge_type="takes_input")
            elif nt == "output_field":
                self.graph.add_edge(data["endpoint"], node, edge_type="produces")

        logger.info("Added %d structural edges", self.graph.number_of_edges())

    def _add_same_tool_edges(self) -> None:
        """Connect endpoints within the same tool via shared normalized fields."""
        count = 0
        tool_endpoints: dict[str, list[str]] = {}

        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "endpoint":
                tool_endpoints.setdefault(data["tool_name"], []).append(node)

        for tool_name, endpoints in tool_endpoints.items():
            if len(endpoints) < 2:
                continue

            for ep_a in endpoints:
                outputs_a = self._get_endpoint_outputs_normalized(ep_a)
                for ep_b in endpoints:
                    if ep_a == ep_b:
                        continue
                    inputs_b = self._get_endpoint_inputs_normalized(ep_b)
                    shared = self._normalized_match(outputs_a, inputs_b, same_tool=True)
                    if shared:
                        self.graph.add_edge(
                            ep_a, ep_b, edge_type="same_tool",
                            shared_fields=[s[0] for s in shared],
                            confidence=1.0,
                            edge_class="hard",
                        )
                        count += 1

        logger.info("Added %d same_tool edges", count)

    def _add_data_flow_edges(self) -> None:
        """Add feeds_into edges with confidence from name match + type + category + specificity."""
        outputs_by_norm: dict[str, list[str]] = {}
        params_by_norm: dict[str, list[str]] = {}

        for node, data in self.graph.nodes(data=True):
            norm = data.get("normalized_name", "")
            if not norm:
                continue
            if data.get("node_type") == "output_field":
                outputs_by_norm.setdefault(norm, []).append(node)
            elif data.get("node_type") == "parameter":
                params_by_norm.setdefault(norm, []).append(node)

        outputs_by_root: dict[str, list[str]] = {}
        params_by_root: dict[str, list[str]] = {}

        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "output_field":
                root = self._field_root(data["name"])
                outputs_by_root.setdefault(root, []).append(node)
            elif data.get("node_type") == "parameter":
                root = self._field_root(data["name"])
                params_by_root.setdefault(root, []).append(node)

        count = 0
        hard_count = 0
        soft_count = 0

        # Pass 1: exact normalized name matches
        for norm_name, output_nodes in outputs_by_norm.items():
            if norm_name in self._cross_tool_generic_fields:
                continue

            param_nodes = params_by_norm.get(norm_name, [])
            if not param_nodes:
                continue

            for out_node in output_nodes:
                out_data = self.graph.nodes[out_node]
                out_ep = out_data["endpoint"]
                out_tool = out_ep.split("/")[0].replace("endpoint:", "")
                out_cat = self.graph.nodes[out_ep].get("category", "")

                for param_node in param_nodes:
                    param_data = self.graph.nodes[param_node]
                    param_ep = param_data["endpoint"]
                    param_tool = param_ep.split("/")[0].replace("endpoint:", "")

                    if out_tool == param_tool:
                        continue

                    if not self._types_compatible(
                        out_data.get("field_type", "string"),
                        param_data.get("param_type", "string"),
                    ):
                        continue

                    param_cat = self.graph.nodes[param_ep].get("category", "")
                    confidence = self._compute_confidence(
                        name_quality=1.0,
                        type_a=out_data.get("field_type", "string"),
                        type_b=param_data.get("param_type", "string"),
                        same_category=(out_cat == param_cat),
                        field_name=norm_name,
                    )

                    if confidence < SOFT_EDGE_THRESHOLD:
                        continue

                    edge_class = "hard" if confidence >= HARD_EDGE_THRESHOLD else "soft"
                    self.graph.add_edge(
                        out_node, param_node,
                        edge_type="feeds_into",
                        confidence=round(confidence, 3),
                        edge_class=edge_class,
                    )
                    count += 1
                    if edge_class == "hard":
                        hard_count += 1
                    else:
                        soft_count += 1

        # Pass 2: root matches (e.g., hotel_id output → hotel_code param)
        for root, output_nodes in outputs_by_root.items():
            if not root or len(root) < 3:
                continue
            if root in self._cross_tool_generic_fields:
                continue
            param_nodes = params_by_root.get(root, [])
            if not param_nodes:
                continue

            for out_node in output_nodes:
                out_data = self.graph.nodes[out_node]
                out_norm = out_data.get("normalized_name", "")
                out_ep = out_data["endpoint"]
                out_tool = out_ep.split("/")[0].replace("endpoint:", "")
                out_cat = self.graph.nodes[out_ep].get("category", "")

                for param_node in param_nodes:
                    param_data = self.graph.nodes[param_node]
                    param_norm = param_data.get("normalized_name", "")
                    param_ep = param_data["endpoint"]
                    param_tool = param_ep.split("/")[0].replace("endpoint:", "")

                    if out_tool == param_tool:
                        continue

                    if self.graph.has_edge(out_node, param_node):
                        continue

                    if out_norm == param_norm:
                        continue

                    if not self._types_compatible(
                        out_data.get("field_type", "string"),
                        param_data.get("param_type", "string"),
                    ):
                        continue

                    param_cat = self.graph.nodes[param_ep].get("category", "")
                    confidence = self._compute_confidence(
                        name_quality=0.5,
                        type_a=out_data.get("field_type", "string"),
                        type_b=param_data.get("param_type", "string"),
                        same_category=(out_cat == param_cat),
                        field_name=root,
                    )

                    if confidence < SOFT_EDGE_THRESHOLD:
                        continue

                    edge_class = "hard" if confidence >= HARD_EDGE_THRESHOLD else "soft"
                    self.graph.add_edge(
                        out_node, param_node,
                        edge_type="feeds_into",
                        confidence=round(confidence, 3),
                        edge_class=edge_class,
                    )
                    count += 1
                    if edge_class == "hard":
                        hard_count += 1
                    else:
                        soft_count += 1

        logger.info("Added %d feeds_into edges (%d hard, %d soft)", count, hard_count, soft_count)

    def _compute_confidence(
        self,
        name_quality: float,
        type_a: str,
        type_b: str,
        same_category: bool,
        field_name: str,
    ) -> float:
        """Compute confidence score for a data flow edge.

        Weighted combination: name 0.20 + type 0.10 + category 0.30 + specificity 0.40.
        """
        type_score = 1.0 if type_a == type_b else 0.7
        category_score = 1.0 if same_category else 0.3
        specificity = self._get_field_specificity(field_name)

        confidence = (
            name_quality * 0.20 +
            type_score * 0.10 +
            category_score * 0.30 +
            specificity * 0.40
        )

        return confidence

    def _compute_category_affinity(self, min_shared_fields: int = 3) -> set[tuple[str, str]]:
        """Category pairs sharing ≥ min_shared_fields non-generic fields — data-driven, no hardcoding."""
        cat_fields: dict[str, set[str]] = defaultdict(set)

        for node, data in self.graph.nodes(data=True):
            nt = data.get("node_type")
            if nt not in ("parameter", "output_field"):
                continue

            norm = data.get("normalized_name", "")
            if not norm or self._is_generic_field(norm):
                continue

            ep = data.get("endpoint", "")
            if ep and ep in self.graph.nodes:
                cat = self.graph.nodes[ep].get("category", "")
                if cat:
                    cat_fields[cat].add(norm)

        allowed_pairs: set[tuple[str, str]] = set()
        categories = sorted(cat_fields.keys())

        for i, cat_a in enumerate(categories):
            for cat_b in categories[i + 1:]:
                shared = cat_fields[cat_a] & cat_fields[cat_b]
                if len(shared) >= min_shared_fields:
                    allowed_pairs.add((cat_a, cat_b))

        logger.info(
            "Category affinity: %d pairs allowed (from %d categories, min %d shared fields)",
            len(allowed_pairs), len(categories), min_shared_fields,
        )

        for cat_a, cat_b in sorted(allowed_pairs)[:10]:
            shared = cat_fields[cat_a] & cat_fields[cat_b]
            examples = sorted(shared)[:5]
            logger.debug("  %s <-> %s: %d fields (%s)", cat_a, cat_b, len(shared), examples)

        return allowed_pairs

    def _add_semantic_cross_category_edges(self) -> None:
        """Add semantic_bridge edges between endpoints in related cross-categories via embeddings."""
        import numpy as np
        from sentence_transformers import SentenceTransformer

        allowed_pairs = self._compute_category_affinity(min_shared_fields=3)

        def _is_allowed_pair(cat_a: str, cat_b: str) -> bool:
            return (cat_a, cat_b) in allowed_pairs or (cat_b, cat_a) in allowed_pairs

        endpoints_by_cat: dict[str, list[str]] = {}
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "endpoint":
                cat = data.get("category", "")
                endpoints_by_cat.setdefault(cat, []).append(node)

        all_endpoints = []
        all_descriptions = []
        all_categories = []
        for cat, eps in endpoints_by_cat.items():
            for ep in eps:
                data = self.graph.nodes[ep]
                desc = data.get("description", "")
                tool_desc = ""
                tool_id = f"tool:{data['tool_name']}"
                if tool_id in self.graph.nodes:
                    tool_desc = self.graph.nodes[tool_id].get("description", "")
                combined = f"{data['tool_name']} {data['name']}: {desc or tool_desc}"
                all_endpoints.append(ep)
                all_descriptions.append(combined)
                all_categories.append(cat)

        if len(all_endpoints) < 2:
            return

        logger.info("Computing semantic embeddings for %d endpoints...", len(all_endpoints))
        encoder = SentenceTransformer(self.embedding_model_name)
        embeddings = encoder.encode(all_descriptions, show_progress_bar=True, batch_size=256)
        embeddings = np.array(embeddings)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        count = 0
        skipped_category = 0
        batch_size = 500
        for i in range(0, len(all_endpoints), batch_size):
            batch_end = min(i + batch_size, len(all_endpoints))
            sim_matrix = embeddings[i:batch_end] @ embeddings.T

            for bi, row in enumerate(sim_matrix):
                global_i = i + bi
                cat_i = all_categories[global_i]

                for j in range(global_i + 1, len(all_endpoints)):
                    cat_j = all_categories[j]

                    if cat_i == cat_j:
                        continue

                    if row[j] < self.semantic_threshold:
                        continue

                    if not _is_allowed_pair(cat_i, cat_j):
                        skipped_category += 1
                        continue

                    ep_a = all_endpoints[global_i]
                    ep_b = all_endpoints[j]

                    if not self.graph.has_edge(ep_a, ep_b):
                        self.graph.add_edge(
                            ep_a, ep_b,
                            edge_type="semantic_bridge",
                            confidence=round(float(row[j]), 3),
                            edge_class="soft",
                        )
                        count += 1
                    if not self.graph.has_edge(ep_b, ep_a):
                        self.graph.add_edge(
                            ep_b, ep_a,
                            edge_type="semantic_bridge",
                            confidence=round(float(row[j]), 3),
                            edge_class="soft",
                        )
                        count += 1

        logger.info(
            "Added %d semantic_bridge edges (skipped %d from unrelated category pairs)",
            count, skipped_category,
        )

    def _get_endpoint_outputs_normalized(self, ep_id: str) -> dict[str, tuple[str, str]]:
        """{normalized_name: (original_name, field_type)} for an endpoint's outputs."""
        result = {}
        for succ in self.graph.successors(ep_id):
            data = self.graph.nodes[succ]
            if data.get("node_type") == "output_field":
                norm = data.get("normalized_name", data["name"])
                result[norm] = (data["name"], data.get("field_type", "string"))
        return result

    def _get_endpoint_inputs_normalized(self, ep_id: str) -> dict[str, tuple[str, str]]:
        """{normalized_name: (original_name, param_type)} for an endpoint's inputs."""
        result = {}
        for succ in self.graph.successors(ep_id):
            data = self.graph.nodes[succ]
            if data.get("node_type") == "parameter":
                norm = data.get("normalized_name", data["name"])
                result[norm] = (data["name"], data.get("param_type", "string"))
        return result

    def _normalized_match(
        self,
        outputs: dict[str, tuple[str, str]],
        inputs: dict[str, tuple[str, str]],
        same_tool: bool = False,
    ) -> list[tuple[str, float]]:
        """Match outputs to inputs by normalized name. Returns [(name, confidence)]."""
        matches = []
        for norm_name, (orig_out, out_type) in outputs.items():
            if not same_tool and self._is_generic_field(orig_out):
                continue
            if norm_name in inputs:
                _, in_type = inputs[norm_name]
                if self._types_compatible(out_type, in_type):
                    matches.append((norm_name, 1.0))
        return matches

    @staticmethod
    def _types_compatible(type_a: str, type_b: str) -> bool:
        if type_a == type_b:
            return True
        compatible_groups = [
            {"string", "integer", "number"},
        ]
        for group in compatible_groups:
            if type_a in group and type_b in group:
                return True
        return False

    @staticmethod
    def _extract_output_fields(schema: dict[str, Any] | None) -> dict[str, str]:
        """{field_name: field_type} from a response schema."""
        if not schema or not isinstance(schema, dict):
            return {}
        fields: dict[str, str] = {}
        ToolGraphBuilder._walk_schema(schema, fields)
        return fields

    @staticmethod
    def _walk_schema(schema: Any, fields: dict[str, str], depth: int = 0) -> None:
        """Recursively extract field names and types from a schema."""
        if depth > 4 or not isinstance(schema, dict):
            return

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, prop in properties.items():
                if isinstance(prop, dict):
                    field_type = prop.get("type", "string")
                    fields[key] = field_type if isinstance(field_type, str) else "string"
                    ToolGraphBuilder._walk_schema(prop, fields, depth + 1)
                    items = prop.get("items", {})
                    if isinstance(items, dict):
                        ToolGraphBuilder._walk_schema(items, fields, depth + 1)

        type_strings = {"str", "int", "float", "bool", "string", "integer", "number", "boolean", "NoneType"}
        for key, value in schema.items():
            if key in ("type", "properties", "items", "required", "description",
                       "default", "enum", "format", "additionalProperties"):
                continue
            if isinstance(value, str) and value.lower() in type_strings:
                fields[key] = value
            elif isinstance(value, dict):
                fields[key] = "object"
                ToolGraphBuilder._walk_schema(value, fields, depth + 1)
            elif isinstance(value, list):
                fields[key] = "array"
                for item in value:
                    if isinstance(item, dict):
                        ToolGraphBuilder._walk_schema(item, fields, depth + 1)
                        break

    @staticmethod
    def _assess_schema_quality(schema: dict[str, Any] | None) -> str:
        """Tag a schema as complete / inferred / minimal."""
        if not schema or not isinstance(schema, dict):
            return "minimal"
        props = schema.get("properties", {})
        if isinstance(props, dict) and len(props) > 0:
            return "complete"
        if len(schema) > 1:
            return "inferred"
        return "minimal"

    def save(self, path: Path) -> None:
        """Pickle the graph to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info("Saved KG to %s", path)

    @classmethod
    def load(cls, path: Path) -> nx.DiGraph:
        with open(path, "rb") as f:
            graph = pickle.load(f)
        logger.info("Loaded KG: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
        return graph

    def export_json(self, path: Path) -> None:
        """Dump the graph as plain JSON for inspection."""
        path.parent.mkdir(parents=True, exist_ok=True)

        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            entry = {"id": node_id, **data}
            nodes.append(entry)

        edges = []
        for src, dst, data in self.graph.edges(data=True):
            entry = {"source": src, "target": dst, **data}
            edges.append(entry)

        export = {
            "summary": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "node_types": {},
                "edge_types": {},
            },
            "nodes": nodes,
            "edges": edges,
        }

        for data in [d for _, d in self.graph.nodes(data=True)]:
            t = data.get("node_type", "unknown")
            export["summary"]["node_types"][t] = export["summary"]["node_types"].get(t, 0) + 1
        for _, _, data in self.graph.edges(data=True):
            t = data.get("edge_type", "unknown")
            export["summary"]["edge_types"][t] = export["summary"]["edge_types"].get(t, 0) + 1

        with open(path, "w") as f:
            json.dump(export, f, indent=2, default=str)
        logger.info("Exported KG JSON to %s", path)

    def export_html(self, path: Path, max_nodes: int = 2000) -> None:
        """Interactive pyvis HTML visualization."""
        from pyvis.network import Network

        path.parent.mkdir(parents=True, exist_ok=True)

        colors = {
            "category": "#e74c3c", "tool": "#3498db", "endpoint": "#2ecc71",
            "parameter": "#f39c12", "output_field": "#9b59b6",
        }

        nodes_to_show = set()
        if self.graph.number_of_nodes() > max_nodes:
            for node, data in self.graph.nodes(data=True):
                if data.get("node_type") in ("category", "tool", "endpoint"):
                    nodes_to_show.add(node)
            for src, dst, data in self.graph.edges(data=True):
                if data.get("edge_type") == "feeds_into":
                    nodes_to_show.add(src)
                    nodes_to_show.add(dst)
        else:
            nodes_to_show = set(self.graph.nodes())

        net = Network(height="900px", width="100%", directed=True, notebook=False)
        net.barnes_hut(gravity=-3000, spring_length=150)

        for node in nodes_to_show:
            data = self.graph.nodes[node]
            nt = data.get("node_type", "unknown")
            color = colors.get(nt, "#95a5a6")
            label = data.get("name", str(node))
            size_map = {"category": 30, "tool": 20, "endpoint": 15, "parameter": 8, "output_field": 8}
            size = size_map.get(nt, 10)

            title = f"Type: {nt}\nID: {node}"
            for k, v in data.items():
                if k not in ("node_type", "name") and v:
                    title += f"\n{k}: {v}"

            net.add_node(str(node), label=label, color=color, size=size, title=title)

        edge_colors = {
            "belongs_to": "#bdc3c7", "has_endpoint": "#3498db",
            "takes_input": "#f39c12", "produces": "#9b59b6",
            "feeds_into": "#e74c3c", "same_tool": "#2ecc71",
        }

        for src, dst, data in self.graph.edges(data=True):
            if src not in nodes_to_show or dst not in nodes_to_show:
                continue
            et = data.get("edge_type", "unknown")
            color = edge_colors.get(et, "#95a5a6")
            title = et
            if "confidence" in data:
                title += f" ({data['confidence']:.2f}, {data.get('edge_class', '?')})"
            net.add_edge(str(src), str(dst), color=color, title=title, label=et)

        net.save_graph(str(path))
        logger.info("Exported KG HTML visualization to %s", path)
