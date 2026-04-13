"""Tool chain sampling from the Knowledge Graph."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum

import networkx as nx

from conv_gen.ingestor.registry import ToolRegistry

logger = logging.getLogger(__name__)


class SamplingPattern(str, Enum):
    """Tool-calling patterns supported by the sampler."""

    SEQUENTIAL = "sequential"  # A → B → C (each depends on previous)
    PARALLEL = "parallel"  # A + B + C (independent, called simultaneously)
    SINGLE = "single"  # Just A (simple lookup)


@dataclass
class SamplingConstraints:
    """Composable constraints for tool chain sampling.

    Fields:
      min_steps, max_steps: inclusive chain-length range
      exact_steps: convenience — collapses min/max to this value
      min_tools, max_tools: distinct-tool count bounds
      max_categories: cap on distinct categories spanned
      categories: "only from" filter — restrict the chain to these categories
      must_include_categories: "at least one from each" quota
      required_tools: seed must come from one of these tools
      exclude_tools: hard exclusion from the candidate pool
    """

    min_tools: int = 2
    max_tools: int = 3
    min_steps: int = 2
    max_steps: int = 4
    exact_steps: int | None = None
    categories: list[str] | None = None
    must_include_categories: list[str] | None = None
    required_tools: list[str] | None = None
    exclude_tools: list[str] = field(default_factory=list)
    max_categories: int = 3

    def __post_init__(self) -> None:
        """Apply exact_steps shortcut and normalize empty lists to None."""
        if self.exact_steps is not None:
            if self.exact_steps < 1:
                raise ValueError(
                    f"exact_steps must be >= 1, got {self.exact_steps}"
                )
            self.min_steps = self.exact_steps
            self.max_steps = self.exact_steps

        if self.categories is not None and len(self.categories) == 0:
            self.categories = None
        if self.must_include_categories is not None and len(self.must_include_categories) == 0:
            self.must_include_categories = None


@dataclass
class SampledChain:
    """Result of sampling — includes the pattern and structured tool groups.

    For SEQUENTIAL: steps = [[A], [B], [C]] — one tool per step, in order
    For PARALLEL: steps = [[A, B, C]] — all tools in one step, simultaneous
    For FAN_OUT: steps = [[A], [B, C]] — first step sequential, second parallel
    For SINGLE: steps = [[A]] — one tool, one step
    """

    pattern: SamplingPattern
    steps: list[list[tuple[str, str]]]

    @property
    def flat_chain(self) -> list[tuple[str, str]]:
        """Flatten all steps into a single ordered list."""
        return [tool for step in self.steps for tool in step]

    @property
    def num_tools(self) -> int:
        return len(set(t for t, a in self.flat_chain))

    @property
    def num_steps(self) -> int:
        return len(self.flat_chain)


class ToolChainSampler:
    """Samples tool chains by traversing the Knowledge Graph."""

    def __init__(
        self,
        graph: nx.DiGraph,
        registry: ToolRegistry,
        rng: random.Random | None = None,
        coherence_client=None,
        coherence_model: str = "gpt-4.1-nano",
    ):
        self.graph = graph
        self.registry = registry
        self.rng = rng or random.Random()
        self.coherence_client = coherence_client
        self.coherence_model = coherence_model
        self._coherence_stats = {"hard_passed": 0, "soft_approved": 0, "soft_rejected": 0}

        # Pre-index endpoint nodes for fast access
        self._endpoints: list[str] = []
        self._endpoints_by_tool: dict[str, list[str]] = {}
        self._endpoints_by_category: dict[str, list[str]] = {}

        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "endpoint":
                self._endpoints.append(node)
                tool = data["tool_name"]
                cat = data.get("category", "")
                self._endpoints_by_tool.setdefault(tool, []).append(node)
                self._endpoints_by_category.setdefault(cat, []).append(node)

    # ------------------------------------------------------------------
    # Public sampling methods — one per pattern
    # ------------------------------------------------------------------

    def sample_sequential(
        self, constraints: SamplingConstraints | None = None
    ) -> SampledChain:
        """Sample a sequential chain A → B → C with hybrid hard/soft quality gate."""
        constraints = constraints or SamplingConstraints()
        best_chain: list[str] = []
        best_edge_classes: list[tuple[str, float]] = []

        for attempt in range(8):
            chain, edge_classes = self._walk_chain(constraints)

            if len(chain) > len(best_chain):
                best_chain = chain
                best_edge_classes = edge_classes

            if len(chain) < constraints.min_steps:
                continue

            if not self._chain_satisfies_must_include(chain, constraints):
                logger.debug(
                    "Chain missing must_include_categories=%s (attempt %d)",
                    constraints.must_include_categories, attempt,
                )
                continue

            if self._is_hard_chain(edge_classes):
                self._coherence_stats["hard_passed"] += 1
                best_chain = chain
                break

            if self.coherence_client is not None:
                if self._llm_coherence_check(chain):
                    self._coherence_stats["soft_approved"] += 1
                    best_chain = chain
                    break
                else:
                    self._coherence_stats["soft_rejected"] += 1
                    continue
            else:
                best_chain = chain
                break
        else:
            logger.warning(
                "Sampler could not produce a gate-passing chain after 8 "
                "attempts; returning best chain of length %d",
                len(best_chain),
            )

        steps = [[self._ep_to_tuple(ep)] for ep in best_chain]
        return SampledChain(pattern=SamplingPattern.SEQUENTIAL, steps=steps)

    @staticmethod
    def _is_hard_chain(edge_classes: list[tuple[str, float]]) -> bool:
        """True if every hop is same_tool or feeds_into_hard."""
        hard_classes = {"same_tool", "feeds_into_hard", "diversify_feeds_into_hard"}
        return all(ec in hard_classes for ec, _ in edge_classes)

    def _chain_satisfies_must_include(
        self, chain: list[str], constraints: SamplingConstraints
    ) -> bool:
        """Check the chain contains at least one endpoint from each required category."""
        required = constraints.must_include_categories
        if not required:
            return True
        chain_categories = {
            self.graph.nodes[ep].get("category", "") for ep in chain
        }
        return all(req in chain_categories for req in required)

    def _llm_coherence_check(self, chain: list[str]) -> bool:
        """Ask an LLM if the chain is a realistic single-session workflow (defaults to True on error)."""
        lines = []
        for i, ep_id in enumerate(chain, 1):
            data = self.graph.nodes[ep_id]
            tool_name = data.get("tool_name", "?")
            api_name = data.get("name", "?")
            endpoint = self.registry.get_endpoint(tool_name, api_name)
            desc = ""
            if endpoint and endpoint.description:
                desc = endpoint.description.strip()[:120]
            lines.append(f"{i}. {tool_name} — {api_name}" + (f": {desc}" if desc else ""))

        chain_text = "\n".join(lines)
        prompt = (
            "You are checking whether a sequence of API calls from "
            "different services can actually work together as a chain "
            "in one conversation.\n\n"
            f"Chain:\n{chain_text}\n\n"
            "Answer NO if ANY of these are true:\n"
            "- The tools come from different unrelated platforms/vendors "
            "that wouldn't share identifiers (e.g. Upcall webhook vs "
            "Viber webhook — different vendors, different IDs).\n"
            "- The output of one step cannot realistically feed the input "
            "of the next step (e.g. a YouTube video list doesn't give you "
            "Google Maps place IDs).\n"
            "- A real user would need external manual steps "
            "(copy-paste across systems, different logins) to bridge "
            "one tool to the next.\n"
            "- The chain mixes two unrelated tasks with only a thin "
            "narrative bridge (e.g. 'clean up email' + 'check stock prices').\n\n"
            "Answer YES only if every step's output can flow into the next "
            "step's input AND a single user would genuinely do all of "
            "these in one session on the same platform ecosystem.\n\n"
            "Answer with just one word: YES or NO."
        )

        try:
            response = self.coherence_client.chat.completions.create(
                model=self.coherence_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            answer = (response.choices[0].message.content or "").strip().upper()
            approved = answer.startswith("YES")
            if not approved:
                logger.debug("Coherence check REJECTED chain: %s", chain_text[:200])
            return approved
        except Exception as e:
            logger.warning("Coherence check failed, defaulting to accept: %s", e)
            return True

    def sample_parallel(
        self, constraints: SamplingConstraints | None = None
    ) -> SampledChain:
        """Sample independent parallel tools: A + B + C.

        Finds endpoints in the same category that have no data flow
        dependency between them.
        """
        constraints = constraints or SamplingConstraints()
        seed = self._pick_endpoint(constraints)
        if seed is None:
            return SampledChain(pattern=SamplingPattern.PARALLEL, steps=[])

        seed_data = self.graph.nodes[seed]
        seed_cat = seed_data.get("category", "")
        seed_tool = seed_data["tool_name"]

        # Find endpoints in the same category from different tools
        candidates = []
        for ep in self._endpoints_by_category.get(seed_cat, []):
            ep_data = self.graph.nodes[ep]
            if ep_data["tool_name"] != seed_tool and ep not in [seed]:
                if self._passes_constraints(ep, constraints):
                    candidates.append(ep)

        # Filter to only independent endpoints (no data flow between them and seed)
        independent = [c for c in candidates if self._is_independent(seed, c)]

        if not independent:
            # Broaden: try any category
            for ep in self._endpoints:
                ep_data = self.graph.nodes[ep]
                if (ep_data["tool_name"] != seed_tool
                        and ep != seed
                        and self._passes_constraints(ep, constraints)
                        and self._is_independent(seed, ep)):
                    independent.append(ep)
                    if len(independent) >= 20:  # enough candidates
                        break

        if not independent:
            return SampledChain(pattern=SamplingPattern.PARALLEL, steps=[[self._ep_to_tuple(seed)]])

        num_parallel = min(
            self.rng.randint(2, constraints.max_tools),
            len(independent) + 1,
        )
        selected = self.rng.sample(independent, min(num_parallel - 1, len(independent)))
        all_eps = [seed] + selected

        return SampledChain(
            pattern=SamplingPattern.PARALLEL,
            steps=[[self._ep_to_tuple(ep) for ep in all_eps]],
        )


    def sample_single(
        self, constraints: SamplingConstraints | None = None
    ) -> SampledChain:
        """Sample a single tool call."""
        constraints = constraints or SamplingConstraints(
            min_tools=1, max_tools=1, min_steps=1, max_steps=1
        )
        seed = self._pick_endpoint(constraints)
        if seed is None:
            return SampledChain(pattern=SamplingPattern.SINGLE, steps=[])
        return SampledChain(pattern=SamplingPattern.SINGLE, steps=[[self._ep_to_tuple(seed)]])

    # ------------------------------------------------------------------
    # Legacy interface
    # ------------------------------------------------------------------

    def sample_chain(
        self, constraints: SamplingConstraints | None = None
    ) -> list[tuple[str, str]]:
        """Sample a sequential tool chain (legacy interface)."""
        result = self.sample_sequential(constraints)
        return result.flat_chain

    def sample_parallel_group(
        self, constraints: SamplingConstraints | None = None
    ) -> list[list[tuple[str, str]]]:
        """Sample parallel groups (legacy interface)."""
        result = self.sample_parallel(constraints)
        return result.steps

    # ------------------------------------------------------------------
    # Internal: graph walking
    # ------------------------------------------------------------------

    def _walk_chain(
        self, constraints: SamplingConstraints | None = None
    ) -> tuple[list[str], list[tuple[str, float]]]:
        """Walk the KG to produce a chain; returns (endpoints, per-hop edge classes)."""
        constraints = constraints or SamplingConstraints()
        target_steps = self.rng.randint(constraints.min_steps, constraints.max_steps)

        seed = self._pick_endpoint(constraints)
        if seed is None:
            logger.warning("Could not find a suitable seed endpoint")
            return [], []

        chain = [seed]
        edge_classes: list[tuple[str, float]] = []
        visited_tools = {self.graph.nodes[seed]["tool_name"]}
        visited_eps = {seed}

        for _ in range(target_steps - 1):
            current = chain[-1]
            result = self._select_next(current, visited_eps, visited_tools, constraints)
            if result is None:
                break
            next_ep, edge_class, confidence = result
            chain.append(next_ep)
            edge_classes.append((edge_class, confidence))
            visited_eps.add(next_ep)
            visited_tools.add(self.graph.nodes[next_ep]["tool_name"])

        if len(visited_tools) < constraints.min_tools:
            chain, edge_classes = self._diversify_chain(
                chain, edge_classes, visited_tools, visited_eps, constraints
            )

        return chain, edge_classes

    def _select_next(
        self,
        current: str,
        visited_eps: set[str],
        visited_tools: set[str],
        constraints: SamplingConstraints,
    ) -> tuple[str, str, float] | None:
        """Pick next endpoint via tiered priority; returns (endpoint, edge_class, confidence)."""
        candidates: list[tuple[str, float, str, float]] = []

        # Tier 1: same_tool edges
        for succ in self.graph.successors(current):
            edge_data = self.graph.edges[current, succ]
            if edge_data.get("edge_type") != "same_tool":
                continue
            if succ in visited_eps:
                continue
            if not self._passes_constraints(succ, constraints):
                continue
            weight = 3.0
            if self.graph.nodes[succ]["tool_name"] in visited_tools and len(visited_tools) < constraints.min_tools:
                weight = 1.0
            candidates.append((succ, weight, "same_tool", 1.0))

        # Tier 2: hard feeds_into
        flow_targets = self._find_data_flow_targets(current)
        for target, confidence, edge_class in flow_targets:
            if target in visited_eps:
                continue
            if not self._passes_constraints(target, constraints):
                continue
            target_tool = self.graph.nodes[target]["tool_name"]
            if edge_class == "hard":
                weight = 4.0 * confidence
                if target_tool in visited_tools and len(visited_tools) < constraints.min_tools:
                    weight *= 0.5
                candidates.append((target, weight, "feeds_into_hard", confidence))

        # Tier 3: soft feeds_into
        if not candidates:
            for target, confidence, edge_class in flow_targets:
                if target in visited_eps:
                    continue
                if not self._passes_constraints(target, constraints):
                    continue
                if edge_class == "soft":
                    weight = 2.0 * confidence
                    candidates.append((target, weight, "feeds_into_soft", confidence))

        # Tier 4: semantic_bridge
        if not candidates:
            for succ in self.graph.successors(current):
                edge_data = self.graph.edges[current, succ]
                if edge_data.get("edge_type") != "semantic_bridge":
                    continue
                if succ in visited_eps:
                    continue
                if not self._passes_constraints(succ, constraints):
                    continue
                if self.graph.nodes[succ].get("node_type") != "endpoint":
                    continue
                confidence = edge_data.get("confidence", 0.5)
                weight = 2.5 * confidence
                candidates.append((succ, weight, "semantic_bridge", confidence))

        # Tier 5: same-category fallback
        if not candidates:
            current_cat = self.graph.nodes[current].get("category", "")
            current_outputs = self._get_output_field_names(current)
            same_cat_eps = self._endpoints_by_category.get(current_cat, [])
            for ep in same_cat_eps:
                if ep in visited_eps:
                    continue
                ep_tool = self.graph.nodes[ep]["tool_name"]
                if ep_tool in visited_tools and len(visited_tools) >= constraints.min_tools:
                    continue
                if not self._passes_constraints(ep, constraints):
                    continue
                ep_inputs = self._get_input_field_names(ep)
                shared = current_outputs & ep_inputs
                if shared:
                    conf = 0.5 + len(shared) * 0.3
                    candidates.append((ep, conf, "same_category_fallback", conf))

        if not candidates:
            return None

        weights = [c[1] for c in candidates]
        chosen_idx = self.rng.choices(range(len(candidates)), weights=weights, k=1)[0]
        ep, _, edge_class, confidence = candidates[chosen_idx]
        return (ep, edge_class, confidence)

    def _find_data_flow_targets(self, endpoint_id: str) -> list[tuple[str, float, str]]:
        """Find endpoints consuming this endpoint's outputs via feeds_into edges."""
        target_info: dict[str, tuple[float, str]] = {}

        for succ in self.graph.successors(endpoint_id):
            if self.graph.nodes[succ].get("node_type") != "output_field":
                continue
            edge = self.graph.edges[endpoint_id, succ]
            if edge.get("edge_type") != "produces":
                continue

            for param_node in self.graph.successors(succ):
                if self.graph.nodes[param_node].get("node_type") != "parameter":
                    continue
                edge2 = self.graph.edges[succ, param_node]
                if edge2.get("edge_type") != "feeds_into":
                    continue

                param_ep = self.graph.nodes[param_node].get("endpoint")
                if param_ep and param_ep != endpoint_id:
                    conf = edge2.get("confidence", 0.5)
                    ec = edge2.get("edge_class", "soft")
                    if param_ep not in target_info or conf > target_info[param_ep][0]:
                        target_info[param_ep] = (conf, ec)

        return sorted(
            [(ep, conf, ec) for ep, (conf, ec) in target_info.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    def _is_independent(self, ep_a: str, ep_b: str) -> bool:
        """True if neither endpoint's outputs feed into the other (hard edges only)."""
        targets_a = {ep for ep, conf, ec in self._find_data_flow_targets(ep_a) if ec == "hard"}
        if ep_b in targets_a:
            return False
        targets_b = {ep for ep, conf, ec in self._find_data_flow_targets(ep_b) if ec == "hard"}
        if ep_a in targets_b:
            return False
        return True

    def _has_chain_potential(self, ep: str) -> bool:
        """Check if an endpoint can chain to at least one other endpoint."""
        for succ in self.graph.successors(ep):
            edge = self.graph.edges[ep, succ]
            if edge.get("edge_type") == "same_tool":
                return True

        targets = self._find_data_flow_targets(ep)
        for target, confidence, edge_class in targets:
            if edge_class == "hard":
                return True

        return False

    def _pick_endpoint(self, constraints: SamplingConstraints) -> str | None:
        """Pick a starting endpoint weighted by schema quality + chain potential."""
        need_chain = constraints.min_steps > 1

        if constraints.categories:
            candidates = []
            for cat in constraints.categories:
                candidates.extend(self._endpoints_by_category.get(cat, []))
        else:
            candidates = list(self._endpoints)

        if constraints.exclude_tools:
            candidates = [
                ep for ep in candidates
                if self.graph.nodes[ep]["tool_name"] not in constraints.exclude_tools
            ]

        if constraints.required_tools:
            required = [
                ep for ep in candidates
                if self.graph.nodes[ep]["tool_name"] in constraints.required_tools
            ]
            if required:
                candidates = required

        if not candidates:
            return None

        if need_chain:
            chainable = [ep for ep in candidates if self._has_chain_potential(ep)]
            if chainable:
                candidates = chainable

        if not candidates:
            return None

        weights = []
        must_include = set(constraints.must_include_categories or [])
        for ep in candidates:
            w = {"complete": 1.2, "inferred": 1.0, "minimal": 0.8}.get(
                self.graph.nodes[ep].get("schema_quality", "minimal"), 0.8
            )
            if need_chain and self._find_data_flow_targets(ep):
                w *= 1.5
            # 3× seed bias for must_include_categories
            if must_include:
                ep_cat = self.graph.nodes[ep].get("category", "")
                if ep_cat in must_include:
                    w *= 3.0
            weights.append(w)

        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _passes_constraints(self, ep_id: str, constraints: SamplingConstraints) -> bool:
        """Check if an endpoint passes the given constraints."""
        data = self.graph.nodes[ep_id]
        if constraints.categories and data.get("category") not in constraints.categories:
            return False
        if constraints.exclude_tools and data["tool_name"] in constraints.exclude_tools:
            return False
        return True

    def _diversify_chain(
        self,
        chain: list[str],
        edge_classes: list[tuple[str, float]],
        visited_tools: set[str],
        visited_eps: set[str],
        constraints: SamplingConstraints,
    ) -> tuple[list[str], list[tuple[str, float]]]:
        """Add endpoints from new tools to meet min_tools."""
        needed = constraints.min_tools - len(visited_tools)

        for _ in range(needed):
            candidates: list[tuple[str, float, str]] = []
            for ep in chain:
                for target, conf, ec in self._find_data_flow_targets(ep):
                    if target not in visited_eps:
                        target_tool = self.graph.nodes[target]["tool_name"]
                        if target_tool not in visited_tools and self._passes_constraints(target, constraints):
                            diversify_class = (
                                "diversify_feeds_into_hard" if ec == "hard"
                                else "diversify_feeds_into_soft"
                            )
                            candidates.append((target, conf, diversify_class))

            if not candidates:
                # Fallback: any endpoint from same categories, new tool
                chain_cats = {self.graph.nodes[ep].get("category") for ep in chain}
                for cat in chain_cats:
                    for ep in self._endpoints_by_category.get(cat, []):
                        ep_tool = self.graph.nodes[ep]["tool_name"]
                        if ep_tool not in visited_tools and ep not in visited_eps:
                            if self._passes_constraints(ep, constraints):
                                candidates.append((ep, 0.5, "diversify_fallback"))

            if not candidates:
                break

            pick_ep, pick_conf, pick_class = self.rng.choice(candidates)
            chain.append(pick_ep)
            edge_classes.append((pick_class, pick_conf))
            visited_eps.add(pick_ep)
            visited_tools.add(self.graph.nodes[pick_ep]["tool_name"])

        return chain, edge_classes

    def _get_output_field_names(self, ep_id: str) -> set[str]:
        """Get normalized output field names for an endpoint."""
        names = set()
        for succ in self.graph.successors(ep_id):
            data = self.graph.nodes[succ]
            if data.get("node_type") == "output_field":
                names.add(data.get("normalized_name", data["name"]))
        return names

    def _get_input_field_names(self, ep_id: str) -> set[str]:
        """Get normalized input parameter names for an endpoint."""
        names = set()
        for succ in self.graph.successors(ep_id):
            data = self.graph.nodes[succ]
            if data.get("node_type") == "parameter":
                names.add(data.get("normalized_name", data["name"]))
        return names

    def _ep_to_tuple(self, ep_id: str) -> tuple[str, str]:
        """Convert endpoint node ID to (tool_name, api_name) tuple."""
        data = self.graph.nodes[ep_id]
        return (data["tool_name"], data["name"])
