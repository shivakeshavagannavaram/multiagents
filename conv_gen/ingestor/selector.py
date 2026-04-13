"""Select a high-quality subset of tools from ToolBench for chain potential."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher

from conv_gen.models import Tool

logger = logging.getLogger(__name__)

GENERIC_FIELDS = frozenset({
    "id", "status", "message", "results", "data", "page", "limit",
    "offset", "error", "code", "url", "text", "value", "key",
    "format", "title", "label", "description", "name", "type",
    "result", "success", "total", "total_count", "count",
    "errors", "response", "output",
})

JUNK_NAME_PATTERNS = [
    r"^test",                # test, test2, Test_v2, Testing_v3
    r"demo",                 # demo, pe-demo, Demo Project
    r"petstore",             # petstore variants
    r"onboarding",           # onboarding project
    r"^sample",              # sample APIs
    r"^my\s*(api|store)",    # My API 12345, My Store
    r"^a{3,}$",              # aaaa
    r"^[a-z]{1,4}$",         # apfd, asd, pe, 13
    r"^\d+$",                # just numbers like "13"
    r"asdf",                 # asdfadsf
    r"should\s*be\s*free",   # ThisshouldbeFREE
    r"hard\s*limit",         # FreePlanwithHardLimit
    r"rate\s*limit",         # PetstoreRateLimit
    r"blitz$",               # petstore blitz
    r"testing\s*inbox",      # PublicAPITestingInbox
    r"^erictestpet",
    r"^platformbil",
    r"^urltest",
    r"^team\s+petstore",
    r"^teste$",
    r"^flow\s+study",
    r"^colegiosantaana",
    r"multipleteams",
    r"privatepublic",
    r"freeplanhardlimit",
    r"^pe-",
]

JUNK_COMPILED = [re.compile(p, re.IGNORECASE) for p in JUNK_NAME_PATTERNS]


def _is_junk_tool(tool: Tool) -> bool:
    """True for test/demo/placeholder tools that should be excluded."""
    name = tool.tool_name.strip()

    if name and not name[0].isascii():
        return True
    if name.startswith("\U0001f44b"):
        return True

    for pattern in JUNK_COMPILED:
        if pattern.search(name):
            return True

    if len(name) <= 3 and (not tool.tool_description or len(tool.tool_description) < 20):
        return True

    desc = (tool.tool_description or "").strip().lower()
    if desc in ("test api", "this is the description", "test", "testing", ""):
        if len(name) < 15 or not any(c.isupper() for c in name[1:]):
            return True

    if tool.api_list and len(tool.api_list) >= 3:
        ep_names = [ep.name for ep in tool.api_list]
        if len(set(ep_names)) == 1:
            return True

    return False


def _standardize_for_dedup(name: str) -> str:
    """Normalize a tool name for near-duplicate comparison."""
    s = name.strip().lower()
    s = re.sub(r"[_\s]*v\d+$", "", s)
    s = re.sub(r"\s*\(v\d+\)", "", s)
    s = re.sub(r"[_\s]+\d+$", "", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _deduplicate_tools(tools: list[tuple[float, Tool]]) -> list[tuple[float, Tool]]:
    """Collapse near-duplicates, keeping the highest-scored variant."""
    groups: dict[str, list[tuple[float, Tool]]] = defaultdict(list)

    for score, tool in tools:
        std_name = _standardize_for_dedup(tool.tool_name)
        matched = False
        for existing_key in list(groups.keys()):
            if std_name == existing_key:
                groups[existing_key].append((score, tool))
                matched = True
                break
            if SequenceMatcher(None, std_name, existing_key).ratio() > 0.85:
                groups[existing_key].append((score, tool))
                matched = True
                break
        if not matched:
            groups[std_name].append((score, tool))

    deduped = []
    removed = 0
    for key, group in groups.items():
        group.sort(key=lambda x: x[0], reverse=True)
        deduped.append(group[0])
        if len(group) > 1:
            removed += len(group) - 1

    if removed:
        logger.info("Deduplication removed %d duplicate tools", removed)

    return deduped


def _trim_endpoints(tool: Tool, max_endpoints: int = 35) -> Tool:
    """Keep the top-quality endpoints when a tool has too many."""
    if len(tool.api_list) <= max_endpoints:
        return tool

    def ep_quality(ep):
        score = 0
        schema = ep.response_schema
        if schema and isinstance(schema, dict):
            props = schema.get("properties", {})
            if isinstance(props, dict):
                score += len(props) * 2
        score += len(ep.required_parameters) * 3
        score += len(ep.optional_parameters)
        if ep.description and len(ep.description) > 10:
            score += 2
        return score

    ranked = sorted(tool.api_list, key=ep_quality, reverse=True)
    return Tool(
        tool_name=tool.tool_name,
        standardized_name=tool.standardized_name,
        tool_description=tool.tool_description,
        category=tool.category,
        api_list=ranked[:max_endpoints],
    )


def _get_real_output_fields(tool: Tool) -> set[str]:
    """Domain-specific output fields — not generic, not just echoed inputs."""
    outputs = set()
    for ep in tool.api_list:
        input_names = {p.name.lower() for p in ep.all_parameters}
        if ep.response_schema and isinstance(ep.response_schema, dict):
            props = ep.response_schema.get("properties", {})
            if isinstance(props, dict):
                for k in props:
                    kl = k.lower()
                    if kl not in input_names and kl not in GENERIC_FIELDS:
                        outputs.add(kl)
    return outputs


def _passes_factor1(tool: Tool, min_fields: int = 2) -> bool:
    """Factor 1: at least min_fields domain-specific output fields."""
    return len(_get_real_output_fields(tool)) >= min_fields


def _has_search_action_pair(tool: Tool) -> bool:
    """Factor 2: tool has both a search-like and an action-like endpoint."""
    has_search = False
    has_action = False
    for ep in tool.api_list:
        name = ep.name.lower()
        if any(w in name for w in ("search", "find", "list", "get_all", "query", "browse")):
            has_search = True
        if any(w in name for w in ("book", "create", "register", "order",
                                    "add", "reserve", "update", "delete")):
            has_action = True
    return has_search and has_action


def _get_input_fields(tool: Tool) -> set[str]:
    """Non-generic input parameter names."""
    inputs = set()
    for ep in tool.api_list:
        for p in ep.all_parameters:
            if p.name.lower() not in GENERIC_FIELDS:
                inputs.add(p.name.lower())
    return inputs


def _compute_cross_tool_score(
    tool: Tool,
    all_inputs_by_cat: dict[str, set[str]],
) -> float:
    """Factor 3: how many other tools could consume this tool's outputs (capped at 30)."""
    outs = _get_real_output_fields(tool)
    if not outs:
        return 0.0

    score = 0
    for cat, inputs in all_inputs_by_cat.items():
        shared = outs & inputs
        score += len(shared)

    return min(score, 30)


def _normalize_field(name: str) -> str:
    """camelCase → snake_case."""
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return s.lower().strip('_')


def _build_cross_category_index(
    all_tools: list[Tool],
) -> dict[str, set[str]]:
    """normalized_field_name → set of categories containing it."""
    field_categories: dict[str, set[str]] = defaultdict(set)

    for tool in all_tools:
        cat = tool.category or "uncategorized"
        for ep in tool.api_list:
            for param in ep.all_parameters:
                norm = _normalize_field(param.name)
                if norm not in GENERIC_FIELDS and len(norm) > 2:
                    field_categories[norm].add(cat)

            if ep.response_schema and isinstance(ep.response_schema, dict):
                for field_name in _walk_schema_fields(ep.response_schema):
                    norm = _normalize_field(field_name)
                    if norm not in GENERIC_FIELDS and len(norm) > 2:
                        field_categories[norm].add(cat)

    return {
        field: cats for field, cats in field_categories.items()
        if len(cats) >= 2
    }


def _walk_schema_fields(schema: dict, depth: int = 0) -> list[str]:
    """Recursively collect field names from a response schema."""
    if depth > 3 or not isinstance(schema, dict):
        return []

    fields = []
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for key, prop in properties.items():
            fields.append(key)
            if isinstance(prop, dict):
                fields.extend(_walk_schema_fields(prop, depth + 1))
                items = prop.get("items", {})
                if isinstance(items, dict):
                    fields.extend(_walk_schema_fields(items, depth + 1))
    return fields


def _compute_cross_category_bonus(
    tool: Tool,
    bridge_fields: dict[str, set[str]],
) -> float:
    """Score how well a tool bridges to other categories."""
    if not bridge_fields:
        return 0.0

    tool_cat = tool.category or "uncategorized"
    bridged_categories: set[str] = set()
    bridge_field_count = 0

    for ep in tool.api_list:
        for param in ep.all_parameters:
            norm = _normalize_field(param.name)
            if norm in bridge_fields:
                other_cats = bridge_fields[norm] - {tool_cat}
                if other_cats:
                    bridged_categories.update(other_cats)
                    bridge_field_count += 1

        if ep.response_schema and isinstance(ep.response_schema, dict):
            for field_name in _walk_schema_fields(ep.response_schema):
                norm = _normalize_field(field_name)
                if norm in bridge_fields:
                    other_cats = bridge_fields[norm] - {tool_cat}
                    if other_cats:
                        bridged_categories.update(other_cats)
                        bridge_field_count += 1

    field_score = min(bridge_field_count, 15) * 1.0
    breadth_score = min(len(bridged_categories), 5) * 3.0
    return field_score + breadth_score


def _is_inferred_schema(properties) -> bool:
    """True if this looks like the parser's synthesized schema, not a real one."""
    if not isinstance(properties, dict):
        return True
    keys = set(properties.keys())
    generic = {"id", "status", "message", "created_at", "results", "total_count"}
    if keys and keys.issubset(generic | {"type", "properties"}):
        return True
    return False


def score_tool(
    tool: Tool,
    bridge_fields: dict[str, set[str]] | None = None,
    all_inputs_by_cat: dict[str, set[str]] | None = None,
) -> float:
    """Composite quality + chainability + cross-category score."""
    score = 0.0

    for ep in tool.api_list:
        schema = ep.response_schema
        if schema and isinstance(schema, dict) and "properties" in schema:
            props = schema["properties"]
            if not _is_inferred_schema(props):
                score += 5.0
            else:
                score += 1.0

        score += len(ep.required_parameters) * 0.5
        score += len(ep.optional_parameters) * 0.3

        if ep.description and len(ep.description) > 10:
            score += 0.5

    ep_count = min(len(tool.api_list), 20)
    if ep_count >= 2:
        score += ep_count * 1.5

    if tool.tool_description and len(tool.tool_description) > 20:
        score += 2.0

    real_outputs = len(_get_real_output_fields(tool))
    score += min(real_outputs, 10) * 2.0

    if _has_search_action_pair(tool):
        score += 20.0

    if all_inputs_by_cat:
        score += _compute_cross_tool_score(tool, all_inputs_by_cat) * 2.0

    if bridge_fields:
        score += _compute_cross_category_bonus(tool, bridge_fields)

    return score


def select_tools(
    all_tools: list[Tool],
    target_count: int = 500,
    min_per_category: int = 3,
    max_per_category: int | None = None,
    max_endpoints_per_tool: int = 35,
) -> list[Tool]:
    """Pick a balanced, chain-friendly subset of the full ToolBench pool."""
    clean_tools = []
    junk_count = 0
    for tool in all_tools:
        if _is_junk_tool(tool):
            junk_count += 1
        else:
            clean_tools.append(tool)

    if junk_count:
        logger.info("Filtered %d junk/test tools from %d total", junk_count, len(all_tools))

    domain_tools = [t for t in clean_tools if _passes_factor1(t)]
    logger.info(
        "Factor 1: %d tools have domain-specific outputs (from %d clean)",
        len(domain_tools), len(clean_tools),
    )

    logger.info("Building cross-category field index...")
    bridge_fields = _build_cross_category_index(domain_tools)
    logger.info("Found %d bridge fields spanning 2+ categories", len(bridge_fields))

    all_inputs_by_cat: dict[str, set[str]] = defaultdict(set)
    for t in domain_tools:
        for field in _get_input_fields(t):
            all_inputs_by_cat[t.category].add(field)

    by_category: dict[str, list[Tool]] = defaultdict(list)
    for tool in domain_tools:
        cat = tool.category or "uncategorized"
        by_category[cat].append(tool)

    num_categories = len(by_category)
    if max_per_category is None:
        max_per_category = max(target_count // num_categories * 2, min_per_category + 2)

    scored: dict[str, list[tuple[float, Tool]]] = {}
    for cat, tools in by_category.items():
        scored_list = [
            (score_tool(t, bridge_fields, all_inputs_by_cat), t)
            for t in tools
        ]
        scored[cat] = _deduplicate_tools(scored_list)
        scored[cat].sort(key=lambda x: x[0], reverse=True)

    selected: list[Tool] = []
    selected_names: set[str] = set()
    cat_counts: dict[str, int] = defaultdict(int)

    for cat in sorted(scored.keys()):
        for score, tool in scored[cat]:
            if cat_counts[cat] >= min_per_category:
                break
            if tool.tool_name not in selected_names:
                selected.append(tool)
                selected_names.add(tool.tool_name)
                cat_counts[cat] += 1

    remaining = target_count - len(selected)
    if remaining > 0:
        pool = []
        for cat, items in scored.items():
            for score, tool in items:
                if tool.tool_name not in selected_names:
                    pool.append((score, cat, tool))
        pool.sort(key=lambda x: x[0], reverse=True)

        for score, cat, tool in pool:
            if len(selected) >= target_count:
                break
            if cat_counts[cat] >= max_per_category:
                continue
            selected.append(tool)
            selected_names.add(tool.tool_name)
            cat_counts[cat] += 1

    bridge_categories = {"Weather", "Location", "Mapping", "Translation", "Finance", "Financial"}
    for bcat in bridge_categories:
        if cat_counts.get(bcat, 0) < 2:
            logger.warning("Bridge category '%s' has only %d tools", bcat, cat_counts.get(bcat, 0))

    selected = [_trim_endpoints(t, max_endpoints_per_tool) for t in selected]

    total_eps = sum(len(t.api_list) for t in selected)
    search_action = sum(1 for t in selected if _has_search_action_pair(t))
    logger.info(
        "Selected %d tools (%d endpoints) across %d categories",
        len(selected), total_eps, len(cat_counts),
    )
    logger.info("  Search+Action workflow tools: %d", search_action)

    return selected
