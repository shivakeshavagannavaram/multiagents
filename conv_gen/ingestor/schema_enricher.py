"""Enrich response schemas using LLM inference.

ToolBench response schemas are often generic (id, status, message),
echoed from inputs, or missing entirely. This module uses an LLM to
generate domain-specific response schemas based on the endpoint's
description and input parameters.

Only endpoints with poor schemas are enriched — those that already
have domain-specific output fields are left unchanged.
"""

from __future__ import annotations

import json
import logging

import openai

from conv_gen.models import Tool
from conv_gen.ingestor.selector import GENERIC_FIELDS

logger = logging.getLogger(__name__)

# Infrastructure fields that look non-generic but aren't domain-specific
_INFRA_FIELDS = frozenset({
    "get", "parameters", "errors", "response", "object",
    "created", "usage", "choices", "model", "version",
    "haserror", "statuscode", "statusmessage", "responsecode",
    "responsemessage", "paymentmethods", "item_name", "item_id",
    "created_at", "updated_at", "callback", "next", "previous",
    "per_page", "page_size", "sort", "order", "order_by",
})


def _needs_enrichment(ep) -> bool:
    """Check if an endpoint's response schema needs LLM enrichment."""
    if not ep.response_schema or not isinstance(ep.response_schema, dict):
        return True

    props = ep.response_schema.get("properties", {})
    if not isinstance(props, dict) or len(props) == 0:
        return True

    input_names = {p.name.lower() for p in ep.all_parameters}
    all_fields = {k.lower() for k in props}

    # Echoed: >50% of output fields are input params
    echoed = all_fields & input_names
    if len(echoed) / max(len(all_fields), 1) > 0.5:
        return True

    # Only generic/infra fields remain after filtering
    remaining = all_fields - input_names - GENERIC_FIELDS
    domain_fields = remaining - _INFRA_FIELDS
    if len(domain_fields) < 2:
        return True

    return False


def _generate_schema(
    client: openai.OpenAI,
    tool_name: str,
    endpoint_name: str,
    description: str,
    params: list[str],
    model: str = "gpt-4.1-nano",
) -> dict | None:
    """Generate a domain-specific response schema using LLM."""
    params_str = ", ".join(params[:8]) if params else "none"

    prompt = f"""Generate a response schema for this API.

API: {tool_name} / {endpoint_name}
Description: {description}
Inputs: {params_str}

Return JSON with "properties" containing domain-specific fields this API would return.
Keep it concise — 5-8 fields max. Use types: string, number, integer, boolean, array.
Do NOT include generic fields like id, status, message, data, results.
Include fields specific to what this API does.
Return ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=300,
        )

        schema = json.loads(response.choices[0].message.content)

        # Validate it has properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            # Wrap in standard schema format
            return {"type": "object", "properties": schema["properties"]}

        return None

    except Exception as e:
        logger.debug("Schema generation failed for %s/%s: %s", tool_name, endpoint_name, e)
        return None


def enrich_schemas(
    tools: list[Tool],
    client: openai.OpenAI,
    model: str = "gpt-4.1-nano",
    max_concurrent: int = 50,
) -> list[Tool]:
    """Enrich response schemas for endpoints that need it.

    Endpoints with existing domain-specific schemas are left unchanged.
    Only endpoints with generic, echoed, or missing schemas get LLM-generated ones.

    Uses concurrent threads for speed (~50 parallel requests).

    Returns the tools list with enriched schemas.
    """
    import concurrent.futures

    # Collect endpoints that need enrichment
    tasks = []
    skipped = 0

    for tool in tools:
        for ep in tool.api_list:
            if not _needs_enrichment(ep):
                skipped += 1
                continue

            params = [f"{p.name} ({p.type})" for p in ep.all_parameters]
            desc = ep.description or tool.tool_description or f"{tool.tool_name} {ep.name}"

            tasks.append({
                "ep": ep,
                "tool_name": tool.tool_name,
                "ep_name": ep.name,
                "desc": desc,
                "params": params,
            })

    logger.info(
        "Schema enrichment: %d endpoints to enrich, %d already good",
        len(tasks), skipped,
    )

    if not tasks:
        return tools

    enriched = 0
    failed = 0

    def process(task):
        return _generate_schema(
            client, task["tool_name"], task["ep_name"],
            task["desc"], task["params"], model,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(process, t): t for t in tasks}

        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                schema = future.result()
                if schema:
                    task["ep"].response_schema = schema
                    enriched += 1
                else:
                    failed += 1
            except Exception as e:
                logger.debug("Enrichment failed for %s/%s: %s", task["tool_name"], task["ep_name"], e)
                failed += 1

            done = enriched + failed
            if done % 500 == 0:
                logger.info("Schema enrichment progress: %d/%d done", done, len(tasks))

    logger.info(
        "Schema enrichment complete: %d enriched, %d already good, %d failed (out of %d)",
        enriched, skipped, failed, enriched + skipped + failed,
    )

    return tools
