"""Generate natural-language scenarios from sampled tool chains."""

from __future__ import annotations

import logging

import anthropic

from saplvl.ingestor.registry import ToolRegistry

logger = logging.getLogger(__name__)

SCENARIO_PROMPT = """You are a scenario designer for a synthetic conversation dataset.

Given a sequence of API tools that will be used in a conversation, generate a realistic
user scenario that would naturally require these tools in this order.

Tools in order:
{tool_descriptions}

Requirements:
- Write a 2-3 sentence scenario describing what a user wants to accomplish
- The scenario should make it natural to use these tools in roughly this order
- Include specific details (city names, dates, preferences) to make it realistic
- Sometimes make the request slightly ambiguous so the assistant needs to ask a clarifying question
- Do NOT mention the tool names or API endpoints directly

Respond with ONLY the scenario description, nothing else."""


class ScenarioGenerator:
    """Turns sampled tool chains into natural-language scenario descriptions."""

    def __init__(self, client: anthropic.Anthropic, model: str = "claude-sonnet-4-20250514"):
        self.client = client
        self.model = model

    def generate_scenario(
        self,
        chain: list[tuple[str, str]],
        registry: ToolRegistry,
    ) -> str:
        """Generate a realistic scenario for the given tool chain."""
        tool_descriptions = self._format_chain(chain, registry)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": SCENARIO_PROMPT.format(tool_descriptions=tool_descriptions),
                }
            ],
        )

        scenario = response.content[0].text.strip()
        logger.debug("Generated scenario for chain %s: %s", chain, scenario[:100])
        return scenario

    @staticmethod
    def _format_chain(chain: list[tuple[str, str]], registry: ToolRegistry) -> str:
        """Format tool chain with descriptions for the prompt."""
        lines = []
        for i, (tool_name, api_name) in enumerate(chain, 1):
            endpoint = registry.get_endpoint(tool_name, api_name)
            tool = registry.get_tool(tool_name)

            desc = ""
            if endpoint:
                desc = endpoint.description
                params = [p.name for p in endpoint.required_parameters]
                if params:
                    desc += f" (requires: {', '.join(params)})"
            elif tool:
                desc = tool.tool_description

            lines.append(f"{i}. {tool_name}/{api_name}: {desc}")

        return "\n".join(lines)
