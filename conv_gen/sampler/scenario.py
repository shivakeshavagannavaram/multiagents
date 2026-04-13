"""Generate natural-language scenarios from sampled tool chains."""

from __future__ import annotations

import logging

import anthropic

from conv_gen.ingestor.registry import ToolRegistry

logger = logging.getLogger(__name__)

SCENARIO_PROMPT = """You are a scenario designer for a synthetic conversation dataset.

Given a sequence of API capabilities (described below), generate a realistic user scenario that would naturally require these capabilities in this order.

Capabilities in order:
{tool_descriptions}

Requirements:
- Describe what a real person wants to accomplish and WHY
- Do NOT include IDs, codes, or technical identifiers that would only be known AFTER calling a tool. Let those values be discovered through the conversation
- Include only values a real person would naturally know upfront (city names, dates, their own name, a search query)
- Each step should ask for ONE thing that the capability can deliver
- The scenario should flow naturally — each step's result motivates the next step's request
- Do NOT mention any tool names, API names, or technical endpoints
- Keep the overall task focused and achievable
{chaining_instruction}
{disambiguation_instruction}

Respond with ONLY the scenario description (2-4 sentences), nothing else."""

CHAINING_INSTRUCTION = """- IMPORTANT: Design the scenario so each step DEPENDS on the previous step's result. The user should need information from step 1's result to proceed with step 2, and step 2's result to proceed with step 3. For example: "search for X" → "use the ID from the search result to get details" → "use those details to take an action". Do NOT describe independent tasks — each step must logically require the output of the previous step."""

NO_CHAINING_INSTRUCTION = ""

DISAMBIGUATION_VARIANT = """- Deliberately leave ONE subjective preference vague (e.g. budget range, date preference, style, sort order) so the assistant has a reason to ask a clarifying question. Do NOT withhold specific identifiers or factual details — only leave a subjective choice ambiguous"""

NO_DISAMBIGUATION_VARIANT = """- Provide all the details the user would naturally have upfront — the assistant should be able to proceed without asking questions"""


class ScenarioGenerator:
    """Turns sampled tool chains into natural-language scenario descriptions."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-20250514",
        rng: __import__("random").Random | None = None,
    ):
        self.client = client
        self.model = model
        self.rng = rng or __import__("random").Random()

    def generate_scenario(
        self,
        chain: list[tuple[str, str]],
        registry: ToolRegistry,
        require_chaining: bool = False,
    ) -> str:
        """Generate a realistic scenario for the given tool chain.

        Args:
            require_chaining: If True, the scenario will describe steps that
                depend on each other's results (output→input flow).
                If False, steps can be independent.
        """
        tool_descriptions = self._format_chain(chain, registry)

        # Control disambiguation from code — roughly 40% of scenarios
        if self.rng.random() < 0.4:
            disambiguation = DISAMBIGUATION_VARIANT
        else:
            disambiguation = NO_DISAMBIGUATION_VARIANT

        chaining = CHAINING_INSTRUCTION if require_chaining else NO_CHAINING_INSTRUCTION

        prompt = SCENARIO_PROMPT.format(
            tool_descriptions=tool_descriptions,
            chaining_instruction=chaining,
            disambiguation_instruction=disambiguation,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        scenario = response.content[0].text.strip()
        logger.debug("Generated scenario for chain %s: %s", chain, scenario[:100])
        return scenario

    # Parameter names/descriptions that indicate auth or infrastructure — not user-facing
    _AUTH_KEYWORDS = {"key", "token", "auth", "secret", "password", "credential",
                      "apikey", "api_key", "app_id", "app_key", "access_token"}
    # Infrastructure params the user wouldn't naturally specify
    _INFRA_KEYWORDS = {"locale", "lang", "language", "units", "format", "order_by",
                       "sort", "sort_by", "filter_by", "callback", "page", "limit",
                       "offset", "per_page", "page_size", "include", "exclude",
                       "fields", "response_format"}

    @staticmethod
    def _is_auth_param(name: str, description: str = "") -> bool:
        """Check if a parameter is auth or infrastructure — not user-facing."""
        combined = f"{name} {description}".lower()
        if any(kw in combined for kw in ScenarioGenerator._AUTH_KEYWORDS):
            return True
        name_lower = name.lower()
        if any(kw in name_lower for kw in ScenarioGenerator._INFRA_KEYWORDS):
            return True
        return False

    @staticmethod
    def _clean_description(desc: str) -> str:
        """Clean up raw endpoint descriptions — remove technical noise."""
        if not desc:
            return ""
        # Truncate overly long descriptions
        if len(desc) > 200:
            # Cut at the last sentence boundary before 200 chars
            cut = desc[:200].rfind(".")
            if cut > 50:
                desc = desc[:cut + 1]
            else:
                desc = desc[:200] + "..."
        return desc.strip()

    @staticmethod
    def _format_chain(chain: list[tuple[str, str]], registry: ToolRegistry) -> str:
        """Format tool chain as capability descriptions without exposing tool/API names.

        Filters out auth parameters and cleans up descriptions so the
        scenario generator only sees user-facing capabilities.
        """
        lines = []
        for i, (tool_name, api_name) in enumerate(chain, 1):
            endpoint = registry.get_endpoint(tool_name, api_name)
            tool = registry.get_tool(tool_name)

            desc = ""
            if endpoint:
                raw_desc = endpoint.description or ""
                if not raw_desc and tool:
                    raw_desc = tool.tool_description
                desc = ScenarioGenerator._clean_description(raw_desc)

                # Include non-auth required params with example values
                param_parts = []
                for p in endpoint.required_parameters:
                    if ScenarioGenerator._is_auth_param(p.name, p.description):
                        continue
                    example = p.example_value or p.default
                    # Use clean param description, fall back to name
                    param_desc = p.description or p.name
                    # Truncate long param descriptions
                    if len(param_desc) > 60:
                        param_desc = p.name
                    if example:
                        param_parts.append(f"{param_desc}={example}")
                    else:
                        param_parts.append(param_desc)

                if param_parts:
                    desc += f" (accepts: {', '.join(param_parts)})"
                elif not desc:
                    desc = f"Performs an action related to {tool.tool_description[:80] if tool else 'this service'}"
            elif tool:
                desc = ScenarioGenerator._clean_description(tool.tool_description)

            lines.append(f"{i}. {desc}" if desc else f"{i}. Step {i}")

        return "\n".join(lines)
