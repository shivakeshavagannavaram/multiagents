"""Cross-conversation diversity steering using mem0."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter

from saplvl.models import Conversation

logger = logging.getLogger(__name__)

GENERATOR_USER_ID = "conversation_generator"


class DiversitySteering:
    """Steers generation toward diverse tool/domain coverage using mem0.

    Records which tools, categories, and patterns have been generated
    and provides guidance to avoid over-represented combinations.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.memory = None
        self._usage_counts: Counter = Counter()
        self._category_counts: Counter = Counter()
        self._pattern_counts: Counter = Counter()
        self._conversation_count = 0

        if enabled:
            try:
                from mem0 import Memory
                self.memory = Memory()
                logger.info("mem0 initialized for cross-conversation steering")
            except Exception as e:
                logger.warning("Failed to initialize mem0 (%s), falling back to counter-based steering", e)

    def record_conversation(self, conversation: Conversation) -> None:
        """Record a generated conversation's tool/domain usage."""
        if not self.enabled:
            return

        tools_used = conversation.tools_used
        categories = list({
            conversation.metadata.get("categories", {}).get(t, "unknown")
            for t in tools_used
        })
        pattern = self._classify_pattern(conversation)

        # Update local counters
        for tool in tools_used:
            self._usage_counts[tool] += 1
        for cat in categories:
            self._category_counts[cat] += 1
        self._pattern_counts[pattern] += 1
        self._conversation_count += 1

        # Store in mem0 if available
        if self.memory:
            summary = (
                f"Generated conversation using tools: {tools_used}. "
                f"Categories: {categories}. Pattern: {pattern}. "
                f"Turns: {conversation.num_turns}. "
                f"Tool calls: {conversation.num_tool_calls}."
            )
            try:
                self.memory.add(
                    messages=[{"role": "system", "content": summary}],
                    user_id=GENERATOR_USER_ID,
                )
            except Exception as e:
                logger.debug("mem0 add failed: %s", e)

    def get_steering_guidance(self, candidate_tools: list[str]) -> str:
        """Get guidance for generating a conversation with the candidate tools.

        Returns a string instruction to inject into the scenario generator prompt.
        """
        if not self.enabled or self._conversation_count < 5:
            return ""

        guidance_parts = []

        # Check for over-represented tools
        overused = self._find_overrepresented_tools(candidate_tools)
        if overused:
            guidance_parts.append(
                f"Note: The tools {overused} have been heavily used. "
                "Try to create a scenario that uses them in a novel way, "
                "different from typical usage patterns."
            )

        # Check for under-represented categories
        underused_cats = self._find_underrepresented_categories()
        if underused_cats:
            guidance_parts.append(
                f"Consider incorporating aspects related to these under-explored "
                f"domains: {underused_cats[:3]}"
            )

        # mem0 semantic search for similar combinations
        if self.memory:
            try:
                results = self.memory.search(
                    query=f"conversations using tools: {candidate_tools}",
                    user_id=GENERATOR_USER_ID,
                    limit=3,
                )
                if results and results.get("results"):
                    guidance_parts.append(
                        "Similar tool combinations have been generated before. "
                        "Make this conversation distinct in its scenario and user intent."
                    )
            except Exception as e:
                logger.debug("mem0 search failed: %s", e)

        return " ".join(guidance_parts)

    def get_exclude_tools(self, top_n: int = 10) -> list[str]:
        """Get the most overused tools to potentially exclude from sampling."""
        if not self.enabled or self._conversation_count < 10:
            return []

        if not self._usage_counts:
            return []

        mean_usage = sum(self._usage_counts.values()) / len(self._usage_counts)
        threshold = mean_usage * 3  # 3x average usage

        return [
            tool for tool, count in self._usage_counts.most_common(top_n)
            if count > threshold
        ]

    def get_usage_stats(self) -> dict:
        """Return current usage statistics."""
        return {
            "total_conversations": self._conversation_count,
            "tool_usage": dict(self._usage_counts.most_common()),
            "category_usage": dict(self._category_counts.most_common()),
            "pattern_usage": dict(self._pattern_counts.most_common()),
        }

    def _find_overrepresented_tools(self, candidates: list[str]) -> list[str]:
        """Find tools in candidates that are over-represented."""
        if not self._usage_counts:
            return []

        mean = sum(self._usage_counts.values()) / max(len(self._usage_counts), 1)
        return [t for t in candidates if self._usage_counts.get(t, 0) > mean * 2]

    def _find_underrepresented_categories(self) -> list[str]:
        """Find categories with below-average representation."""
        if not self._category_counts:
            return []

        mean = sum(self._category_counts.values()) / len(self._category_counts)
        return [
            cat for cat, count in self._category_counts.items()
            if count < mean * 0.5
        ]

    @staticmethod
    def _classify_pattern(conversation: Conversation) -> str:
        """Classify the conversation's tool usage pattern."""
        n_calls = conversation.num_tool_calls
        n_tools = conversation.num_distinct_tools

        if n_calls == 1:
            return "single_call"
        elif n_tools == 1:
            return "single_tool_multi_step"
        elif n_calls <= 3:
            return "short_multi_tool"
        else:
            return "long_multi_tool"


class DiversityMetrics:
    """Compute diversity metrics for a corpus of conversations."""

    @staticmethod
    def tool_combination_entropy(conversations: list[Conversation]) -> float:
        """Shannon entropy over unique tool combinations.

        Higher values indicate more diverse tool usage.
        """
        if not conversations:
            return 0.0

        combos = Counter()
        for conv in conversations:
            key = tuple(sorted(conv.tools_used))
            combos[key] += 1

        total = sum(combos.values())
        entropy = 0.0
        for count in combos.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    @staticmethod
    def domain_coverage_uniformity(
        conversations: list[Conversation],
        all_categories: list[str] | None = None,
    ) -> float:
        """How uniformly categories are covered across conversations.

        1.0 = perfectly uniform, 0.0 = single category.
        Computed as 1 - normalized standard deviation of category counts.
        """
        if not conversations:
            return 0.0

        cat_counts = Counter()
        for conv in conversations:
            categories = conv.metadata.get("categories_list", [])
            if not categories:
                # Fallback: extract from tools_used
                categories = list({t.split("/")[0] for t in conv.tools_used})
            for cat in categories:
                cat_counts[cat] += 1

        if all_categories:
            # Include zero counts for uncovered categories
            for cat in all_categories:
                if cat not in cat_counts:
                    cat_counts[cat] = 0

        if len(cat_counts) <= 1:
            return 0.0

        counts = list(cat_counts.values())
        mean = sum(counts) / len(counts)
        if mean == 0:
            return 0.0

        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        std = math.sqrt(variance)
        normalized_std = std / mean  # Coefficient of variation

        # Clamp to [0, 1]
        return max(0.0, min(1.0, 1.0 - normalized_std))

    @staticmethod
    def unique_tool_ratio(conversations: list[Conversation]) -> float:
        """Ratio of unique tool combinations to total conversations."""
        if not conversations:
            return 0.0

        combos = {tuple(sorted(conv.tools_used)) for conv in conversations}
        return len(combos) / len(conversations)

    @staticmethod
    def summary(conversations: list[Conversation], all_categories: list[str] | None = None) -> dict:
        """Compute all diversity metrics."""
        return {
            "tool_combination_entropy": DiversityMetrics.tool_combination_entropy(conversations),
            "domain_coverage_uniformity": DiversityMetrics.domain_coverage_uniformity(
                conversations, all_categories
            ),
            "unique_tool_ratio": DiversityMetrics.unique_tool_ratio(conversations),
            "total_conversations": len(conversations),
            "unique_combinations": len({
                tuple(sorted(conv.tools_used)) for conv in conversations
            }),
        }
