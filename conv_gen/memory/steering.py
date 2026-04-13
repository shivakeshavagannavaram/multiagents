"""Cross-conversation diversity steering using mem0."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter

from conv_gen.models import Conversation

logger = logging.getLogger(__name__)

GENERATOR_USER_ID = "conversation_generator"


class DiversitySteering:
    """Cross-conversation diversity steering via counters + mem0."""

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
        """Update counters and push a first-person summary to mem0."""
        if not self.enabled:
            return

        tools_used = conversation.tools_used
        categories = list({
            conversation.metadata.get("categories", {}).get(t, "unknown")
            for t in tools_used
        })
        pattern = self._classify_pattern(conversation)

        for tool in tools_used:
            self._usage_counts[tool] += 1
        for cat in categories:
            self._category_counts[cat] += 1
        self._pattern_counts[pattern] += 1
        self._conversation_count += 1

        if self.memory:
            tools_str = ", ".join(t.split("/")[0] for t in tools_used)
            cats_str = ", ".join(categories)
            summary = (
                f"I generated a {pattern} conversation in the {cats_str} category "
                f"using {tools_str}. It had {conversation.num_turns} turns "
                f"and {conversation.num_tool_calls} tool calls."
            )
            try:
                self.memory.add(summary, user_id=GENERATOR_USER_ID)
            except Exception as e:
                logger.debug("mem0 add failed: %s", e)

    def get_steering_guidance(self, candidate_tools: list[str]) -> str:
        """Prose hint for the scenario generator based on counters + mem0 search."""
        if not self.enabled or self._conversation_count < 5:
            return ""

        guidance_parts = []

        overused = self._find_overrepresented_tools(candidate_tools)
        if overused:
            guidance_parts.append(
                f"Note: The tools {overused} have been heavily used. "
                "Try to create a scenario that uses them in a novel way, "
                "different from typical usage patterns."
            )

        underused_cats = self._find_underrepresented_categories()
        if underused_cats:
            guidance_parts.append(
                f"Consider incorporating aspects related to these under-explored "
                f"domains: {underused_cats[:3]}"
            )

        if self.memory:
            try:
                tools_query = ", ".join(t.split("/")[0] for t in candidate_tools)
                results = self.memory.search(
                    query=f"I used {tools_query}",
                    user_id=GENERATOR_USER_ID,
                    limit=5,
                )
                if results and results.get("results"):
                    past = [r["memory"] for r in results["results"] if r.get("score", 0) > 0.2]
                    if past:
                        past_str = "; ".join(past[:3])
                        guidance_parts.append(
                            f"Similar tool combinations have been used before: {past_str}. "
                            "Create a scenario that is distinct from these past conversations — "
                            "use a different user intent, domain angle, or interaction style."
                        )
            except Exception as e:
                logger.debug("mem0 search failed: %s", e)

        return " ".join(guidance_parts)

    def get_exclude_tools(self, top_n: int = 10) -> list[str]:
        """Tools above 3× mean usage — the sampler's hard exclude list."""
        if not self.enabled or self._conversation_count < 10:
            return []

        if not self._usage_counts:
            return []

        mean_usage = sum(self._usage_counts.values()) / len(self._usage_counts)
        threshold = mean_usage * 3

        return [
            tool for tool, count in self._usage_counts.most_common(top_n)
            if count > threshold
        ]

    def get_usage_stats(self) -> dict:
        return {
            "total_conversations": self._conversation_count,
            "tool_usage": dict(self._usage_counts.most_common()),
            "category_usage": dict(self._category_counts.most_common()),
            "pattern_usage": dict(self._pattern_counts.most_common()),
        }

    def _find_overrepresented_tools(self, candidates: list[str]) -> list[str]:
        if not self._usage_counts:
            return []

        mean = sum(self._usage_counts.values()) / max(len(self._usage_counts), 1)
        return [t for t in candidates if self._usage_counts.get(t, 0) > mean * 2]

    def _find_underrepresented_categories(self) -> list[str]:
        if not self._category_counts:
            return []

        mean = sum(self._category_counts.values()) / len(self._category_counts)
        return [
            cat for cat, count in self._category_counts.items()
            if count < mean * 0.5
        ]

    @staticmethod
    def _classify_pattern(conversation: Conversation) -> str:
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
    """Diversity metrics for a corpus of conversations."""

    @staticmethod
    def tool_usage_entropy(conversations: list[Conversation]) -> float:
        """Normalized Shannon entropy over tool-call frequencies. 1.0 = flat, 0.0 = monopoly."""
        if not conversations:
            return 0.0

        tool_counts: Counter[str] = Counter()
        for conv in conversations:
            for tc in conv.tool_calls or []:
                key = f"{tc.tool_name}/{tc.api_name}"
                tool_counts[key] += 1

        if len(tool_counts) <= 1:
            return 0.0

        total = sum(tool_counts.values())
        entropy = 0.0
        for count in tool_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(tool_counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @staticmethod
    def unique_tools_used(conversations: list[Conversation]) -> int:
        """Distinct (tool, api) pairs across all calls."""
        seen: set[tuple[str, str]] = set()
        for conv in conversations:
            for tc in conv.tool_calls or []:
                seen.add((tc.tool_name, tc.api_name))
        return len(seen)

    @staticmethod
    def unique_tool_coverage_ratio(
        conversations: list[Conversation],
        registry_size: int,
    ) -> float:
        """unique_tools_used / registry_size."""
        if not conversations or registry_size <= 0:
            return 0.0
        return DiversityMetrics.unique_tools_used(conversations) / registry_size

    @staticmethod
    def top_n_tool_concentration(
        conversations: list[Conversation], n: int = 5
    ) -> float:
        """Fraction of calls going to the top-N tools. Lower = more spread out."""
        if not conversations:
            return 0.0

        tool_counts: Counter[str] = Counter()
        for conv in conversations:
            for tc in conv.tool_calls or []:
                key = f"{tc.tool_name}/{tc.api_name}"
                tool_counts[key] += 1

        total = sum(tool_counts.values())
        if total == 0:
            return 0.0

        top_n = sum(count for _, count in tool_counts.most_common(n))
        return top_n / total

    @staticmethod
    def tool_combination_entropy(conversations: list[Conversation]) -> float:
        """Shannon entropy over unique chain signatures (sorted tool sets)."""
        if not conversations:
            return 0.0

        combos: Counter = Counter()
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
        """Normalized Shannon entropy over category usage. 1.0 = uniform."""
        if not conversations:
            return 0.0

        cat_counts: Counter = Counter()
        for conv in conversations:
            categories = conv.metadata.get("categories_list", [])
            if not categories:
                categories = list({t.split("/")[0] for t in conv.tools_used})
            for cat in categories:
                cat_counts[cat] += 1

        if not cat_counts:
            return 0.0

        total = sum(cat_counts.values())
        entropy = 0.0
        for count in cat_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        if all_categories:
            max_cats = max(1, len(all_categories))
        else:
            max_cats = max(1, len(cat_counts))

        if max_cats <= 1:
            return 0.0
        max_entropy = math.log2(max_cats)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @staticmethod
    def unique_chain_ratio(conversations: list[Conversation]) -> float:
        """Unique chain signatures / conversation count. 1.0 = every chain distinct."""
        if not conversations:
            return 0.0
        combos = {tuple(sorted(conv.tools_used)) for conv in conversations}
        return len(combos) / len(conversations)

    @staticmethod
    def summary(
        conversations: list[Conversation],
        registry_size: int | None = None,
        all_categories: list[str] | None = None,
    ) -> dict:
        """All diversity metrics grouped as {primary, secondary, totals}."""
        total_tool_calls = sum(
            len(conv.tool_calls or []) for conv in conversations
        )

        primary = {
            "tool_usage_entropy": DiversityMetrics.tool_usage_entropy(conversations),
            "unique_tools_used": DiversityMetrics.unique_tools_used(conversations),
        }
        if registry_size:
            primary["unique_tool_coverage_ratio"] = (
                DiversityMetrics.unique_tool_coverage_ratio(conversations, registry_size)
            )

        secondary = {
            "top_5_tool_concentration": DiversityMetrics.top_n_tool_concentration(conversations, 5),
            "tool_combination_entropy": DiversityMetrics.tool_combination_entropy(conversations),
            "domain_coverage_uniformity": DiversityMetrics.domain_coverage_uniformity(
                conversations, all_categories
            ),
            "unique_chain_ratio": DiversityMetrics.unique_chain_ratio(conversations),
            "unique_chain_combinations": len({
                tuple(sorted(conv.tools_used)) for conv in conversations
            }),
        }

        totals = {
            "total_conversations": len(conversations),
            "total_tool_calls": total_tool_calls,
            "registry_size": registry_size,
        }

        return {
            "primary": primary,
            "secondary": secondary,
            "totals": totals,
        }


class QualityMetrics:
    """LLM-judge and spec-compliance metrics."""

    @staticmethod
    def mean_scores(conversations: list[Conversation]) -> dict:
        """Mean judge scores across scored conversations."""
        scored = [c for c in conversations if c.judge_scores]
        if not scored:
            return {
                "naturalness": 0.0,
                "tool_correctness": 0.0,
                "task_completion": 0.0,
                "overall_mean": 0.0,
                "scored_count": 0,
            }
        nat = sum(c.judge_scores.naturalness for c in scored) / len(scored)
        tool = sum(c.judge_scores.tool_correctness for c in scored) / len(scored)
        task = sum(c.judge_scores.task_completion for c in scored) / len(scored)
        return {
            "naturalness": nat,
            "tool_correctness": tool,
            "task_completion": task,
            "overall_mean": (nat + tool + task) / 3,
            "scored_count": len(scored),
        }

    @staticmethod
    def ms_mt_rate(conversations: list[Conversation]) -> float:
        """Fraction with ≥3 tool calls AND ≥2 distinct tools."""
        if not conversations:
            return 0.0
        hits = 0
        for conv in conversations:
            tcs = conv.tool_calls or []
            distinct = {(tc.tool_name, tc.api_name) for tc in tcs}
            if len(tcs) >= 3 and len(distinct) >= 2:
                hits += 1
        return hits / len(conversations)

    @staticmethod
    def real_chaining_rate(conversations: list[Conversation]) -> float:
        """Fraction where a later call's argument value came from an earlier response."""
        if not conversations:
            return 0.0

        def walk(d, vals):
            if isinstance(d, dict):
                for v in d.values():
                    walk(v, vals)
            elif isinstance(d, list):
                for v in d:
                    walk(v, vals)
            elif isinstance(d, str) and d:
                vals.add(d)
            elif isinstance(d, (int, float)):
                vals.add(str(d))

        hits = 0
        for conv in conversations:
            tcs = conv.tool_calls or []
            outs = conv.tool_outputs or []
            seen: set = set()
            chained = False
            for idx, tc in enumerate(tcs):
                for v in (tc.arguments or {}).values():
                    if isinstance(v, str) and v in seen and len(v) >= 3:
                        chained = True
                        break
                    if isinstance(v, (int, float)) and str(v) in seen:
                        chained = True
                        break
                if chained:
                    break
                if idx < len(outs):
                    walk(outs[idx].response, seen)
            if chained:
                hits += 1
        return hits / len(conversations)

    @staticmethod
    def summary(conversations: list[Conversation]) -> dict:
        """Combined quality and spec-compliance block."""
        scores = QualityMetrics.mean_scores(conversations)
        return {
            "quality": scores,
            "spec": {
                "ms_mt_rate": QualityMetrics.ms_mt_rate(conversations),
                "real_chaining_rate": QualityMetrics.real_chaining_rate(conversations),
            },
        }
