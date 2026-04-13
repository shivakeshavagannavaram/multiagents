"""ConversationDirector — decides what type of conversation to generate next.

Manages dataset composition to meet spec requirements:
- 50-60% multi-step (≥3 calls) + multi-tool (≥2 distinct tools)
- Mix of short and long conversations
- Diverse patterns (sequential, parallel, single)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from conv_gen.sampler.sampler import SamplingConstraints, SamplingPattern


# ------------------------------------------------------------------ #
# Conversation type definitions                                       #
# ------------------------------------------------------------------ #

@dataclass
class ConversationType:
    """Configuration for a conversation type."""
    name: str
    min_tools: int
    max_tools: int
    min_distinct_tools: int
    min_steps: int
    max_steps: int
    pattern: SamplingPattern
    require_chaining: bool
    max_turns: int
    max_messages: int


CONVERSATION_TYPES = {
    "quick_lookup": ConversationType(
        name="quick_lookup",
        min_tools=1, max_tools=1,
        min_distinct_tools=1,
        min_steps=1, max_steps=1,
        pattern=SamplingPattern.SINGLE,
        require_chaining=False,
        max_turns=2,
        max_messages=6,
    ),
    "parallel_lookup": ConversationType(
        name="parallel_lookup",
        min_tools=2, max_tools=3,
        min_distinct_tools=2,
        min_steps=2, max_steps=3,
        pattern=SamplingPattern.PARALLEL,
        require_chaining=False,
        max_turns=3,
        max_messages=10,
    ),
    "simple_sequential": ConversationType(
        name="simple_sequential",
        min_tools=2, max_tools=2,
        min_distinct_tools=2,
        min_steps=2, max_steps=2,
        pattern=SamplingPattern.SEQUENTIAL,
        require_chaining=True,
        max_turns=3,
        max_messages=10,
    ),
    "multi_step_chain": ConversationType(
        name="multi_step_chain",
        min_tools=2, max_tools=4,
        min_distinct_tools=2,
        min_steps=3, max_steps=4,
        pattern=SamplingPattern.SEQUENTIAL,
        require_chaining=True,
        max_turns=5,
        max_messages=16,
    ),
    "full_workflow": ConversationType(
        name="full_workflow",
        min_tools=3, max_tools=4,
        min_distinct_tools=3,
        min_steps=4, max_steps=4,
        pattern=SamplingPattern.SEQUENTIAL,
        require_chaining=True,
        max_turns=5,
        max_messages=16,
    ),
}

# Target composition for spec compliance — 5 categories.
# multi_step_chain + full_workflow both satisfy MS+MT (≥3 calls,
# ≥2 distinct tools), so targeting 70% of the dataset at those two
# types gives ~50-60% MS+MT after execution drop-off.
TARGET_DISTRIBUTION = {
    "quick_lookup": 0.10,
    "parallel_lookup": 0.10,
    "simple_sequential": 0.10,
    "multi_step_chain": 0.40,
    "full_workflow": 0.30,
}


class ConversationDirector:
    """Decides which conversation type to generate next.

    Tracks current dataset composition and picks the type that
    moves the dataset closest to the target distribution.
    """

    def __init__(self, rng=None):
        self.rng = rng or __import__("random").Random()
        self._counts: dict[str, int] = {t: 0 for t in CONVERSATION_TYPES}
        self._total = 0

    def next_type(self) -> ConversationType:
        """Pick the next conversation type to generate.

        Selects the type that is most under-represented relative
        to its target proportion.
        """
        if self._total == 0:
            # First conversation — pick multi_step_chain to start strong
            return CONVERSATION_TYPES["multi_step_chain"]

        # Find the type with the biggest gap between target and actual
        best_type = None
        best_gap = -1.0

        for type_name, target_pct in TARGET_DISTRIBUTION.items():
            actual_pct = self._counts[type_name] / self._total
            gap = target_pct - actual_pct
            if gap > best_gap:
                best_gap = gap
                best_type = type_name

        return CONVERSATION_TYPES[best_type]

    def build_sampler_constraints(
        self,
        conv_type: ConversationType,
        exclude_tools: list[str] | None = None,
    ) -> SamplingConstraints:
        """Build sampler constraints from the conversation type."""
        return SamplingConstraints(
            min_tools=conv_type.min_distinct_tools,
            max_tools=conv_type.max_tools,
            min_steps=conv_type.min_steps,
            max_steps=conv_type.max_steps,
            exclude_tools=exclude_tools or [],
        )

    def record(self, type_name: str) -> None:
        """Record that a conversation of this type was generated."""
        self._counts[type_name] = self._counts.get(type_name, 0) + 1
        self._total += 1

    def stats(self) -> dict:
        """Current dataset composition."""
        return {
            "total": self._total,
            "counts": dict(self._counts),
            "percentages": {
                t: f"{c / max(self._total, 1) * 100:.0f}%"
                for t, c in self._counts.items()
            },
            "multi_step_pct": f"{(self._counts.get('multi_step_chain', 0) + self._counts.get('full_workflow', 0)) / max(self._total, 1) * 100:.0f}%",
        }
