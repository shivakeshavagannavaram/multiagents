"""Conversation plan — owns the completion decision for a conversation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ConversationStatus(str, Enum):
    """Conversation lifecycle state."""
    IN_PROGRESS = "in_progress"
    COMPLETING  = "completing"
    DONE        = "done"


@dataclass
class ConversationPlan:
    """Owns the stop decision for a conversation."""

    planned_tools: list[tuple[str, str]]
    max_turns:     int = 0
    max_messages:  int = 0

    def __post_init__(self):
        n = len(self.planned_tools)
        if self.max_turns == 0:
            self.max_turns = n + 2
        if self.max_messages == 0:
            self.max_messages = (n * 4) + 4
        self._hard_cap = (n * 5) + 6

    used_tools:            set[tuple[str, str]] = field(default_factory=set)
    unplanned_tools_used:  set[tuple[str, str]] = field(default_factory=set)
    clarification_pending: bool = False
    turn:                  int  = 0
    message_count:         int  = 0

    def record_tool_use(self, tool_name: str, api_name: str) -> None:
        """Record a tool call, tracking planned vs unplanned separately."""
        key = (tool_name, api_name)
        self.used_tools.add(key)
        if key not in set(self.planned_tools):
            self.unplanned_tools_used.add(key)

    def mark_clarification_asked(self) -> None:
        self.clarification_pending = True

    def mark_clarification_answered(self) -> None:
        self.clarification_pending = False

    def advance_turn(self) -> None:
        self.turn += 1

    def add_message(self) -> None:
        self.message_count += 1

    @property
    def planned_tools_completed(self) -> set[tuple[str, str]]:
        return self.used_tools & set(self.planned_tools)

    @property
    def tools_remaining(self) -> list[str]:
        remaining = set(self.planned_tools) - self.planned_tools_completed
        return [t[0] for t in remaining]

    @property
    def progress_fraction(self) -> float:
        if not self.planned_tools:
            return 1.0
        return len(self.planned_tools_completed) / len(self.planned_tools)

    @property
    def status(self) -> ConversationStatus:
        """Single source of truth for conversation state.

        Never completes while tools_remaining is non-empty — extends limits instead.
        """
        has_remaining = len(self.tools_remaining) > 0

        if self.clarification_pending:
            return ConversationStatus.IN_PROGRESS

        if self.message_count >= self._hard_cap:
            return ConversationStatus.DONE

        if has_remaining:
            if self.message_count >= self.max_messages - 2:
                self.max_messages += 4
            if self.turn >= self.max_turns - 1:
                self.max_turns += 1
            return ConversationStatus.IN_PROGRESS

        if self.planned_tools_completed >= set(self.planned_tools):
            return ConversationStatus.COMPLETING

        if self.turn >= self.max_turns - 1 and not self.used_tools:
            return ConversationStatus.COMPLETING

        return ConversationStatus.IN_PROGRESS

    def is_complete(self) -> bool:
        return self.status == ConversationStatus.DONE

    def is_completing(self) -> bool:
        return self.status in (
            ConversationStatus.COMPLETING,
            ConversationStatus.DONE,
        )

    @property
    def completion_hint(self) -> str:
        """Plain-English status line for prompt injection."""
        s = self.status

        if s == ConversationStatus.IN_PROGRESS:
            pct = int(self.progress_fraction * 100)
            remaining = self.tools_remaining
            return (
                f"Task {pct}% complete. "
                f"Tools not yet used: {remaining}. "
                f"Continue the conversation."
            )

        elif s == ConversationStatus.COMPLETING:
            if self.tools_remaining:
                return (
                    f"Almost done. "
                    f"These tools were not called: {self.tools_remaining}. "
                    f"Wrap up naturally and summarize what was accomplished."
                )
            return (
                "All planned tools have been called. "
                "Wrap up the conversation naturally. "
                "Summarize results and key values."
            )

        return "The conversation is complete."

    @property
    def assistant_instruction(self) -> str:
        completed = [f"{t}/{a}" for t, a in self.planned_tools_completed]

        if self.status == ConversationStatus.IN_PROGRESS:
            if self.tools_remaining:
                return (
                    f"Tools completed: {completed}. "
                    f"Tools still available: {self.tools_remaining}. "
                    f"Use these tools when the user's request calls for them. "
                    f"Do NOT call tools the user hasn't asked for."
                )
            return "Continue the conversation."

        elif self.status == ConversationStatus.COMPLETING:
            return (
                "All tools have been used. "
                "Give a brief final summary with key values. "
                "Do NOT ask questions or offer more help."
            )

        return ""

    @property
    def user_instruction(self) -> str:
        if self.status == ConversationStatus.IN_PROGRESS:
            if self.clarification_pending:
                return "Answer the assistant's question naturally."
            if self.tools_remaining:
                return (
                    "The assistant completed one step. Based on the scenario, "
                    "ask for the next thing you need — use specific values "
                    "from the results you just received."
                )
            return "React naturally to what the assistant said."

        elif self.status == ConversationStatus.COMPLETING:
            return (
                "The assistant has finished helping you. "
                "Acknowledge the results naturally and end the conversation. "
                "Do NOT ask follow-up questions."
            )

        return ""

    def summary(self) -> dict:
        """State snapshot for logging."""
        return {
            "status":                self.status.value,
            "turn":                  self.turn,
            "message_count":         self.message_count,
            "progress":              round(self.progress_fraction, 2),
            "planned_tools":         [f"{t}/{a}" for t, a in self.planned_tools],
            "completed_tools":       [f"{t}/{a}" for t, a in self.planned_tools_completed],
            "tools_remaining":       self.tools_remaining,
            "unplanned_tools":       [f"{t}/{a}" for t, a in self.unplanned_tools_used],
            "clarification_pending": self.clarification_pending,
        }
