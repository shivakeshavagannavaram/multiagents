"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from conv_gen.memory.context import ConversationContext
from conv_gen.models import Message


class BaseAgent(ABC):
    """Abstract base for all conversation agents."""

    @abstractmethod
    def run(self, context: ConversationContext, **kwargs) -> Message | list:
        """Execute the agent's action given the conversation context."""
        ...
