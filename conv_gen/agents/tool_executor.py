"""Tool executor agent wrapping the ToolSimulator."""

from __future__ import annotations

import logging

from conv_gen.agents.base import BaseAgent
from conv_gen.memory.context import ConversationContext
from conv_gen.models import Message, ToolCall, ToolOutput
from conv_gen.simulator.executor import SessionState, ToolSimulator

logger = logging.getLogger(__name__)


class ToolExecutorAgent(BaseAgent):
    """Executes tool calls through the ToolSimulator and returns mock results."""

    def __init__(self, simulator: ToolSimulator, session: SessionState):
        self.simulator = simulator
        self.session = session

    def run(
        self,
        context: ConversationContext,
        tool_calls: list[ToolCall] | None = None,
        scenario: str = "",
        chain_context: str = "",
        **kwargs,
    ) -> Message:
        """Execute tool calls and return a tool message with results."""
        if not tool_calls:
            return Message(role="tool", content="No tool calls to execute.")

        outputs = []
        for tc in tool_calls:
            logger.debug("Executing: %s/%s(%s)", tc.tool_name, tc.api_name, tc.arguments)
            output = self.simulator.execute(
                tc, self.session, scenario=scenario, chain_context=chain_context,
            )
            outputs.append(output)
            context.add_tool_output(output)

        return Message(
            role="tool",
            content=None,
            tool_outputs=outputs,
        )
