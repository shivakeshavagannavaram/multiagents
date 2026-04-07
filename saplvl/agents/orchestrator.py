"""Orchestrates multi-agent conversation generation."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from saplvl.agents.assistant import AssistantAgent
from saplvl.agents.tool_executor import ToolExecutorAgent
from saplvl.agents.user_simulator import UserSimulatorAgent
from saplvl.memory.context import ConversationContext
from saplvl.memory.steering import DiversitySteering
from saplvl.models import Conversation, JudgeScore, Message

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """Orchestrates the full conversation generation loop.

    Flow:
    1. UserSimulator generates initial user message
    2. Assistant responds (possibly with tool calls)
    3. If tool calls: ToolExecutor runs them, results fed back
    4. Assistant generates response using tool results
    5. Loop continues until all tools used or max turns reached
    6. Judge scores the conversation
    7. If score < threshold: retry with adjusted prompts
    8. Record in steering memory
    """

    def __init__(
        self,
        user_agent: UserSimulatorAgent,
        assistant_agent: AssistantAgent,
        tool_executor: ToolExecutorAgent,
        judge: Any | None = None,  # JudgeLLM, imported lazily to avoid circular
        steering: DiversitySteering | None = None,
        max_turns: int = 10,
        max_retries: int = 2,
        quality_threshold: float = 3.0,
    ):
        self.user_agent = user_agent
        self.assistant_agent = assistant_agent
        self.tool_executor = tool_executor
        self.judge = judge
        self.steering = steering
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.quality_threshold = quality_threshold

    def generate_conversation(
        self,
        chain: list[tuple[str, str]],
        scenario: str,
        metadata: dict[str, Any] | None = None,
    ) -> Conversation:
        """Generate a complete conversation for the given tool chain and scenario."""
        conversation = None
        best_conversation = None
        best_score = 0.0

        for attempt in range(self.max_retries + 1):
            repair_hints = ""
            if attempt > 0 and conversation and conversation.judge_scores:
                repair_hints = self._build_repair_hints(conversation.judge_scores)
                logger.info("Retry %d with hints: %s", attempt, repair_hints[:100])

            conversation = self._run_conversation(
                chain, scenario, repair_hints, metadata or {}
            )

            # Score with judge
            if self.judge:
                try:
                    scores = self.judge.score(conversation)
                    conversation.judge_scores = scores
                    mean = scores.mean_score

                    if mean > best_score:
                        best_score = mean
                        best_conversation = conversation

                    if mean >= self.quality_threshold:
                        logger.info(
                            "Conversation %s passed quality check (%.2f >= %.2f)",
                            conversation.conversation_id, mean, self.quality_threshold,
                        )
                        break

                    logger.info(
                        "Conversation %s below threshold (%.2f < %.2f), attempt %d/%d",
                        conversation.conversation_id, mean, self.quality_threshold,
                        attempt + 1, self.max_retries + 1,
                    )
                except Exception as e:
                    logger.warning("Judge scoring failed: %s", e)
                    best_conversation = conversation
                    break
            else:
                best_conversation = conversation
                break

        result = best_conversation or conversation

        # Record for diversity steering
        if self.steering and result:
            self.steering.record_conversation(result)

        result.metadata["num_retries"] = attempt

        return result

    def _run_conversation(
        self,
        chain: list[tuple[str, str]],
        scenario: str,
        repair_hints: str,
        metadata: dict[str, Any],
    ) -> Conversation:
        """Run a single conversation generation attempt."""
        context = ConversationContext()
        all_tool_calls = []
        all_tool_outputs = []
        used_tools: set[tuple[str, str]] = set()
        required_tools = set(chain)

        steering_guidance = ""
        if self.steering:
            tool_names = [f"{t}/{a}" for t, a in chain]
            steering_guidance = self.steering.get_steering_guidance(tool_names)

        combined_guidance = "\n".join(filter(None, [steering_guidance, repair_hints]))

        for turn in range(self.max_turns):
            # 1. User message
            user_msg = self.user_agent.run(
                context,
                scenario=scenario,
                steering_guidance=combined_guidance if turn == 0 else "",
            )
            context.add_message(user_msg)

            # 2. Assistant response
            remaining_tools = list(required_tools - used_tools)
            # Provide all chain tools so the assistant knows what's available
            assistant_msg = self.assistant_agent.run(
                context,
                available_tools=chain,
            )
            context.add_message(assistant_msg)

            # 3. If assistant made tool calls, execute them
            if assistant_msg.tool_calls:
                tool_msg = self.tool_executor.run(
                    context,
                    tool_calls=assistant_msg.tool_calls,
                )
                context.add_message(tool_msg)

                all_tool_calls.extend(assistant_msg.tool_calls)
                if tool_msg.tool_outputs:
                    all_tool_outputs.extend(tool_msg.tool_outputs)

                for tc in assistant_msg.tool_calls:
                    used_tools.add(tc.endpoint_key)

                # 4. Assistant responds to tool results
                followup_msg = self.assistant_agent.run(
                    context,
                    available_tools=chain,
                )
                context.add_message(followup_msg)

                # Check if assistant made more tool calls in follow-up
                if followup_msg.tool_calls:
                    tool_msg2 = self.tool_executor.run(
                        context,
                        tool_calls=followup_msg.tool_calls,
                    )
                    context.add_message(tool_msg2)

                    all_tool_calls.extend(followup_msg.tool_calls)
                    if tool_msg2.tool_outputs:
                        all_tool_outputs.extend(tool_msg2.tool_outputs)

                    for tc in followup_msg.tool_calls:
                        used_tools.add(tc.endpoint_key)

                    # Final summary after all tool calls
                    summary_msg = self.assistant_agent.run(
                        context,
                        available_tools=chain,
                    )
                    context.add_message(summary_msg)

            # 5. Check if we should end
            if self._should_end(context, used_tools, required_tools, turn):
                break

        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        return Conversation(
            conversation_id=conv_id,
            messages=context.messages,
            tool_calls=all_tool_calls,
            tool_outputs=all_tool_outputs,
            metadata={
                **metadata,
                "scenario": scenario,
                "chain": [f"{t}/{a}" for t, a in chain],
                "tools_used": list({f"{t}/{a}" for t, a in used_tools}),
                "num_turns": context.get_turn_count(),
                "tools_planned": len(chain),
                "tools_executed": len(used_tools),
            },
        )

    def _should_end(
        self,
        context: ConversationContext,
        used_tools: set,
        required_tools: set,
        turn: int,
    ) -> bool:
        """Determine if the conversation should end."""
        # End if all required tools have been used
        if used_tools >= required_tools:
            return True

        # End if we've had enough turns without using tools
        if turn >= 3 and not used_tools:
            return True

        # Let max_turns handle the rest
        return False

    @staticmethod
    def _build_repair_hints(scores: JudgeScore) -> str:
        """Generate prompt hints based on which dimensions scored low."""
        hints = []

        if scores.naturalness < 3.0:
            hints.append(
                "IMPORTANT: Make the conversation more natural and conversational. "
                "Use contractions, informal language, and realistic phrasing."
            )

        if scores.tool_correctness < 3.0:
            hints.append(
                "IMPORTANT: Ensure tool calls use correct parameter names and valid values. "
                "Double-check that arguments match the tool's schema exactly."
            )

        if scores.task_completion < 3.0:
            hints.append(
                "IMPORTANT: Make sure the user's request is fully addressed. "
                "The assistant should complete all steps needed to fulfill the task."
            )

        return " ".join(hints)
