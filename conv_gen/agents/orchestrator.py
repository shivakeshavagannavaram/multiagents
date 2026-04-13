"""Orchestrates multi-agent conversation generation."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from conv_gen.agents.assistant import AssistantAgent
from conv_gen.agents.plan import ConversationPlan, ConversationStatus
from conv_gen.agents.tool_executor import ToolExecutorAgent
from conv_gen.agents.user_simulator import UserSimulatorAgent, build_completion_guidance
from conv_gen.memory.context import ConversationContext
from conv_gen.memory.steering import DiversitySteering
from conv_gen.models import Conversation, JudgeScore, Message

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """Orchestrates the conversation generation loop with judge-driven retry/repair."""

    def __init__(
        self,
        user_agent: UserSimulatorAgent,
        assistant_agent: AssistantAgent,
        tool_executor: ToolExecutorAgent,
        judge: Any | None = None,
        steering: DiversitySteering | None = None,
        max_turns: int = 5,
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
        pattern: str = "sequential",
        steps: list[list[tuple[str, str]]] | None = None,
        plan_kwargs: dict[str, Any] | None = None,
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
                chain, scenario, repair_hints, metadata or {},
                pattern=pattern, steps=steps,
                plan_kwargs=plan_kwargs,
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
        pattern: str = "sequential",
        steps: list[list[tuple[str, str]]] | None = None,
        plan_kwargs: dict[str, Any] | None = None,
    ) -> Conversation:
        """Run a single conversation generation attempt."""
        context = ConversationContext()
        all_tool_calls = []
        all_tool_outputs = []

        # Create the plan — Director can override limits via plan_kwargs
        plan = ConversationPlan(planned_tools=chain, **(plan_kwargs or {}))

        steering_guidance = ""
        if self.steering:
            tool_names = [f"{t}/{a}" for t, a in chain]
            steering_guidance = self.steering.get_steering_guidance(tool_names)

        combined_guidance = "\n".join(filter(None, [steering_guidance, repair_hints]))

        # Build step-aware assistant hints for parallel patterns
        parallel_hint = ""
        if pattern == "parallel" and steps and len(steps[0]) > 1:
            tool_names_str = ", ".join(f"{t}/{a}" for t, a in steps[0])
            parallel_hint = (
                f"Call these tools together in a single response (parallel): "
                f"{tool_names_str}. They are independent and can run simultaneously."
            )

        for turn in range(plan.max_turns):
            plan.advance_turn()

            if plan.is_complete():
                break

            # 1. User message
            need_user_msg = True

            if need_user_msg:
                completion_guidance = build_completion_guidance(
                    tools_remaining_count=len(plan.tools_remaining),
                    tools_completed_count=len(plan.planned_tools_completed),
                    conversation_type=metadata.get("conversation_type", ""),
                )
                user_msg = self.user_agent.run(
                    context,
                    scenario=scenario,
                    steering_guidance=combined_guidance if turn == 0 else "",
                    plan_status=completion_guidance,
                )
                context.add_message(user_msg)
                plan.add_message()

                if plan.is_complete():
                    break

                if plan.clarification_pending:
                    plan.mark_clarification_answered()

            # 2. Assistant response
            should_have_tools = not plan.is_completing() or not plan.used_tools
            if should_have_tools:
                remaining_tools = [
                    t for t in chain if t not in plan.planned_tools_completed
                ]
                if not remaining_tools:
                    remaining_tools = chain
            else:
                remaining_tools = []

            assistant_instruction = plan.assistant_instruction
            if parallel_hint and not plan.used_tools:
                assistant_instruction = f"{parallel_hint}\n{assistant_instruction}"

            assistant_msg = self.assistant_agent.run(
                context,
                available_tools=remaining_tools,
                plan_status=assistant_instruction,
            )
            if not assistant_msg.content and not assistant_msg.tool_calls:
                continue
            context.add_message(assistant_msg)
            plan.add_message()

            # Track clarification questions (text-only with a question mark)
            if not assistant_msg.tool_calls:
                content = assistant_msg.content or ""
                is_question = "?" in content
                if not plan.used_tools or is_question:
                    plan.mark_clarification_asked()

            # Main-loop retry on refusal/narration — re-prompt with remaining tools
            if (
                not assistant_msg.tool_calls
                and plan.tools_remaining
                and not plan.is_complete()
                and "?" not in (assistant_msg.content or "")
            ):
                remaining_tools = [
                    t for t in chain if t not in plan.planned_tools_completed
                ]
                if remaining_tools:
                    retry_msg = self.assistant_agent.run(
                        context,
                        available_tools=remaining_tools,
                        plan_status=(
                            "The user's request requires more steps. Call "
                            "one of the available tools now — do not refuse "
                            "or explain without calling a tool."
                        ),
                    )
                    if retry_msg.tool_calls:
                        context.add_message(retry_msg)
                        plan.add_message()
                        assistant_msg = retry_msg

            # 3. Execute tool calls
            if assistant_msg.tool_calls:
                all_tool_calls, all_tool_outputs = self._execute_and_follow_up(
                    context, plan, chain, assistant_msg,
                    all_tool_calls, all_tool_outputs,
                    scenario=scenario,
                )

            if plan.is_complete():
                break

            # 4. Closing user message on completion
            if plan.is_completing():
                closing_guidance = build_completion_guidance(
                    tools_remaining_count=len(plan.tools_remaining),
                    tools_completed_count=len(plan.planned_tools_completed),
                    conversation_type=metadata.get("conversation_type", ""),
                )
                closing_msg = self.user_agent.run(
                    context,
                    scenario=scenario,
                    plan_status=closing_guidance,
                )
                context.add_message(closing_msg)
                plan.add_message()
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
                "tools_used": list({f"{t}/{a}" for t, a in plan.used_tools}),
                "num_turns": context.get_turn_count(),
                "tools_planned": len(chain),
                "tools_executed": len(plan.planned_tools_completed),
                "plan_summary": plan.summary(),
            },
        )

    def _execute_and_follow_up(
        self,
        context: ConversationContext,
        plan: ConversationPlan,
        chain: list[tuple[str, str]],
        assistant_msg: Message,
        all_tool_calls: list,
        all_tool_outputs: list,
        scenario: str = "",
        max_depth: int = 3,
    ) -> tuple[list, list]:
        """Execute tool calls and handle chained follow-ups up to max_depth.

        Invariants: no empty assistant messages get added (ghost-turn
        prevention) and no tool_use block gets added without a matching
        tool_result (orphan prevention).
        """
        current_msg = assistant_msg

        for depth in range(max_depth):
            is_last_iter = (depth == max_depth - 1)

            chain_context = self._build_chain_context(current_msg.tool_calls, chain)

            tool_msg = self.tool_executor.run(
                context,
                tool_calls=current_msg.tool_calls,
                scenario=scenario,
                chain_context=chain_context,
            )
            context.add_message(tool_msg)
            plan.add_message()

            all_tool_calls.extend(current_msg.tool_calls)
            if tool_msg.tool_outputs:
                all_tool_outputs.extend(tool_msg.tool_outputs)

            for tc in current_msg.tool_calls:
                plan.record_tool_use(tc.tool_name, tc.api_name)

            if plan.is_complete():
                break

            should_have_tools = not plan.is_completing() or not plan.used_tools
            followup_msg = self.assistant_agent.run(
                context,
                available_tools=chain if should_have_tools else [],
                plan_status=plan.assistant_instruction,
            )

            if not followup_msg.content and not followup_msg.tool_calls:
                break

            # Don't orphan a pending tool_use at max_depth
            if followup_msg.tool_calls and is_last_iter:
                logger.warning(
                    "Dropping follow-up tool call at max_depth=%d (tool=%s/%s)",
                    max_depth,
                    followup_msg.tool_calls[0].tool_name,
                    followup_msg.tool_calls[0].api_name,
                )
                break

            context.add_message(followup_msg)
            plan.add_message()

            if followup_msg.tool_calls and not plan.is_complete():
                current_msg = followup_msg
                continue

            # Assistant returned text — re-prompt with remaining tools
            if plan.tools_remaining and not plan.is_complete():
                remaining_tools = [
                    t for t in chain
                    if t not in plan.planned_tools_completed
                ]
                if remaining_tools:
                    retry_msg = self.assistant_agent.run(
                        context,
                        available_tools=remaining_tools,
                        plan_status="Continue with the next step using the available tools.",
                    )

                    if not retry_msg.content and not retry_msg.tool_calls:
                        break

                    if retry_msg.tool_calls and is_last_iter:
                        logger.warning(
                            "Dropping retry tool call at max_depth=%d (tool=%s/%s)",
                            max_depth,
                            retry_msg.tool_calls[0].tool_name,
                            retry_msg.tool_calls[0].api_name,
                        )
                        break

                    context.add_message(retry_msg)
                    plan.add_message()

                    if retry_msg.tool_calls and not plan.is_complete():
                        current_msg = retry_msg
                        continue

            break

        return all_tool_calls, all_tool_outputs

    def _build_chain_context(
        self,
        current_tool_calls: list,
        chain: list[tuple[str, str]],
    ) -> str:
        """Describe what the next chain step needs, to hint the mock generator."""
        current_set = {(tc.tool_name, tc.api_name) for tc in current_tool_calls}
        remaining = []
        found_current = False
        for tool in chain:
            if tool in current_set:
                found_current = True
                continue
            if found_current:
                remaining.append(tool)

        if not remaining:
            return ""

        next_tool, next_api = remaining[0]
        endpoint = self.assistant_agent.registry.get_endpoint(next_tool, next_api)
        if not endpoint:
            return ""

        param_descs = []
        for p in endpoint.required_parameters:
            param_descs.append(f"{p.name} ({p.description or p.type})")
        for p in endpoint.optional_parameters[:3]:
            param_descs.append(f"{p.name} ({p.description or p.type})")

        if not param_descs:
            return ""

        next_desc = endpoint.description or f"{next_tool}/{next_api}"
        return (
            f"This API call is part of a workflow. The next step will be: "
            f"{next_desc}. It will need these values: {', '.join(param_descs)}. "
            f"Include relevant fields in your response that would naturally "
            f"be needed for that next step."
        )

    @staticmethod
    def _build_repair_hints(scores: JudgeScore) -> str:
        """Generate repair hints from judge's specific_issues or low-scoring dimensions."""
        hints = []

        if "specific_issues:" in scores.reasoning:
            issues_part = scores.reasoning.split("specific_issues:")[-1].strip()
            if issues_part:
                hints.append(f"FIX THESE SPECIFIC ISSUES: {issues_part}")

        if not hints:
            if scores.naturalness < 3.0:
                hints.append(
                    "Make the conversation more natural. "
                    "No filler phrases, no empty messages."
                )

            if scores.tool_correctness < 3.0:
                hints.append(
                    "Use correct values in tool arguments. "
                    "If a prior tool returned an ID, use that exact ID in the next call. "
                    "Do not put a user-provided value where a system-generated ID belongs."
                )

            if scores.task_completion < 3.0:
                hints.append(
                    "Ensure all steps form a coherent workflow. "
                    "Each tool call should logically depend on the previous one."
                )

        return " ".join(hints)
