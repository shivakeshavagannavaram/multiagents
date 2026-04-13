"""LLM-as-judge for conversation quality scoring."""

from __future__ import annotations

import json
import logging
import re

import openai

from conv_gen.models import Conversation, JudgeScore

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of AI assistant conversations involving tool use. You score conservatively — a 5 is exceptional, a 3 is mediocre, and you give 1-2 when things are clearly wrong.

## Scoring Rubric

### Naturalness (1-5)
- 5: Indistinguishable from a real user-assistant interaction. Concise, varied tone, no filler
- 4: Mostly natural with minor awkwardness (one redundant sentence, slightly stiff phrasing)
- 3: Functional but issues: repetitive phrasing, echoing the user's request back, unnecessary "Is there anything else?", or robotic structure
- 2: Clearly artificial — formulaic turn-taking, every response follows the same template, or unnatural length
- 1: Incoherent, nonsensical, or completely breaks conversational norms

### Tool Correctness (1-5)
- 5: Correct tools, valid arguments matching user intent, proper sequencing. Chained arguments use EXACT values from prior tool responses (not hallucinated)
- 4: Correct tools and sequencing but minor issues (a missing optional parameter, a slightly wrong format)
- 3: Right tools but arguments don't match user intent, OR tool results are ignored in follow-up calls
- 2: Wrong tools selected, OR arguments use hallucinated values (IDs, names) not present in any tool response. This is a critical failure
- 1: Invalid tool calls, nonsensical arguments, or no tool calls when tools were clearly needed

### Task Completion (1-5)
- 5: User's original goal is fully addressed with a clear, useful final answer that references concrete results
- 4: Goal mostly addressed but minor gaps (e.g., didn't mention one detail from the results)
- 3: Partial — some progress but the user's goal is not fully met, OR conversation ends abruptly without summarizing results
- 2: Significant parts of the user's request are unaddressed or abandoned
- 1: Task not attempted, wrong task performed, or assistant refused without reason

## Critical Checks (apply BEFORE scoring)
- Hallucinated values: If the assistant uses an ID, name, or value in a tool call that was NOT returned by any previous tool response OR mentioned by the user, tool_correctness cannot exceed 2
- Parameter misuse: If a tool argument uses a value that semantically doesn't match the parameter's purpose (e.g., putting an employee ID where a user ID is expected, when the correct user ID was available from a prior response), tool_correctness should be penalized
- Incoherent workflow: If the conversation combines completely unrelated tasks with no logical dependency between them (e.g., setting up authentication then checking stock prices), task_completion cannot be 5 — the tasks should form a coherent workflow
- Empty/blank messages: If the assistant produces empty responses with no content and no tool calls, naturalness should be penalized
- Scenario mismatch: If the conversation doesn't address the original user scenario, task_completion cannot exceed 2
- No tool calls: If tools were available and the user's request clearly needed them but the assistant only gave a text response, tool_correctness = 1

## Few-Shot Examples

### Example A — Score: 5, 5, 5
USER: Find me a hotel in Paris under 200 euros
ASSISTANT [TOOL CALL]: hotels/search({"city": "Paris", "max_price": 200})
TOOL RESPONSE: {"results": [{"hotel_id": "htl_a8f2", "name": "Hotel du Marais", "price": 175}]}
ASSISTANT: I found Hotel du Marais at €175/night. Want me to book it?
USER: Yes, for June 15-18
ASSISTANT [TOOL CALL]: hotels/book({"hotel_id": "htl_a8f2", "checkin": "2026-06-15", "checkout": "2026-06-18"})
TOOL RESPONSE: {"booking_id": "bk_3391", "status": "confirmed"}
ASSISTANT: Booked! Confirmation: bk_3391. Hotel du Marais, Jun 15-18, €175/night.
→ Natural flow, correct tool args, exact IDs from responses, task fully done.

### Example B — Score: 3, 2, 3
USER: Book me a flight to London
ASSISTANT: I'd be happy to help you book a flight to London! Let me search for available flights for you.
ASSISTANT [TOOL CALL]: flights/search({"destination": "London"})
TOOL RESPONSE: {"flights": [{"flight_id": "fl_992", "airline": "BA", "price": 245}]}
ASSISTANT: I found some flights! Let me book the best option for you.
ASSISTANT [TOOL CALL]: flights/book({"flight_id": "fl_100", "passenger": "John Doe"})
TOOL RESPONSE: {"booking_id": "bk_441", "status": "confirmed"}
ASSISTANT: I've booked your flight to London! Is there anything else I can help with?
→ Filler phrases ("I'd be happy to help"), hallucinated flight_id (fl_100 not in search results), generic closing.

### Example C — Score: 2, 1, 1
USER: What's the weather in Tokyo?
ASSISTANT: Sure, I can help with that! Let me look up the weather information for Tokyo for you right away.
USER: Thanks
ASSISTANT: You're welcome! Unfortunately I don't have access to weather data right now. Is there anything else I can help you with?
→ Tools available but not used, filler everywhere, task not attempted.

## Your Task

Score the conversation below. For EACH dimension, first write your reasoning (what you observed), THEN assign the score. Reasoning must come before the score — do not decide the score first."""

JUDGE_USER_PROMPT = """## Scenario
{scenario}

## Conversation
{conversation}

## Tools Used
{tools_info}

Think through each dimension carefully, then respond with ONLY this JSON:
{{
    "naturalness_reasoning": "<what you observed about conversation flow, tone, repetition>",
    "naturalness": <float 1-5>,
    "tool_correctness_reasoning": "<what you observed about tool selection, arguments, value chaining>",
    "tool_correctness": <float 1-5>,
    "task_completion_reasoning": "<what you observed about whether the scenario goal was met>",
    "task_completion": <float 1-5>,
    "specific_issues": ["<list each specific problem found, e.g. 'Step 2 used EMP12345 as userID but should have used Userid from step 1 response', 'Steps 1-2 and step 3 are unrelated tasks'>"]
}}"""


class JudgeLLM:
    """Scores conversations using OpenAI as an LLM-as-judge.

    Evaluates on three dimensions:
    - naturalness: How natural the conversation feels
    - tool_correctness: Correctness of tool selection and arguments
    - task_completion: Whether the user's goal was achieved
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "gpt-4o",
    ):
        self.client = client
        self.model = model

    def score(self, conversation: Conversation) -> JudgeScore:
        """Score a single conversation using LLM + structural checks."""
        conv_text = self._format_conversation(conversation)
        tools_info = self._format_tools(conversation)
        scenario = conversation.metadata.get("scenario", "Not provided")

        user_prompt = JUDGE_USER_PROMPT.format(
            scenario=scenario,
            conversation=conv_text,
            tools_info=tools_info,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1000,
        )

        raw = json.loads(response.choices[0].message.content)

        # Extract per-dimension reasoning for the combined reasoning field
        reasoning_parts = []
        for dim in ("naturalness", "tool_correctness", "task_completion"):
            dim_reasoning = raw.get(f"{dim}_reasoning", "")
            if dim_reasoning:
                reasoning_parts.append(f"{dim}: {dim_reasoning}")

        # Extract specific issues for repair hints
        specific_issues = raw.get("specific_issues", [])
        if specific_issues:
            reasoning_parts.append(f"specific_issues: {'; '.join(specific_issues)}")

        llm_scores = JudgeScore(
            naturalness=self._clamp(raw.get("naturalness", 3.0)),
            tool_correctness=self._clamp(raw.get("tool_correctness", 3.0)),
            task_completion=self._clamp(raw.get("task_completion", 3.0)),
            reasoning="\n".join(reasoning_parts) if reasoning_parts else raw.get("reasoning", ""),
        )

        # Apply structural checks — can only lower scores, not raise them
        return self._apply_structural_checks(conversation, llm_scores)

    def _apply_structural_checks(
        self, conversation: Conversation, scores: JudgeScore
    ) -> JudgeScore:
        """Apply mechanical checks that catch errors the LLM might miss.

        These can only LOWER scores, never raise them.
        """
        penalties = []
        tool_correctness = scores.tool_correctness
        task_completion = scores.task_completion

        # Check 1: No tool calls when tools were available
        planned_tools = conversation.metadata.get("tools_planned", 0)
        if planned_tools > 0 and len(conversation.tool_calls) == 0:
            tool_correctness = min(tool_correctness, 1.0)
            penalties.append("No tool calls made despite tools being available")

        # Check 2: chain-break detection — field-name-aware hallucination check.
        user_values = set()
        for msg in conversation.messages:
            if msg.role == "user" and msg.content:
                for word in msg.content.split():
                    cleaned = word.strip(".,!?;:\"'()[]{}").strip()
                    if cleaned and len(cleaned) >= 3:
                        user_values.add(cleaned)

        hallucinated = False
        prior_field_values: dict[str, set[str]] = {}

        for i, tc in enumerate(conversation.tool_calls):
            if i > 0:
                for arg_name, arg_val in tc.arguments.items():
                    if not isinstance(arg_val, str):
                        continue
                    name_lower = arg_name.lower()
                    if not any(
                        w in name_lower
                        for w in ("_id", "id", "_code", "_key", "_token")
                    ):
                        continue
                    if arg_val in ("default", ""):
                        continue

                    norm_arg_name = self._normalize_field_name(arg_name)
                    matching_prior_values = prior_field_values.get(norm_arg_name, set())

                    if not matching_prior_values:
                        continue  # no prior field of this name — fabrication OK
                    if arg_val in matching_prior_values:
                        continue
                    if arg_val in user_values:
                        continue

                    hallucinated = True
                    sample_prior = sorted(matching_prior_values)[:3]
                    penalties.append(
                        f"Chain break: {arg_name}={arg_val} — prior response "
                        f"had {norm_arg_name} in {sample_prior}"
                    )

            if i < len(conversation.tool_outputs):
                self._collect_field_values(
                    conversation.tool_outputs[i].response, prior_field_values
                )

        if hallucinated:
            tool_correctness = min(tool_correctness, 2.5)

        # Check 3: Very short conversation with no resolution
        if len(conversation.messages) <= 2 and planned_tools > 0:
            task_completion = min(task_completion, 2.0)
            penalties.append("Conversation too short to complete task")

        # Build updated reasoning
        reasoning = scores.reasoning
        if penalties:
            reasoning += "\n[Structural penalties: " + "; ".join(penalties) + "]"

        return JudgeScore(
            naturalness=scores.naturalness,
            tool_correctness=self._clamp(tool_correctness),
            task_completion=self._clamp(task_completion),
            reasoning=reasoning,
        )

    @staticmethod
    def _collect_values(data: dict, values: set, depth: int = 0) -> None:
        """Recursively collect all scalar values from a response dict."""
        if depth > 5 or not isinstance(data, dict):
            return
        for key, val in data.items():
            if isinstance(val, str) and val:
                values.add(val)
            elif isinstance(val, (int, float)):
                values.add(str(val))
            elif isinstance(val, dict):
                JudgeLLM._collect_values(val, values, depth + 1)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        JudgeLLM._collect_values(item, values, depth + 1)
                    elif isinstance(item, str) and item:
                        values.add(item)

    @staticmethod
    def _normalize_field_name(name: str) -> str:
        """Normalize camelCase/PascalCase/kebab-case to lowercase snake_case."""
        if not name:
            return ""
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
        return s.lower().strip("_")

    @staticmethod
    def _collect_field_values(
        data, out: dict, depth: int = 0, max_depth: int = 5
    ) -> None:
        """Build a {normalized_field_name: set of values} map from a response."""
        if depth > max_depth or not isinstance(data, dict):
            return
        for key, val in data.items():
            norm_key = JudgeLLM._normalize_field_name(key)
            # bool is a subclass of int — skip first
            if isinstance(val, bool):
                continue
            if isinstance(val, str) and val:
                out.setdefault(norm_key, set()).add(val)
            elif isinstance(val, (int, float)):
                out.setdefault(norm_key, set()).add(str(val))
            elif isinstance(val, dict):
                JudgeLLM._collect_field_values(val, out, depth + 1, max_depth)
            elif isinstance(val, list):
                for item in val[:10]:
                    if isinstance(item, bool):
                        continue
                    if isinstance(item, dict):
                        JudgeLLM._collect_field_values(
                            item, out, depth + 1, max_depth
                        )
                    elif isinstance(item, str) and item:
                        out.setdefault(norm_key, set()).add(item)
                    elif isinstance(item, (int, float)):
                        out.setdefault(norm_key, set()).add(str(item))

    def batch_score(self, conversations: list[Conversation]) -> list[JudgeScore]:
        """Score multiple conversations sequentially."""
        scores = []
        for i, conv in enumerate(conversations):
            try:
                score = self.score(conv)
                scores.append(score)
                logger.info(
                    "Scored %d/%d: naturalness=%.1f, tool_correctness=%.1f, task_completion=%.1f",
                    i + 1, len(conversations),
                    score.naturalness, score.tool_correctness, score.task_completion,
                )
            except Exception as e:
                logger.warning("Failed to score conversation %d: %s", i, e)
                # Default below quality threshold so retry kicks in
                scores.append(JudgeScore(
                    naturalness=2.0, tool_correctness=2.0, task_completion=2.0,
                    reasoning=f"Scoring failed: {e}",
                ))
        return scores

    @staticmethod
    def _format_conversation(conversation: Conversation) -> str:
        """Format conversation messages for the judge prompt."""
        lines = []
        for msg in conversation.messages:
            role = msg.role.upper()
            if msg.content:
                lines.append(f"{role}: {msg.content}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(
                        f"{role} [TOOL CALL]: {tc.tool_name}/{tc.api_name}"
                        f"({json.dumps(tc.arguments)})"
                    )
            if msg.tool_outputs:
                for to in msg.tool_outputs:
                    lines.append(
                        f"TOOL RESPONSE [{to.tool_call.tool_name}/{to.tool_call.api_name}]: "
                        f"{json.dumps(to.response)}"
                    )
        return "\n".join(lines)

    @staticmethod
    def _format_tools(conversation: Conversation) -> str:
        """Format tool usage summary."""
        if not conversation.tool_calls:
            return "No tools were used."

        lines = [f"Tools called ({len(conversation.tool_calls)} total):"]
        for tc in conversation.tool_calls:
            lines.append(f"  - {tc.tool_name}/{tc.api_name}: args={json.dumps(tc.arguments)}")
        return "\n".join(lines)

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp a score to [1.0, 5.0]."""
        try:
            return max(1.0, min(5.0, float(value)))
        except (TypeError, ValueError):
            return 3.0
