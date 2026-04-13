"""JSONL wire format serialization for generated conversations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conv_gen.models import Conversation, JudgeScore, Message, ToolCall, ToolOutput


# ---------------------------------------------------------------------------- #
# Conversation -> wire dict                                                    #
# ---------------------------------------------------------------------------- #

def to_wire_dict(conv: Conversation) -> dict[str, Any]:
    """Serialize a Conversation to the wire format dict."""
    wire_messages: list[dict[str, Any]] = []

    for msg in conv.messages:
        if msg.role in ("user", "system"):
            wire_messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        elif msg.role == "assistant":
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content,
            }
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "endpoint": f"{tc.tool_name}/{tc.api_name}",
                        "arguments": tc.arguments or {},
                    }
                    for tc in msg.tool_calls
                ]
            wire_messages.append(entry)
        elif msg.role == "tool":
            # Expand parallel tool_outputs into N consecutive tool messages
            if msg.tool_outputs:
                for to in msg.tool_outputs:
                    wire_messages.append({
                        "role": "tool",
                        "content": to.response,
                    })

    result: dict[str, Any] = {
        "conversation_id": conv.conversation_id,
        "messages": wire_messages,
    }

    if conv.judge_scores is not None:
        scores: dict[str, Any] = {
            "naturalness": conv.judge_scores.naturalness,
            "tool_correctness": conv.judge_scores.tool_correctness,
            "task_completion": conv.judge_scores.task_completion,
        }
        if conv.judge_scores.reasoning:
            scores["reasoning"] = conv.judge_scores.reasoning
        result["judge_scores"] = scores

    if conv.metadata:
        result["metadata"] = conv.metadata

    return result


# ---------------------------------------------------------------------------- #
# wire dict -> Conversation                                                    #
# ---------------------------------------------------------------------------- #

def from_wire_dict(wire: dict[str, Any]) -> Conversation:
    """Parse a wire-format dict back into an internal Conversation."""
    wire_messages = wire.get("messages", [])
    internal_messages: list[Message] = []
    all_tool_calls: list[ToolCall] = []
    all_tool_outputs: list[ToolOutput] = []

    i = 0
    while i < len(wire_messages):
        wm = wire_messages[i]
        role = wm.get("role")

        if role in ("user", "system"):
            internal_messages.append(Message(
                role=role,
                content=wm.get("content"),
            ))
            i += 1
        elif role == "assistant":
            content = wm.get("content")
            if content is not None and not isinstance(content, str):
                content = json.dumps(content)

            wire_calls = wm.get("tool_calls") or []
            internal_calls: list[ToolCall] = []
            for wc in wire_calls:
                endpoint = wc.get("endpoint", "")
                if "/" in endpoint:
                    tool_name, api_name = endpoint.split("/", 1)
                else:
                    tool_name, api_name = endpoint, endpoint
                tc = ToolCall(
                    tool_name=tool_name,
                    api_name=api_name,
                    arguments=wc.get("arguments") or {},
                )
                internal_calls.append(tc)
                all_tool_calls.append(tc)

            internal_messages.append(Message(
                role="assistant",
                content=content,
                tool_calls=internal_calls or None,
            ))
            i += 1

            # Pair with next N tool messages
            if internal_calls:
                paired_outputs: list[ToolOutput] = []
                for tc in internal_calls:
                    if i >= len(wire_messages):
                        break
                    next_msg = wire_messages[i]
                    if next_msg.get("role") != "tool":
                        break
                    response = next_msg.get("content")
                    if not isinstance(response, dict):
                        response = {"result": response} if response is not None else {}
                    to = ToolOutput(
                        tool_call=tc,
                        response=response,
                        success=True,
                    )
                    paired_outputs.append(to)
                    all_tool_outputs.append(to)
                    i += 1
                if paired_outputs:
                    internal_messages.append(Message(
                        role="tool",
                        content=None,
                        tool_outputs=paired_outputs,
                    ))
        elif role == "tool":
            # Orphan tool message — preserve as unanchored tool_output
            response = wm.get("content")
            if not isinstance(response, dict):
                response = {"result": response} if response is not None else {}
            placeholder_call = ToolCall(
                tool_name="unknown", api_name="unknown", arguments={},
            )
            to = ToolOutput(
                tool_call=placeholder_call,
                response=response,
                success=True,
            )
            internal_messages.append(Message(
                role="tool",
                content=None,
                tool_outputs=[to],
            ))
            all_tool_outputs.append(to)
            i += 1
        else:
            i += 1

    judge_scores = None
    if "judge_scores" in wire:
        js = wire["judge_scores"]
        judge_scores = JudgeScore(
            naturalness=js.get("naturalness", 3.0),
            tool_correctness=js.get("tool_correctness", 3.0),
            task_completion=js.get("task_completion", 3.0),
            reasoning=js.get("reasoning", ""),
        )

    return Conversation(
        conversation_id=wire.get("conversation_id", "conv_unknown"),
        messages=internal_messages,
        tool_calls=all_tool_calls,
        tool_outputs=all_tool_outputs,
        judge_scores=judge_scores,
        metadata=wire.get("metadata") or {},
    )


# ---------------------------------------------------------------------------- #
# Format detection and JSONL helpers                                            #
# ---------------------------------------------------------------------------- #

def _looks_like_wire_format(record: dict[str, Any]) -> bool:
    """True if record has no top-level tool_calls/tool_outputs arrays."""
    if "tool_calls" in record or "tool_outputs" in record:
        return False
    return "messages" in record


def to_wire_json(conv: Conversation) -> str:
    """Serialize a Conversation to a single-line wire-format JSON string."""
    return json.dumps(to_wire_dict(conv), ensure_ascii=False)


def from_any_json(line: str) -> Conversation:
    """Parse a JSONL line to a Conversation, auto-detecting wire or legacy format."""
    data = json.loads(line)
    if _looks_like_wire_format(data):
        return from_wire_dict(data)
    return Conversation.model_validate(data)


def write_jsonl(path: Path | str, conversations: list[Conversation]) -> None:
    """Write conversations to a JSONL file in wire format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for conv in conversations:
            f.write(to_wire_json(conv) + "\n")


def read_jsonl(path: Path | str) -> list[Conversation]:
    """Read a JSONL file into a list of Conversations."""
    path = Path(path)
    conversations: list[Conversation] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(from_any_json(line))
    return conversations
