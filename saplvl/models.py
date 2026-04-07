"""Core data models used across all modules."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """A single parameter for an API endpoint."""

    name: str
    type: str = "string"
    description: str = ""
    default: Any | None = None
    example_value: Any | None = None


class APIEndpoint(BaseModel):
    """A single API endpoint within a tool."""

    name: str
    url: str = ""
    description: str = ""
    method: str = "GET"
    required_parameters: list[ToolParameter] = Field(default_factory=list)
    optional_parameters: list[ToolParameter] = Field(default_factory=list)

    @property
    def all_parameters(self) -> list[ToolParameter]:
        return self.required_parameters + self.optional_parameters


class Tool(BaseModel):
    """A tool containing one or more API endpoints."""

    tool_name: str
    standardized_name: str = ""
    tool_description: str = ""
    category: str = ""
    api_list: list[APIEndpoint] = Field(default_factory=list)


class ToolCall(BaseModel):
    """A single tool call made by the assistant."""

    tool_name: str
    api_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @property
    def endpoint_key(self) -> tuple[str, str]:
        return (self.tool_name, self.api_name)


class ToolOutput(BaseModel):
    """The response from a (mock) tool execution."""

    tool_call: ToolCall
    response: dict[str, Any] = Field(default_factory=dict)
    success: bool = True


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["user", "assistant", "tool", "system"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_outputs: list[ToolOutput] | None = None


class JudgeScore(BaseModel):
    """Quality scores from the LLM-as-judge."""

    naturalness: float = Field(ge=1.0, le=5.0, description="How natural the conversation feels")
    tool_correctness: float = Field(ge=1.0, le=5.0, description="Correctness of tool selection and arguments")
    task_completion: float = Field(ge=1.0, le=5.0, description="Whether the user's goal was achieved")
    reasoning: str = ""

    @property
    def mean_score(self) -> float:
        return (self.naturalness + self.tool_correctness + self.task_completion) / 3


class Conversation(BaseModel):
    """A complete generated conversation with metadata."""

    conversation_id: str = Field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:8]}")
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_outputs: list[ToolOutput] = Field(default_factory=list)
    judge_scores: JudgeScore | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.messages)

    @property
    def tools_used(self) -> list[str]:
        return list({f"{tc.tool_name}/{tc.api_name}" for tc in self.tool_calls})

    @property
    def num_distinct_tools(self) -> int:
        return len({tc.tool_name for tc in self.tool_calls})

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_calls)
