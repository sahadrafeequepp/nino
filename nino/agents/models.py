"""Data models and schemas for Nino sub-agent framework and delegation protocol."""

import time
from typing import Any, Literal
from pydantic import BaseModel, Field

from nino.tools.confirmation import Confirmation


class AgentBudget(BaseModel):
    """Budget limits for sub-agent execution."""

    max_steps: int = Field(default=5, ge=1, le=20, description="Maximum tool execution steps")
    timeout_seconds: float = Field(default=60.0, ge=0.05, le=300.0, description="Maximum runtime in seconds")
    max_tokens: int = Field(default=2048, ge=256, le=8192, description="Estimated token budget")


class AgentDefinition(BaseModel):
    """Configuration definition for a specialized sub-agent."""

    name: str = Field(..., description="Unique identifier for the sub-agent")
    description: str = Field(..., description="Human-readable summary of agent capabilities")
    model: str = Field(default="gemma2:2b", description="Model override for this agent")
    max_tier: str = Field(default="READ_ONLY", description="Maximum allowed permission tier (READ_ONLY, REVERSIBLE, SENSITIVE, RESTRICTED)")
    allowed_tools: list[str] = Field(default_factory=list, description="Allowlist of tool names the agent is permitted to invoke")
    default_budget: AgentBudget = Field(default_factory=AgentBudget, description="Default budget configuration")
    system_prompt: str = Field(..., description="Specialized system prompt instructions")


class AgentStepTrace(BaseModel):
    """Detailed record of a single step taken during sub-agent execution."""

    step: int
    thought: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    is_tainted: bool = False


class DelegationResult(BaseModel):
    """Structured response returned by a sub-agent to the delegating orchestrator."""

    run_id: str
    parent_run_id: str | None = None
    agent_name: str
    goal: str
    status: Literal["success", "failed", "pending_confirmation", "budget_exceeded"]
    result: str = ""
    trace: list[AgentStepTrace] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    steps_taken: int = 0
    tokens_used: int = 0
    duration_ms: float = 0.0
    tainted: bool = False
    pending_confirmation: Confirmation | None = None


class AgentRunRecord(BaseModel):
    """Persisted record of a sub-agent delegation run for audit and observability."""

    run_id: str
    parent_run_id: str | None = None
    agent_name: str
    goal: str
    status: str
    tools_used: list[str] = Field(default_factory=list)
    budget: AgentBudget
    steps_taken: int = 0
    tokens_used: int = 0
    duration_ms: float = 0.0
    tainted: bool = False
    result: str = ""
    has_pending_confirmation: bool = False
    pending_confirmation_id: str | None = None
    trace: list[AgentStepTrace] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
