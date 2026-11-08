"""Delegation tool allowing Nino's orchestrator to dispatch jobs to specialized sub-agents."""

import logging
from typing import Any
from pydantic import BaseModel, Field

from nino.agents.models import DelegationResult
from nino.agents.runner import SubAgentRunner
from nino.models.base import LLMAdapter
from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier

logger = logging.getLogger("nino.tools.delegation")


class DelegateTaskInput(BaseModel):
    agent_name: str = Field(
        ...,
        description="Target sub-agent identifier (e.g. 'researcher', 'coder', 'planner')",
    )
    goal: str = Field(
        ...,
        description="Specific question or objective for the sub-agent to accomplish",
    )
    context: str | None = Field(
        default=None,
        description="Minimal task-specific context or background needed by the agent",
    )


class DelegateTaskTool(BaseTool):
    """Tool allowing Nino orchestrator to delegate work to specialized sub-agents."""

    name = "delegate_task"
    description = (
        "Delegate a focused job to a specialized sub-agent (e.g. 'researcher' for web/notes research, "
        "'coder' for code inspection/execution, 'planner' for calendar/tasks/reminders). "
        "Sub-agents run with restricted tool sets and strict safety constraints."
    )
    args_schema = DelegateTaskInput
    tier = PermissionTier.REVERSIBLE

    def __init__(
        self,
        runner: SubAgentRunner,
        adapter: LLMAdapter,
    ):
        self.runner = runner
        self.adapter = adapter

    async def run(
        self,
        agent_name: str,
        goal: str,
        context: str | None = None,
    ) -> str:
        """Execute delegation via SubAgentRunner and format result."""
        logger.info("Delegating task to sub-agent '%s': %s", agent_name, goal)

        # Depth is explicitly 1 (sub-agents cannot delegate)
        res: DelegationResult = await self.runner.run(
            agent_name=agent_name,
            goal=goal,
            adapter=self.adapter,
            context=context,
            depth=1,
        )

        if res.status == "pending_confirmation" and res.pending_confirmation:
            return (
                f"[Action Requires Confirmation for Agent '{agent_name}']\n"
                f"**Tool**: `{res.pending_confirmation.tool_name}` [{res.pending_confirmation.tier}]\n"
                f"**Summary**: {res.pending_confirmation.summary}\n"
                f"**Confirmation ID**: `{res.pending_confirmation.id}`\n\n"
                "Please approve or deny this action to proceed."
            )

        tools_used_str = ", ".join(res.tools_used) if res.tools_used else "none"
        prefix = f"[Sub-Agent '{agent_name}' Result (Status: {res.status}, Tools: {tools_used_str}, Steps: {res.steps_taken})]\n"

        return prefix + res.result
