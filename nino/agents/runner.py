"""SubAgentRunner: Executes specialized sub-agents with strict containment and budget limits."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from nino.agents.models import (
    AgentBudget,
    AgentDefinition,
    AgentRunRecord,
    AgentStepTrace,
    DelegationResult,
)
from nino.agents.registry import AgentRegistry
from nino.agents.store import DelegationRunStore, get_run_store
from nino.models.base import LLMAdapter
from nino.security.audit import AuditLogger
from nino.tools.base import BaseTool
from nino.tools.executor import CentralExecutor, ToolExecutionResult
from nino.tools.permissions import PermissionTier
from nino.tools.registry import ToolRegistry
from nino.tools.taint import TaintTracker

logger = logging.getLogger("nino.agents.runner")

TIER_ORDER = {
    PermissionTier.READ_ONLY.value: 1,
    PermissionTier.REVERSIBLE.value: 2,
    PermissionTier.SENSITIVE.value: 3,
    PermissionTier.RESTRICTED.value: 4,
}


class SubAgentRunner:
    """Executes a specialized sub-agent with strict tool allowlists, budgets, and security boundaries."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        global_tool_registry: ToolRegistry,
        executor: CentralExecutor,
        audit_logger: AuditLogger | None = None,
        run_store: DelegationRunStore | None = None,
    ):
        self.agent_registry = agent_registry
        self.global_tool_registry = global_tool_registry
        self.executor = executor
        self.audit_logger = audit_logger
        self.run_store = run_store or get_run_store()

    async def run(
        self,
        agent_name: str,
        goal: str,
        adapter: LLMAdapter,
        context: str | None = None,
        budget: AgentBudget | None = None,
        parent_run_id: str | None = None,
        depth: int = 1,
        confirmation_id: str | None = None,
    ) -> DelegationResult:
        """Execute a delegated task through a specialized sub-agent.

        Args:
            agent_name: Name of the registered sub-agent (e.g. 'researcher', 'coder', 'planner').
            goal: Explicit goal or question for the sub-agent.
            adapter: LLM adapter for generating completions.
            context: Minimal specific context needed for this task (no global history).
            budget: Optional budget limits overriding agent defaults.
            parent_run_id: Traceability ID linking this run to the caller request.
            depth: Delegation depth (depth > 1 is strictly prohibited).
            confirmation_id: Optional user-provided token to resume a pending confirmation.

        Returns:
            DelegationResult containing outcome, status, trace, and taint metadata.
        """
        run_id = f"agent_run_{uuid.uuid4().hex[:12]}"
        start_time = time.perf_counter()

        # 1. Enforce strict depth limit (Maximum delegation depth = 1)
        if depth > 1:
            err_msg = (
                f"Delegation rejected: Maximum delegation depth (1) exceeded (attempted depth: {depth}). "
                "Sub-agents are strictly prohibited from spawning other sub-agents."
            )
            logger.warning(err_msg)
            if self.audit_logger:
                self.audit_logger.log(
                    endpoint=f"/agent/{agent_name}/delegate",
                    method="DELEGATE",
                    status_code=403,
                    duration_ms=0.0,
                    extra={"error": "depth_limit_exceeded", "depth": depth, "parent_run_id": parent_run_id},
                )
            return DelegationResult(
                run_id=run_id,
                parent_run_id=parent_run_id,
                agent_name=agent_name,
                goal=goal,
                status="failed",
                result=err_msg,
            )

        # 2. Retrieve agent definition
        agent_def = self.agent_registry.get(agent_name)
        if not agent_def:
            err_msg = f"Unknown sub-agent '{agent_name}'. Registered agents: {[a.name for a in self.agent_registry.list_agents()]}"
            logger.error(err_msg)
            return DelegationResult(
                run_id=run_id,
                parent_run_id=parent_run_id,
                agent_name=agent_name,
                goal=goal,
                status="failed",
                result=err_msg,
            )

        # 3. Establish execution budget
        effective_budget = budget or agent_def.default_budget
        max_steps = effective_budget.max_steps
        timeout_seconds = effective_budget.timeout_seconds

        # 4. Build scoped tool registry filtered strictly to agent_def.allowed_tools
        scoped_registry = ToolRegistry()
        allowed_set = set(agent_def.allowed_tools)
        # Exclude delegation tool from all sub-agents (invariant)
        allowed_set.discard("delegate_task")

        for tool_name in allowed_set:
            tool_instance = self.global_tool_registry.get(tool_name)
            if tool_instance:
                scoped_registry.register(tool_instance)
            else:
                logger.debug("Tool '%s' in agent '%s' allowed list is not active in global registry.", tool_name, agent_name)

        # 5. Build minimal context messages (Zero global conversation store / zero facts)
        system_content = (
            f"{agent_def.system_prompt.strip()}\n\n"
            f"AVAILABLE TOOLS:\n{scoped_registry.get_system_prompt_spec()}\n\n"
            "INSTRUCTIONS:\n"
            "- If you need to use a tool, respond with ONLY a JSON block:\n"
            "```json\n"
            '{\n  "tool": "tool_name",\n  "arguments": {\n    "arg_name": "arg_value"\n  }\n}\n'
            "```\n"
            "- If no tools are required or you have obtained all necessary answers, provide your final response in clear text."
        )

        user_content = f"Goal: {goal}"
        if context:
            user_content += f"\n\nContext:\n{context}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        # Tracking state
        traces: list[AgentStepTrace] = []
        tools_used: list[str] = []
        is_tainted = TaintTracker.is_tainted(context) if context else False
        steps_taken = 0
        total_tokens_est = 0
        status = "success"
        final_result = ""
        pending_conf = None

        # 6. Execution Loop
        while steps_taken < max_steps:
            steps_taken += 1
            step_start = time.perf_counter()

            # Check timeout
            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout_seconds:
                status = "budget_exceeded"
                final_result = f"Sub-agent '{agent_name}' timed out after {elapsed:.1f}s (budget: {timeout_seconds}s)."
                break

            # Call LLM with scoped tools
            ollama_tools = scoped_registry.to_ollama_tools()
            model_kwargs = {"return_raw_message": True}
            if agent_def.model:
                model_kwargs["model"] = agent_def.model
            if ollama_tools:
                model_kwargs["tools"] = ollama_tools

            try:
                raw_response = await adapter.chat(messages, **model_kwargs)
            except Exception as e:
                # Retry without native tools if adapter fails
                if "tools" in model_kwargs:
                    del model_kwargs["tools"]
                    raw_response = await adapter.chat(messages, **model_kwargs)
                else:
                    status = "failed"
                    final_result = f"Model execution error: {e}"
                    break

            if isinstance(raw_response, dict):
                content = raw_response.get("content", "") or ""
                native_tool_calls = raw_response.get("tool_calls", [])
            else:
                content = str(raw_response)
                native_tool_calls = []

            total_tokens_est += len(content) // 4

            # Parse tool call
            tool_name: str | None = None
            tool_args: dict[str, Any] = {}

            if native_tool_calls and len(native_tool_calls) > 0:
                fn = native_tool_calls[0].get("function", {})
                tool_name = fn.get("name")
                raw_args = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args)
                    except Exception:
                        tool_args = {}
                elif isinstance(raw_args, dict):
                    tool_args = raw_args
            else:
                # Fallback parse from content
                parsed_json = self._extract_tool_json(content)
                if parsed_json and "tool" in parsed_json:
                    tool_name = parsed_json.get("tool")
                    tool_args = parsed_json.get("arguments") or {}

            # If no tool called, agent formulated its final text answer
            if not tool_name:
                final_result = content
                traces.append(
                    AgentStepTrace(
                        step=steps_taken,
                        thought=content,
                        duration_ms=(time.perf_counter() - step_start) * 1000,
                        is_tainted=is_tainted,
                    )
                )
                break

            # 7. Strict Tool Allowlist & Containment Verification
            if tool_name not in allowed_set:
                err_msg = (
                    f"Security Block: Sub-agent '{agent_name}' attempted to invoke unauthorized tool '{tool_name}'. "
                    f"Permitted tools: {sorted(list(allowed_set))}."
                )
                logger.warning(err_msg)
                if self.audit_logger:
                    self.audit_logger.log(
                        endpoint=f"/agent/{agent_name}/tool/{tool_name}",
                        method="EXEC",
                        status_code=403,
                        duration_ms=(time.perf_counter() - step_start) * 1000,
                        extra={
                            "decision": "blocked_disallowed_tool",
                            "agent": agent_name,
                            "attempted_tool": tool_name,
                            "parent_run_id": parent_run_id,
                        },
                    )

                traces.append(
                    AgentStepTrace(
                        step=steps_taken,
                        thought=content,
                        tool_name=tool_name,
                        arguments=tool_args,
                        error=err_msg,
                        duration_ms=(time.perf_counter() - step_start) * 1000,
                        is_tainted=is_tainted,
                    )
                )
                # Inform agent of policy violation so it can correct its course or terminate
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"[Tool Error] {err_msg}"})
                continue

            # 8. Check Permission Tier Containment
            target_tool = self.global_tool_registry.get(tool_name)
            if target_tool:
                tool_tier = target_tool.tier.value
                agent_max_tier = agent_def.max_tier
                if TIER_ORDER.get(tool_tier, 99) > TIER_ORDER.get(agent_max_tier, 0):
                    err_msg = (
                        f"Security Block: Tool '{tool_name}' tier [{tool_tier}] exceeds "
                        f"agent '{agent_name}' maximum tier [{agent_max_tier}]."
                    )
                    logger.warning(err_msg)
                    traces.append(
                        AgentStepTrace(
                            step=steps_taken,
                            thought=content,
                            tool_name=tool_name,
                            arguments=tool_args,
                            error=err_msg,
                            duration_ms=(time.perf_counter() - step_start) * 1000,
                            is_tainted=is_tainted,
                        )
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"[Tool Error] {err_msg}"})
                    continue

            # 9. Execute tool through CentralExecutor
            tools_used.append(tool_name)
            exec_res: ToolExecutionResult = await self.executor.execute(
                tool_name=tool_name,
                arguments=tool_args,
                context=messages,
                confirmation_id=confirmation_id,
            )

            step_duration_ms = (time.perf_counter() - step_start) * 1000

            # 10. Confirmation Bubbling: Sub-agent CANNOT auto-approve
            if exec_res.requires_confirmation and exec_res.confirmation:
                status = "pending_confirmation"
                pending_conf = exec_res.confirmation
                final_result = (
                    f"Sub-agent '{agent_name}' action requires human confirmation:\n\n"
                    f"**Tool**: `{pending_conf.tool_name}` [{pending_conf.tier}]\n"
                    f"**Action**: {pending_conf.summary}\n"
                    f"**Confirmation ID**: `{pending_conf.id}`"
                )
                traces.append(
                    AgentStepTrace(
                        step=steps_taken,
                        thought=content,
                        tool_name=tool_name,
                        arguments=tool_args,
                        error=f"Confirmation required: {pending_conf.id}",
                        duration_ms=step_duration_ms,
                        is_tainted=is_tainted,
                    )
                )
                break

            # 11. Taint Propagation
            output_str = str(exec_res.output)
            if TaintTracker.is_tainted(output_str):
                is_tainted = True

            traces.append(
                AgentStepTrace(
                    step=steps_taken,
                    thought=content,
                    tool_name=tool_name,
                    arguments=tool_args,
                    output=exec_res.output,
                    error=exec_res.error,
                    duration_ms=step_duration_ms,
                    is_tainted=is_tainted,
                )
            )

            # Prepare feedback message for the agent
            result_payload = (
                json.dumps(exec_res.output)
                if not isinstance(exec_res.output, str)
                else exec_res.output
            )
            if not exec_res.success:
                result_payload = f"Tool error: {exec_res.error}"

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"[Tool Result for '{tool_name}']\n{result_payload}"})

            # Single-use confirmation consumed
            confirmation_id = None

        if steps_taken >= max_steps and status == "success" and not final_result:
            status = "budget_exceeded"
            final_result = f"Sub-agent '{agent_name}' exceeded step budget ({max_steps} steps)."

        total_duration_ms = (time.perf_counter() - start_time) * 1000

        # 12. Taint output wrapping
        if is_tainted and final_result and not TaintTracker.is_tainted(final_result):
            final_result = TaintTracker.wrap_untrusted(final_result, source=f"subagent:{agent_name}")

        result_obj = DelegationResult(
            run_id=run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            goal=goal,
            status=status,
            result=final_result,
            trace=traces,
            tools_used=list(dict.fromkeys(tools_used)),
            steps_taken=steps_taken,
            tokens_used=total_tokens_est,
            duration_ms=total_duration_ms,
            tainted=is_tainted,
            pending_confirmation=pending_conf,
        )

        # 13. Persist run record for observability
        record = AgentRunRecord(
            run_id=run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            goal=goal,
            status=status,
            tools_used=result_obj.tools_used,
            budget=effective_budget,
            steps_taken=steps_taken,
            tokens_used=total_tokens_est,
            duration_ms=total_duration_ms,
            tainted=is_tainted,
            result=final_result,
            has_pending_confirmation=pending_conf is not None,
            pending_confirmation_id=pending_conf.id if pending_conf else None,
            trace=traces,
            created_at=time.time(),
        )
        self.run_store.record_run(record)

        # 14. Audit log completion
        if self.audit_logger:
            self.audit_logger.log(
                endpoint=f"/agent/{agent_name}/run",
                method="RUN",
                status_code=200 if status in ("success", "pending_confirmation") else 500,
                duration_ms=total_duration_ms,
                extra={
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "agent": agent_name,
                    "status": status,
                    "steps": steps_taken,
                    "tools_used": result_obj.tools_used,
                    "tainted": is_tainted,
                },
            )

        return result_obj

    def _extract_tool_json(self, text: str) -> dict[str, Any] | None:
        """Extract JSON tool call dictionary from text content."""
        if not text:
            return None

        if "```" in text:
            parts = text.split("```")
            for i in range(1, len(parts), 2):
                block = parts[i]
                if block.startswith("json"):
                    block = block[4:]
                parsed = self._try_parse_tool_json(block.strip())
                if parsed:
                    return parsed

        return self._try_parse_tool_json(text)

    def _try_parse_tool_json(self, text: str) -> dict[str, Any] | None:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                data = json.loads(text[start_idx : end_idx + 1])
                if isinstance(data, dict) and "tool" in data:
                    return data
            except Exception:
                pass
        return None
