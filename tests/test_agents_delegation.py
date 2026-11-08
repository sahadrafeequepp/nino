"""Tests for SubAgentRunner delegation protocol, security boundaries, budgets, and taint propagation."""

import asyncio
from pathlib import Path
from typing import Any
import pytest

from nino.agents.models import AgentBudget, AgentDefinition
from nino.agents.registry import AgentRegistry
from nino.agents.runner import SubAgentRunner
from nino.agents.store import DelegationRunStore
from nino.memory.db import MemoryDB
from nino.models.base import LLMAdapter
from nino.security.audit import AuditLogger
from nino.tools.base import BaseTool
from nino.tools.confirmation import ConfirmationManager
from nino.tools.executor import CentralExecutor
from nino.tools.permissions import PermissionTier
from nino.tools.registry import ToolRegistry
from nino.tools.taint import TaintTracker
from pydantic import BaseModel, Field


class DummyInput(BaseModel):
    query: str = Field(default="")


class SafeReadTool(BaseTool):
    name = "safe_read"
    description = "A safe read-only tool"
    args_schema = DummyInput
    tier = PermissionTier.READ_ONLY

    async def run(self, query: str = "") -> str:
        return f"Safe content for {query}"


class SensitiveActionTool(BaseTool):
    name = "sensitive_action"
    description = "A sensitive mutation tool"
    args_schema = DummyInput
    tier = PermissionTier.SENSITIVE

    async def run(self, query: str = "") -> str:
        return f"Executed sensitive action {query}"


class UntrustedWebTool(BaseTool):
    name = "untrusted_web"
    description = "Fetches external web content"
    args_schema = DummyInput
    tier = PermissionTier.READ_ONLY

    async def run(self, query: str = "") -> str:
        raw = f"External web page content for {query}"
        return TaintTracker.wrap_untrusted(raw, source="web")


class MockAdapter(LLMAdapter):
    """Mock LLM adapter returning pre-configured turns or tool calls."""

    def __init__(self, responses: list[Any] | None = None):
        self.responses = list(responses) if responses else []
        self.call_history: list[list[dict[str, Any]]] = []

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.call_history.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return "I have completed the task."

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any):
        yield await self.chat(messages, **kwargs)


@pytest.fixture
def test_setup(tmp_path):
    db_file = tmp_path / "agent_test.db"
    audit_file = tmp_path / "audit.jsonl"

    memory_db = MemoryDB(str(db_file))
    conf_mgr = ConfirmationManager(memory_db=memory_db)
    audit_logger = AuditLogger(log_path=str(audit_file))

    global_tools = ToolRegistry()
    global_tools.register(SafeReadTool())
    global_tools.register(SensitiveActionTool())
    global_tools.register(UntrustedWebTool())

    executor = CentralExecutor(
        registry=global_tools,
        confirmation_manager=conf_mgr,
        audit_logger=audit_logger,
        allow_restricted_tools=True,
    )

    agent_reg = AgentRegistry("persona/agents.yaml")
    # Register test agent with only safe_read and untrusted_web
    agent_reg.register(
        AgentDefinition(
            name="test_worker",
            description="Worker with restricted tools",
            allowed_tools=["safe_read", "untrusted_web"],
            max_tier="READ_ONLY",
            system_prompt="You are a test worker.",
        )
    )
    # Register sensitive worker
    agent_reg.register(
        AgentDefinition(
            name="test_operator",
            description="Operator with sensitive tool",
            allowed_tools=["sensitive_action"],
            max_tier="SENSITIVE",
            system_prompt="You are a test operator.",
        )
    )

    run_store = DelegationRunStore()
    runner = SubAgentRunner(
        agent_registry=agent_reg,
        global_tool_registry=global_tools,
        executor=executor,
        audit_logger=audit_logger,
        run_store=run_store,
    )

    return {
        "runner": runner,
        "conf_mgr": conf_mgr,
        "run_store": run_store,
        "audit_file": audit_file,
    }


@pytest.mark.asyncio
async def test_depth_limit_enforced(test_setup):
    runner: SubAgentRunner = test_setup["runner"]
    adapter = MockAdapter(["done"])

    # Attempt delegation with depth=2
    res = await runner.run(
        agent_name="test_worker",
        goal="Run something nested",
        adapter=adapter,
        depth=2,
    )

    assert res.status == "failed"
    assert "depth (1) exceeded" in res.result


@pytest.mark.asyncio
async def test_tool_allowlist_restriction(test_setup):
    runner: SubAgentRunner = test_setup["runner"]

    # test_worker only has ['safe_read', 'untrusted_web'].
    # Have model attempt to call 'sensitive_action' which is disallowed for test_worker!
    disallowed_call = '```json\n{"tool": "sensitive_action", "arguments": {"query": "hack"}}\n```'
    final_reply = "I was unable to run sensitive_action, here is the answer."
    adapter = MockAdapter([disallowed_call, final_reply])

    res = await runner.run(
        agent_name="test_worker",
        goal="Try to run a disallowed action",
        adapter=adapter,
    )

    assert res.status == "success"
    # Verify sensitive_action was blocked and NOT executed
    assert "sensitive_action" not in res.tools_used

    # Check that error was recorded in trace
    blocked_trace = [t for t in res.trace if t.tool_name == "sensitive_action"]
    assert len(blocked_trace) == 1
    assert "Security Block" in blocked_trace[0].error
    assert "unauthorized tool" in blocked_trace[0].error


@pytest.mark.asyncio
async def test_budget_step_limit_enforced(test_setup):
    runner: SubAgentRunner = test_setup["runner"]

    # Infinite loop of calling safe_read
    call_turn = '```json\n{"tool": "safe_read", "arguments": {"query": "loop"}}\n```'
    adapter = MockAdapter([call_turn, call_turn, call_turn, call_turn, call_turn])

    budget = AgentBudget(max_steps=3, timeout_seconds=30.0)
    res = await runner.run(
        agent_name="test_worker",
        goal="Loop forever",
        adapter=adapter,
        budget=budget,
    )

    assert res.status == "budget_exceeded"
    assert res.steps_taken == 3
    assert "exceeded step budget" in res.result


@pytest.mark.asyncio
async def test_budget_timeout_enforced(test_setup):
    runner: SubAgentRunner = test_setup["runner"]

    class SlowAdapter(LLMAdapter):
        async def chat(self, messages, **kwargs):
            await asyncio.sleep(0.3)
            return '```json\n{"tool": "safe_read", "arguments": {"query": "slow"}}\n```'

        async def stream_chat(self, messages, **kwargs):
            yield await self.chat(messages, **kwargs)

    slow_adapter = SlowAdapter()
    budget = AgentBudget(max_steps=10, timeout_seconds=0.2)
    res = await runner.run(
        agent_name="test_worker",
        goal="Run slowly",
        adapter=slow_adapter,
        budget=budget,
    )

    assert res.status == "budget_exceeded"
    assert "timed out" in res.result


@pytest.mark.asyncio
async def test_confirmation_bubbling_to_user(test_setup):
    runner: SubAgentRunner = test_setup["runner"]
    conf_mgr: ConfirmationManager = test_setup["conf_mgr"]

    # test_operator calls sensitive_action (tier=SENSITIVE)
    call_turn = '```json\n{"tool": "sensitive_action", "arguments": {"query": "transfer_funds"}}\n```'
    adapter = MockAdapter([call_turn])

    res = await runner.run(
        agent_name="test_operator",
        goal="Run sensitive operation",
        adapter=adapter,
    )

    # Sub-agent MUST pause with pending_confirmation and CANNOT auto-approve
    assert res.status == "pending_confirmation"
    assert res.pending_confirmation is not None
    assert res.pending_confirmation.tool_name == "sensitive_action"
    assert res.pending_confirmation.tier == "SENSITIVE"
    conf_id = res.pending_confirmation.id

    # Now resume by providing the valid confirmation_id
    success_turn = "Sensitive action succeeded."
    resume_adapter = MockAdapter([call_turn, success_turn])
    res_resumed = await runner.run(
        agent_name="test_operator",
        goal="Run sensitive operation",
        adapter=resume_adapter,
        confirmation_id=conf_id,
    )

    assert res_resumed.status == "success"
    assert "sensitive_action" in res_resumed.tools_used


@pytest.mark.asyncio
async def test_taint_propagation(test_setup):
    runner: SubAgentRunner = test_setup["runner"]

    # Call untrusted_web tool which returns taint-wrapped content
    web_call = '```json\n{"tool": "untrusted_web", "arguments": {"query": "ai news"}}\n```'
    synthesis = "Based on web search, AI is advancing rapidly."
    adapter = MockAdapter([web_call, synthesis])

    res = await runner.run(
        agent_name="test_worker",
        goal="Search the web",
        adapter=adapter,
    )

    assert res.status == "success"
    assert res.tainted is True
    # The return payload must be wrapped in untrusted delimiters
    assert TaintTracker.is_tainted(res.result)
    assert 'source="subagent:test_worker"' in res.result


@pytest.mark.asyncio
async def test_context_minimization(test_setup):
    runner: SubAgentRunner = test_setup["runner"]
    adapter = MockAdapter(["Summary of specific task"])

    goal = "Investigate error 404"
    explicit_context = "Log snippet: 404 on /auth"

    res = await runner.run(
        agent_name="test_worker",
        goal=goal,
        context=explicit_context,
        adapter=adapter,
    )

    assert res.status == "success"
    # Inspect messages received by the model adapter
    assert len(adapter.call_history) == 1
    passed_messages = adapter.call_history[0]

    # Exactly 2 messages: system prompt + user prompt (with goal and explicit context only)
    assert len(passed_messages) == 2
    assert passed_messages[0]["role"] == "system"
    assert passed_messages[1]["role"] == "user"
    assert goal in passed_messages[1]["content"]
    assert explicit_context in passed_messages[1]["content"]
