"""Tests for tool framework: registry, executor, schemas, tiers, timeouts, secrets, and audit."""

import asyncio
import pytest
from pydantic import BaseModel, Field

from nino.memory.db import MemoryDB
from nino.security.audit import AuditLogger
from nino.tools.base import BaseTool
from nino.tools.confirmation import ConfirmationManager
from nino.tools.executor import CentralExecutor
from nino.tools.permissions import PermissionTier
from nino.tools.registry import ToolRegistry


class DummyInput(BaseModel):
    query: str = Field(..., description="A test string query")
    count: int = Field(default=1, ge=1, le=10, description="Count between 1 and 10")


class DummyReadOnlyTool(BaseTool):
    name = "dummy_read"
    description = "Dummy read-only tool"
    args_schema = DummyInput
    tier = PermissionTier.READ_ONLY

    async def run(self, query: str, count: int = 1) -> str:
        return f"Read: {query} * {count}"


class DummyReversibleTool(BaseTool):
    name = "dummy_reversible"
    description = "Dummy reversible tool"
    args_schema = DummyInput
    tier = PermissionTier.REVERSIBLE

    async def run(self, query: str, count: int = 1) -> str:
        return f"Modified: {query} * {count}"


class DummySensitiveTool(BaseTool):
    name = "dummy_sensitive"
    description = "Dummy sensitive tool"
    args_schema = DummyInput
    tier = PermissionTier.SENSITIVE

    async def run(self, query: str, count: int = 1) -> str:
        return f"Executed sensitive: {query}"


class DummyRestrictedTool(BaseTool):
    name = "dummy_restricted"
    description = "Dummy restricted tool"
    args_schema = DummyInput
    tier = PermissionTier.RESTRICTED

    async def run(self, query: str, count: int = 1) -> str:
        return f"Executed restricted: {query}"


class SecretLeakingTool(BaseTool):
    name = "leak_secret"
    description = "Tool that outputs secrets"
    args_schema = DummyInput
    tier = PermissionTier.READ_ONLY

    async def run(self, query: str, count: int = 1) -> str:
        return "Here is your key: AKIAIOSFODNN7EXAMPLE and private key: -----BEGIN PRIVATE KEY-----"


class SlowHangingTool(BaseTool):
    name = "slow_hang"
    description = "Tool that times out"
    args_schema = DummyInput
    tier = PermissionTier.READ_ONLY

    async def run(self, query: str, count: int = 1) -> str:
        await asyncio.sleep(2.0)
        return "Finished slow task"


@pytest.fixture
def framework(tmp_path):
    db_file = tmp_path / "test_framework.db"
    audit_file = tmp_path / "test_audit.jsonl"
    memory_db = MemoryDB(str(db_file))
    conf_mgr = ConfirmationManager(memory_db)
    audit_logger = AuditLogger(str(audit_file), log_content=True)

    registry = ToolRegistry()
    registry.register(DummyReadOnlyTool())
    registry.register(DummyReversibleTool())
    registry.register(DummySensitiveTool())
    registry.register(DummyRestrictedTool())
    registry.register(SecretLeakingTool())
    registry.register(SlowHangingTool())

    executor = CentralExecutor(
        registry=registry,
        confirmation_manager=conf_mgr,
        audit_logger=audit_logger,
        allow_restricted_tools=False,
        tool_timeout_seconds=0.2,
        output_max_chars=50,
        log_content=True,
    )
    return registry, conf_mgr, executor, audit_file


@pytest.mark.asyncio
async def test_schema_validation(framework):
    _, _, executor, _ = framework

    # Valid args
    res = await executor.execute("dummy_read", {"query": "hello", "count": 2})
    assert res.success is True
    assert res.output == "Read: hello * 2"

    # Missing required field
    res_err = await executor.execute("dummy_read", {"count": 2})
    assert res_err.success is False
    assert "Invalid arguments" in res_err.error

    # Value out of bounds (count > 10)
    res_bound = await executor.execute("dummy_read", {"query": "test", "count": 99})
    assert res_bound.success is False
    assert "Invalid arguments" in res_bound.error


@pytest.mark.asyncio
async def test_tier_policies(framework):
    _, conf_mgr, executor, _ = framework

    # 1. READ_ONLY runs autonomously
    res_ro = await executor.execute("dummy_read", {"query": "status"})
    assert res_ro.success is True
    assert res_ro.decision == "executed"

    # 2. REVERSIBLE runs autonomously without taint
    res_rev = await executor.execute("dummy_reversible", {"query": "new note"})
    assert res_rev.success is True
    assert res_rev.decision == "executed"

    # 3. SENSITIVE requires confirmation
    res_sens = await executor.execute("dummy_sensitive", {"query": "delete item"})
    assert res_sens.success is False
    assert res_sens.requires_confirmation is True
    assert res_sens.confirmation is not None
    assert res_sens.decision == "confirmation_required"

    # 4. RESTRICTED blocked when disabled
    res_restr = await executor.execute("dummy_restricted", {"query": "format drive"})
    assert res_restr.success is False
    assert res_restr.decision == "blocked"
    assert "RESTRICTED tier which is currently disabled" in res_restr.error


@pytest.mark.asyncio
async def test_secret_filtering_and_truncation(framework):
    _, _, executor, _ = framework

    # Secret filtering
    res_secret = await executor.execute("leak_secret", {"query": "test"})
    assert res_secret.success is True
    assert "AKIAIOSFODNN7EXAMPLE" not in str(res_secret.output)
    assert "[REDACTED_SECRET]" in str(res_secret.output)

    # Output truncation (limit set to 50 chars in fixture)
    res_long = await executor.execute("dummy_read", {"query": "A" * 100})
    assert res_long.success is True
    assert len(str(res_long.output)) < 150
    assert "Truncated" in str(res_long.output)


@pytest.mark.asyncio
async def test_timeout_and_audit_logging(framework):
    _, _, executor, audit_file = framework

    # Timeout (limit set to 0.2s, tool sleeps 2.0s)
    res_timeout = await executor.execute("slow_hang", {"query": "wait"})
    assert res_timeout.success is False
    assert res_timeout.decision == "timeout"
    assert "timed out" in res_timeout.error

    # Audit log verification
    content = audit_file.read_text(encoding="utf-8")
    assert "/tool/slow_hang" in content
    assert "timeout" in content
