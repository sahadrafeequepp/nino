"""Tests for confirmation flow: argument-binding, invalidation, single-use, and expiration."""

from datetime import datetime, timezone, timedelta
import pytest

from nino.memory.db import MemoryDB
from nino.security.audit import AuditLogger
from nino.tools.base import BaseTool
from nino.tools.confirmation import ConfirmationManager, compute_arguments_hash
from nino.tools.executor import CentralExecutor
from nino.tools.permissions import PermissionTier
from nino.tools.registry import ToolRegistry
from pydantic import BaseModel, Field


class DeleteTaskInput(BaseModel):
    task_id: int = Field(..., description="ID of the task to delete")
    reason: str = Field(default="completed", description="Reason for deletion")


class MockDeleteTaskTool(BaseTool):
    name = "task_delete"
    description = "Delete a task"
    args_schema = DeleteTaskInput
    tier = PermissionTier.SENSITIVE

    async def run(self, task_id: int, reason: str = "completed") -> dict:
        return {"status": "deleted", "task_id": task_id, "reason": reason}


@pytest.fixture
def setup_conf(tmp_path):
    db_file = tmp_path / "test_conf.db"
    audit_file = tmp_path / "test_conf_audit.jsonl"
    memory_db = MemoryDB(str(db_file))
    conf_mgr = ConfirmationManager(memory_db, default_expiry_minutes=5)
    audit_logger = AuditLogger(str(audit_file))

    registry = ToolRegistry()
    registry.register(MockDeleteTaskTool())

    executor = CentralExecutor(
        registry=registry,
        confirmation_manager=conf_mgr,
        audit_logger=audit_logger,
    )
    return memory_db, conf_mgr, executor


@pytest.mark.asyncio
async def test_confirmation_creation_and_approval(setup_conf):
    _, conf_mgr, executor = setup_conf

    # 1. Calling sensitive tool requires confirmation
    args = {"task_id": 42, "reason": "obsolete"}
    res = await executor.execute("task_delete", args)
    assert res.success is False
    assert res.requires_confirmation is True
    assert res.confirmation is not None

    conf_id = res.confirmation.id
    assert res.confirmation.tool_name == "task_delete"
    assert res.confirmation.status == "pending"

    # 2. Cannot execute with unapproved confirmation
    res_unapproved = await executor.execute("task_delete", args, confirmation_id=conf_id)
    assert res_unapproved.success is False
    assert "not approved" in res_unapproved.error

    # 3. Approve confirmation
    ok, msg = conf_mgr.approve(conf_id)
    assert ok is True

    # 4. Execute with approved confirmation
    res_approved = await executor.execute("task_delete", args, confirmation_id=conf_id)
    assert res_approved.success is True
    assert res_approved.output == {"status": "deleted", "task_id": 42, "reason": "obsolete"}

    # 5. Single-use: trying to reuse the same confirmation ID fails
    res_reuse = await executor.execute("task_delete", args, confirmation_id=conf_id)
    assert res_reuse.success is False
    assert "already been used" in res_reuse.error


@pytest.mark.asyncio
async def test_changed_arguments_invalidates_confirmation(setup_conf):
    _, conf_mgr, executor = setup_conf

    original_args = {"task_id": 42, "reason": "obsolete"}
    res = await executor.execute("task_delete", original_args)
    conf_id = res.confirmation.id

    # Approve original confirmation
    conf_mgr.approve(conf_id)

    # Attack/Mutation: modify task_id from 42 to 99
    tampered_args = {"task_id": 99, "reason": "obsolete"}
    res_tampered = await executor.execute("task_delete", tampered_args, confirmation_id=conf_id)
    assert res_tampered.success is False
    assert "tool arguments have been modified" in res_tampered.error

    # Also test mutating reason field
    tampered_args2 = {"task_id": 42, "reason": "malicious injection"}
    res_tampered2 = await executor.execute("task_delete", tampered_args2, confirmation_id=conf_id)
    assert res_tampered2.success is False
    assert "tool arguments have been modified" in res_tampered2.error


@pytest.mark.asyncio
async def test_expired_confirmation_rejected(setup_conf):
    memory_db, conf_mgr, executor = setup_conf

    args = {"task_id": 1, "reason": "done"}
    res = await executor.execute("task_delete", args)
    conf_id = res.confirmation.id
    conf_mgr.approve(conf_id)

    # Simulate expiration by rolling back expires_at in DB
    past_iso = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    with memory_db.get_connection() as conn:
        conn.execute("UPDATE confirmations SET expires_at = ? WHERE id = ?", (past_iso, conf_id))

    res_expired = await executor.execute("task_delete", args, confirmation_id=conf_id)
    assert res_expired.success is False
    assert "expired" in res_expired.error


@pytest.mark.asyncio
async def test_denied_confirmation_rejected(setup_conf):
    _, conf_mgr, executor = setup_conf

    args = {"task_id": 7, "reason": "cleanup"}
    res = await executor.execute("task_delete", args)
    conf_id = res.confirmation.id

    # User explicitly denies action
    ok, msg = conf_mgr.deny(conf_id)
    assert ok is True

    res_denied = await executor.execute("task_delete", args, confirmation_id=conf_id)
    assert res_denied.success is False
    assert "denied" in res_denied.error
