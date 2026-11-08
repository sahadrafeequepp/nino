"""Tasks and reminders tool for managing todo items."""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

from nino.memory.db import MemoryDB
from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier


class TaskCreateInput(BaseModel):
    title: str = Field(..., description="Description or title of the task")
    due_date: str | None = Field(default=None, description="Optional due date (e.g. YYYY-MM-DD or readable date)")
    priority: str = Field(default="medium", description="Priority level (low, medium, high)")


class TaskCreateTool(BaseTool):
    name = "task_create"
    description = "Create a new task or reminder."
    args_schema = TaskCreateInput
    tier = PermissionTier.REVERSIBLE

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, title: str, due_date: str | None = None, priority: str = "medium") -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tasks (title, due_date, priority, status, created_at)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (title, due_date, priority, now),
            )
            task_id = cursor.lastrowid
        return {
            "status": "created",
            "task_id": task_id,
            "title": title,
            "due_date": due_date,
            "priority": priority,
            "message": f"Task #{task_id} '{title}' created successfully.",
        }


class TaskListInput(BaseModel):
    status: str = Field(default="pending", description="Filter by status: 'pending', 'completed', or 'all'")


class TaskListTool(BaseTool):
    name = "task_list"
    description = "List existing tasks and reminders."
    args_schema = TaskListInput
    tier = PermissionTier.READ_ONLY

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, status: str = "pending") -> str:
        with self.db.get_connection() as conn:
            if status.lower() == "all":
                cursor = conn.execute("SELECT id, title, due_date, priority, status FROM tasks ORDER BY id ASC")
            else:
                cursor = conn.execute(
                    "SELECT id, title, due_date, priority, status FROM tasks WHERE status = ? ORDER BY id ASC",
                    (status.lower(),),
                )
            rows = cursor.fetchall()

        if not rows:
            return f"No tasks found with status '{status}'."

        lines = [f"Found {len(rows)} task(s):"]
        for r in rows:
            due_str = f" (Due: {r['due_date']})" if r["due_date"] else ""
            status_mark = "[x]" if r["status"] == "completed" else "[ ]"
            lines.append(f"{status_mark} #{r['id']}: {r['title']} [Priority: {r['priority']}]{due_str}")

        return "\n".join(lines)


class TaskCompleteInput(BaseModel):
    task_id: int = Field(..., description="ID of the task to mark completed")


class TaskCompleteTool(BaseTool):
    name = "task_complete"
    description = "Mark a task as completed."
    args_schema = TaskCompleteInput
    tier = PermissionTier.REVERSIBLE

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, task_id: int) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT id, title FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                return {"status": "error", "message": f"Task #{task_id} not found."}

            conn.execute(
                "UPDATE tasks SET status = 'completed', completed_at = ? WHERE id = ?",
                (now, task_id),
            )

        return {
            "status": "completed",
            "task_id": task_id,
            "title": row["title"],
            "message": f"Task #{task_id} '{row['title']}' marked as completed.",
        }


class TaskDeleteInput(BaseModel):
    task_id: int = Field(..., description="ID of the task to permanently delete")


class TaskDeleteTool(BaseTool):
    name = "task_delete"
    description = "Permanently delete a task. Requires confirmation."
    args_schema = TaskDeleteInput
    tier = PermissionTier.SENSITIVE

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, task_id: int) -> dict[str, Any]:
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT id, title FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                return {"status": "error", "message": f"Task #{task_id} not found."}

            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

        return {
            "status": "deleted",
            "task_id": task_id,
            "title": row["title"],
            "message": f"Task #{task_id} '{row['title']}' was permanently deleted.",
        }
