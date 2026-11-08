"""SQLite notes tool for creating, searching, reading, and appending notes."""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

from nino.memory.db import MemoryDB
from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier
from nino.tools.taint import TaintTracker


class NoteCreateInput(BaseModel):
    title: str = Field(..., description="Title of the note")
    content: str = Field(..., description="Note body content")
    tags: list[str] = Field(default_factory=list, description="Optional list of tags")


class NoteCreateTool(BaseTool):
    name = "notes_create"
    description = "Create a new note in local persistent memory."
    args_schema = NoteCreateInput
    tier = PermissionTier.REVERSIBLE

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, title: str, content: str, tags: list[str] | None = None) -> dict[str, Any]:
        tags_str = ",".join(tags) if tags else ""
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO notes (title, content, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (title, content, tags_str, now, now),
            )
            note_id = cursor.lastrowid
        return {
            "status": "created",
            "note_id": note_id,
            "title": title,
            "message": f"Note '{title}' (ID: {note_id}) created successfully.",
        }


class NoteSearchInput(BaseModel):
    query: str = Field(..., description="Search query to match against note titles and contents")
    limit: int = Field(default=5, description="Maximum number of matches to return")


class NoteSearchTool(BaseTool):
    name = "notes_search"
    description = "Search existing notes by title or content."
    args_schema = NoteSearchInput
    tier = PermissionTier.READ_ONLY

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, query: str, limit: int = 5) -> str:
        like_query = f"%{query}%"
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, title, content, tags, updated_at
                FROM notes
                WHERE title LIKE ? OR content LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (like_query, like_query, limit),
            )
            rows = cursor.fetchall()

        if not rows:
            return f"No notes found matching '{query}'."

        results = []
        for r in rows:
            preview = r["content"][:150] + ("..." if len(r["content"]) > 150 else "")
            results.append(f"- Note #{r['id']}: '{r['title']}' (tags: {r['tags']})\n  Preview: {preview}")

        raw_text = "\n".join(results)
        return TaintTracker.wrap_untrusted(raw_text, source="sqlite_notes_search")


class NoteReadInput(BaseModel):
    note_id: int = Field(..., description="The integer ID of the note to read")


class NoteReadTool(BaseTool):
    name = "notes_read"
    description = "Read the full contents of a specific note by ID."
    args_schema = NoteReadInput
    tier = PermissionTier.READ_ONLY

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, note_id: int) -> str:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, title, content, tags, created_at, updated_at FROM notes WHERE id = ?",
                (note_id,),
            )
            row = cursor.fetchone()

        if not row:
            return f"Note #{note_id} not found."

        raw_text = (
            f"Title: {row['title']}\n"
            f"ID: {row['id']}\n"
            f"Tags: {row['tags']}\n"
            f"Created: {row['created_at']}\n"
            f"Updated: {row['updated_at']}\n\n"
            f"Content:\n{row['content']}"
        )
        return TaintTracker.wrap_untrusted(raw_text, source=f"sqlite_note_{note_id}")


class NoteAppendInput(BaseModel):
    note_id: int = Field(..., description="The integer ID of the note to append to")
    content: str = Field(..., description="Content to append to the note")


class NoteAppendTool(BaseTool):
    name = "notes_append"
    description = "Append content to an existing note."
    args_schema = NoteAppendInput
    tier = PermissionTier.REVERSIBLE

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, note_id: int, content: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT content FROM notes WHERE id = ?", (note_id,))
            row = cursor.fetchone()
            if not row:
                return {"status": "error", "message": f"Note #{note_id} not found."}

            existing = row["content"]
            new_content = f"{existing}\n{content}" if existing else content
            conn.execute(
                "UPDATE notes SET content = ?, updated_at = ? WHERE id = ?",
                (new_content, now, note_id),
            )

        return {
            "status": "updated",
            "note_id": note_id,
            "message": f"Successfully appended content to note #{note_id}.",
        }
