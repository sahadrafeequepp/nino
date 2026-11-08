"""Conversation store managing sessions, message history, and rolling summarization."""

import datetime
import uuid
from typing import Any

from nino.memory.db import MemoryDB
from nino.models.base import LLMAdapter


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (~4 characters per token)."""
    return max(1, len(text) // 4)


class ConversationStore:
    """Manages conversational session lifecycle, message logs, and rolling summarization."""

    def __init__(self, memory_db: MemoryDB):
        self.memory_db = memory_db

    def create_session(self, title: str | None = None, user_id: str = "admin") -> str:
        """Create a new conversational session and return its ID."""
        session_id = str(uuid.uuid4())
        session_title = title or f"Session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        with self.memory_db.get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, user_id) VALUES (?, ?, ?)",
                (session_id, session_title, user_id),
            )
        return session_id

    def list_sessions(self, limit: int = 50, user_id: str | None = None) -> list[dict[str, Any]]:
        """List recent conversation sessions with message count, optionally scoped to user."""
        with self.memory_db.get_connection() as conn:
            if user_id and user_id != "admin":
                cur = conn.execute(
                    """
                    SELECT s.id, s.title, s.created_at, s.updated_at, s.user_id,
                           COALESCE(u.username, CASE WHEN s.user_id = 'admin' OR s.user_id IS NULL OR s.user_id = '' THEN 'admin' ELSE s.user_id END) AS username,
                           COUNT(m.id) as message_count
                    FROM sessions s
                    LEFT JOIN users u ON s.user_id = u.id
                    LEFT JOIN messages m ON s.id = m.session_id
                    WHERE s.user_id = ?
                    GROUP BY s.id
                    ORDER BY s.updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                )
            else:
                # Admin or unscoped: return all sessions labelled with their author username
                cur = conn.execute(
                    """
                    SELECT s.id, s.title, s.created_at, s.updated_at, s.user_id,
                           COALESCE(u.username, CASE WHEN s.user_id = 'admin' OR s.user_id IS NULL OR s.user_id = '' THEN 'admin' ELSE s.user_id END) AS username,
                           COUNT(m.id) as message_count
                    FROM sessions s
                    LEFT JOIN users u ON s.user_id = u.id
                    LEFT JOIN messages m ON s.id = m.session_id
                    GROUP BY s.id
                    ORDER BY s.updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def get_latest_session(self, require_messages: bool = True, user_id: str | None = None) -> dict[str, Any] | None:
        """Return the most recently updated session, prioritizing sessions with saved messages and scoped to user."""
        with self.memory_db.get_connection() as conn:
            user_clause = ""
            params: list[Any] = []
            if user_id and user_id != "admin":
                user_clause = "WHERE s.user_id = ?"
                params.append(user_id)

            if require_messages:
                query = f"""
                    SELECT s.id, s.title, s.created_at, s.updated_at, s.summary, s.user_id,
                           COALESCE(u.username, CASE WHEN s.user_id = 'admin' OR s.user_id IS NULL OR s.user_id = '' THEN 'admin' ELSE s.user_id END) AS username,
                           COUNT(m.id) as message_count
                    FROM sessions s
                    LEFT JOIN users u ON s.user_id = u.id
                    JOIN messages m ON s.id = m.session_id
                    {user_clause}
                    GROUP BY s.id
                    ORDER BY s.updated_at DESC
                    LIMIT 1
                """
                cur = conn.execute(query, tuple(params))
                row = cur.fetchone()
                if row:
                    res = dict(row)
                    if res.get("summary"):
                        res["summary"] = self.memory_db.decrypt_text(res["summary"])
                    return res

            # Fallback to any session if none have messages yet
            sessions = self.list_sessions(limit=1, user_id=user_id)
            return sessions[0] if sessions else None

    def delete_session(self, session_id: str, user_id: str | None = None) -> bool:
        """Delete a conversation session and all its messages, checking user ownership."""
        session = self.get_session(session_id, user_id=user_id)
        if not session:
            return False
        with self.memory_db.get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0

    def get_session(self, session_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        """Get session metadata and message count, enforcing user isolation."""
        with self.memory_db.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at, s.summary, s.user_id,
                       COALESCE(u.username, CASE WHEN s.user_id = 'admin' OR s.user_id IS NULL OR s.user_id = '' THEN 'admin' ELSE s.user_id END) AS username,
                       COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN users u ON s.user_id = u.id
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.id = ?
                GROUP BY s.id
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            if user_id and user_id != "admin":
                if res.get("user_id") != user_id:
                    return None
            if res.get("summary"):
                res["summary"] = self.memory_db.decrypt_text(res["summary"])
            return res

    def update_session_title(self, session_id: str, title: str) -> None:
        """Update session title/name."""
        with self.memory_db.get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, session_id),
            )

    def ensure_session(self, session_id: str | None, user_id: str = "admin") -> str:
        """Ensure session exists or create a new one bound to user_id."""
        if session_id:
            with self.memory_db.get_connection() as conn:
                cur = conn.execute("SELECT id, user_id FROM sessions WHERE id = ?", (session_id,))
                row = cur.fetchone()
                if row:
                    # If owned by user or caller is admin, retain
                    if user_id == "admin" or row["user_id"] == user_id:
                        return session_id
                    # Belonging to another user: isolate and create new session!
                    return self.create_session(user_id=user_id)
                conn.execute(
                    "INSERT INTO sessions (id, title, user_id) VALUES (?, ?, ?)",
                    (session_id, f"Session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", user_id),
                )
                return session_id
        return self.create_session(user_id=user_id)

    def add_message(self, session_id: str, role: str, content: str) -> str:
        """Persist a message into the session history."""
        msg_id = str(uuid.uuid4())
        toks = estimate_tokens(content)
        encrypted_content = self.memory_db.encrypt_text(content)

        with self.memory_db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, token_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (msg_id, session_id, role, encrypted_content, toks),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
        return msg_id

    def get_messages(self, session_id: str, user_id: str | None = None) -> list[dict[str, Any]]:
        """Retrieve all decrypted messages for a session, checking user ownership if user_id is provided."""
        if user_id and user_id != "admin":
            session = self.get_session(session_id, user_id=user_id)
            if not session:
                return []
        with self.memory_db.get_connection() as conn:
            cur = conn.execute(
                "SELECT id, role, content, timestamp, token_count FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            )
            rows = cur.fetchall()

        messages = []
        for r in rows:
            raw_content = self.memory_db.decrypt_text(r["content"])
            messages.append(
                {
                    "id": r["id"],
                    "role": r["role"],
                    "content": raw_content,
                    "timestamp": r["timestamp"],
                    "token_count": r["token_count"],
                }
            )
        return messages

    async def get_budgeted_messages(
        self,
        session_id: str,
        token_budget: int = 2000,
        adapter: LLMAdapter | None = None,
    ) -> list[dict[str, str]]:
        """Retrieve recent conversation messages within token budget, rolling older ones into a summary."""
        all_messages = self.get_messages(session_id)
        if not all_messages:
            return []

        # Fetch existing summary
        existing_summary = ""
        with self.memory_db.get_connection() as conn:
            cur = conn.execute("SELECT summary FROM sessions WHERE id = ?", (session_id,))
            row = cur.fetchone()
            if row and row["summary"]:
                existing_summary = self.memory_db.decrypt_text(row["summary"])

        total_tokens = sum(m["token_count"] for m in all_messages)
        if total_tokens <= token_budget and not existing_summary:
            return [{"role": m["role"], "content": m["content"]} for m in all_messages]

        # Select recent messages that fit into budget
        reserved_summary_tokens = 250 if existing_summary else 0
        usable_budget = max(10, token_budget - reserved_summary_tokens)

        recent_messages: list[dict[str, str]] = []
        accumulated_tokens = 0
        cutoff_index = 0

        for idx in range(len(all_messages) - 1, -1, -1):
            msg = all_messages[idx]
            if accumulated_tokens + msg["token_count"] > usable_budget and recent_messages:
                cutoff_index = idx + 1
                break
            accumulated_tokens += msg["token_count"]
            recent_messages.insert(0, {"role": msg["role"], "content": msg["content"]})

        # Summarize older messages if adapter is provided
        older_messages = all_messages[:cutoff_index]
        if older_messages and adapter:
            conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in older_messages)
            summary_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a summarizer. Provide a concise, factual summary of the following prior conversation. "
                        "Preserve key user details, requests, and assistant context in 2-4 sentences."
                    ),
                },
                {"role": "user", "content": f"Prior context:\n{existing_summary}\n\nOlder conversation:\n{conversation_text}"},
            ]
            try:
                new_summary = await adapter.chat(summary_prompt)
                encrypted_summary = self.memory_db.encrypt_text(new_summary.strip())
                with self.memory_db.get_connection() as conn:
                    conn.execute("UPDATE sessions SET summary = ? WHERE id = ?", (encrypted_summary, session_id))
                existing_summary = new_summary.strip()
            except Exception:
                # If summarization fails, keep existing summary
                pass

        final_prompt: list[dict[str, str]] = []
        if existing_summary:
            final_prompt.append(
                {
                    "role": "system",
                    "content": f"[Previous Conversation Summary]: {existing_summary}",
                }
            )
        final_prompt.extend(recent_messages)
        return final_prompt
