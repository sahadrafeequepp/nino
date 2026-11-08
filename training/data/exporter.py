"""Opt-in session exporter extracting and splitting clean conversations from MemoryDB."""

import json
import logging
import random
from pathlib import Path
from typing import Any

from nino.memory.db import MemoryDB
from training.data.filters import DataCleaner

logger = logging.getLogger("training.data.exporter")


class SessionExporter:
    """Exports and splits conversations from MemoryDB only for explicitly opted-in sessions."""

    def __init__(self, memory_db: MemoryDB, cleaner: DataCleaner | None = None):
        self.memory_db = memory_db
        self.cleaner = cleaner or DataCleaner()

    def list_available_sessions(self) -> list[dict[str, Any]]:
        """List all available conversational sessions with message counts."""
        with self.memory_db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.id, s.title, s.created_at, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY s.created_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def export_opted_in_sessions(
        self,
        opted_in_session_ids: list[str],
        system_prompt: str = "You are Nino, a private, security-minded, local-first personal AI assistant.",
    ) -> list[dict[str, Any]]:
        """Extract conversations strictly for opted-in sessions and filter them.

        Args:
            opted_in_session_ids: Explicit list of session IDs approved for export.
            system_prompt: Default system instruction to prepend.

        Returns:
            List of ChatML-formatted example dictionaries.
        """
        if not opted_in_session_ids:
            logger.info("No sessions opted in for dataset export.")
            return []

        examples: list[dict[str, Any]] = []

        with self.memory_db.get_connection() as conn:
            for session_id in opted_in_session_ids:
                cursor = conn.execute(
                    """
                    SELECT role, content FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (session_id,),
                )
                rows = cursor.fetchall()
                if not rows:
                    continue

                # Group into (user, assistant) pairs or sequential dialog
                turn_messages: list[dict[str, str]] = []
                for row in rows:
                    decrypted_content = self.memory_db.decrypt_text(row["content"])
                    turn_messages.append({"role": row["role"], "content": decrypted_content})

                # Validate and clean with DataCleaner
                is_valid, cleaned_messages, drop_reason = self.cleaner.process_turn(turn_messages)
                if not is_valid:
                    logger.debug("Skipped session %s: %s", session_id, drop_reason)
                    continue

                # Prepend system prompt if not present
                final_messages = [{"role": "system", "content": system_prompt}] + cleaned_messages
                examples.append({
                    "session_id": session_id,
                    "source": "exported_session",
                    "messages": final_messages,
                })

        return examples

    @staticmethod
    def train_val_split(
        examples: list[dict[str, Any]],
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Deterministically split examples into train and validation sets."""
        if not examples:
            return [], []

        shuffled = list(examples)
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        val_size = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val_set = shuffled[:val_size]
        train_set = shuffled[val_size:]

        if not train_set and val_set:
            train_set = val_set
            val_set = []

        return train_set, val_set

    @staticmethod
    def save_jsonl(examples: list[dict[str, Any]], file_path: str | Path) -> None:
        """Save ChatML examples to a JSONL file."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        logger.info("Saved %d examples to %s", len(examples), target)
