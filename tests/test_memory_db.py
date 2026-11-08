"""Tests for SQLite database schema, encryption, and conversation store."""

from pathlib import Path
import pytest
from cryptography.fernet import Fernet

from nino.memory.db import MemoryDB
from nino.memory.conversation import ConversationStore


class MockLLM:
    async def chat(self, messages, **kwargs):
        return "User discussed project priorities and tasks."


def test_db_initialization_and_migrations(tmp_path: Path):
    db_file = tmp_path / "nino_test.db"
    db = MemoryDB(db_path=str(db_file))

    with db.get_connection() as conn:
        cursor = conn.execute("SELECT version FROM schema_migrations")
        row = cursor.fetchone()
        assert row["version"] == 1

        # Verify tables exist
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "sessions" in tables
        assert "messages" in tables
        assert "facts" in tables
        assert "documents" in tables
        assert "document_chunks" in tables
        assert "vector_entries" in tables


def test_db_fernet_encryption(tmp_path: Path):
    key = Fernet.generate_key().decode("utf-8")
    db_file = tmp_path / "nino_enc.db"
    db = MemoryDB(db_path=str(db_file), encryption_key=key)

    plaintext = "Sensitive private fact: Alice loves quantum computing."
    encrypted = db.encrypt_text(plaintext)
    assert encrypted.startswith("enc:")
    assert plaintext not in encrypted

    decrypted = db.decrypt_text(encrypted)
    assert decrypted == plaintext


def test_conversation_store_sessions_and_messages(tmp_path: Path):
    db_file = tmp_path / "conv_test.db"
    db = MemoryDB(db_path=str(db_file))
    store = ConversationStore(memory_db=db)

    sess_id = store.create_session("Test Session")
    assert sess_id is not None

    store.add_message(sess_id, "user", "Hello Nino")
    store.add_message(sess_id, "assistant", "Hello! How can I assist?")

    msgs = store.get_messages(sess_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello Nino"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Hello! How can I assist?"


@pytest.mark.asyncio
async def test_conversation_store_rolling_summarization(tmp_path: Path):
    db_file = tmp_path / "budget_test.db"
    db = MemoryDB(db_path=str(db_file))
    store = ConversationStore(memory_db=db)
    sess_id = store.create_session("Budget Test")

    # Add several long messages exceeding a tiny token budget of 60 tokens
    for i in range(8):
        store.add_message(sess_id, "user", f"Turn {i}: This is a fairly detailed statement about topic {i}.")
        store.add_message(sess_id, "assistant", f"Turn {i}: I have recorded details for topic {i}.")

    mock_adapter = MockLLM()
    budgeted = await store.get_budgeted_messages(session_id=sess_id, token_budget=50, adapter=mock_adapter)

    # Must contain a summary message from older turns and only recent messages
    assert len(budgeted) < 16
    roles = [m["role"] for m in budgeted]
    assert "system" in roles
    assert any("[Previous Conversation Summary]" in m["content"] for m in budgeted)
