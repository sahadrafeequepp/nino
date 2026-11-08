"""
Tests for Nino fine-tuning dataset builder, filters, PII scrubbers, and exporter.
"""

import json
from pathlib import Path
import pytest

from nino.memory.db import MemoryDB
from training.data.filters import (
    DataCleaner,
    PIIScrubber,
    SecretFilter,
    SensitiveMemoryFilter,
)
from training.data.exporter import SessionExporter


@pytest.fixture
def mock_memory_db(tmp_path):
    db_file = tmp_path / "test_memory.db"
    db = MemoryDB(str(db_file))

    with db.get_connection() as conn:
        # Create sessions
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES ('sess_opt_in', 'Chat 1', '2026-09-22 00:00:00', '2026-09-22 00:00:00')"
        )
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES ('sess_not_opted', 'Private Chat', '2026-09-22 00:00:00', '2026-09-22 00:00:00')"
        )

        # Messages for sess_opt_in
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES ('m1', 'sess_opt_in', 'user', 'What is your favorite color?', '2026-09-22 00:00:01')"
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES ('m2', 'sess_opt_in', 'assistant', 'I love calm shades of teal and indigo.', '2026-09-22 00:00:02')"
        )

        # Messages for sess_not_opted
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES ('m3', 'sess_not_opted', 'user', 'This is private.', '2026-09-22 00:00:01')"
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES ('m4', 'sess_not_opted', 'assistant', 'Understood.', '2026-09-22 00:00:02')"
        )
        conn.commit()

    return db


def test_secret_filter_detects_keys():
    filter_ = SecretFilter()
    assert filter_.contains_secret("My key is sk-1234567890abcdef1234567890abcdef12") is True
    assert filter_.contains_secret("AWS key: AKIAIOSFODNN7EXAMPLE") is True
    assert filter_.contains_secret("GitHub token: ghp_1234567890abcdefghijklmnopqrstuvwxyz12") is True
    assert filter_.contains_secret("Just a regular chat message here.") is False


def test_pii_scrubber_redacts_info():
    scrubber = PIIScrubber()
    text = "Reach me at test.user@example.com or call +1 555 123 4567. IP: 192.168.1.50"
    cleaned = scrubber.scrub(text)

    assert "[REDACTED_EMAIL]" in cleaned
    assert "test.user@example.com" not in cleaned
    assert "[REDACTED_PHONE]" in cleaned
    assert "[REDACTED_IP]" in cleaned


def test_sensitive_memory_filter():
    mem_filter = SensitiveMemoryFilter()
    assert mem_filter.is_sensitive("Some text", category="health") is True
    assert mem_filter.is_sensitive("Some text", category="finances") is True
    assert mem_filter.is_sensitive("Good morning Nino!", category="general") is False
    assert mem_filter.is_sensitive("I was diagnosed with flu yesterday") is True


def test_data_cleaner_integration():
    cleaner = DataCleaner()

    # Secret message should be rejected
    secret_conv = [
        {"role": "user", "content": "Here is my secret sk-1234567890abcdef1234567890abcdef12"},
        {"role": "assistant", "content": "Got it."},
    ]
    valid, _, reason = cleaner.process_turn(secret_conv)
    assert valid is False
    assert "credential" in reason

    # Sensitive medical message should be rejected
    medical_conv = [
        {"role": "user", "content": "My blood pressure prescription dosage is 20mg"},
        {"role": "assistant", "content": "Please check with your doctor."},
    ]
    valid, _, reason = cleaner.process_turn(medical_conv)
    assert valid is False
    assert "sensitive" in reason

    # Normal conversation with email should be sanitized and preserved
    normal_conv = [
        {"role": "user", "content": "Send the notes to jane@example.com please."},
        {"role": "assistant", "content": "I have prepared the notes."},
    ]
    valid, cleaned, _ = cleaner.process_turn(normal_conv)
    assert valid is True
    assert "[REDACTED_EMAIL]" in cleaned[0]["content"]
    assert "jane@example.com" not in cleaned[0]["content"]


def test_session_exporter_opt_in_only(mock_memory_db):
    exporter = SessionExporter(memory_db=mock_memory_db)

    # Exporting empty opted-in list yields nothing
    assert exporter.export_opted_in_sessions([]) == []

    # Exporting strictly opted-in session
    sessions = exporter.export_opted_in_sessions(["sess_opt_in"])
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "sess_opt_in"
    assert sessions[0]["source"] == "exported_session"
    # system + user + assistant = 3 messages
    assert len(sessions[0]["messages"]) == 3
    assert sessions[0]["messages"][0]["role"] == "system"


def test_train_val_split():
    items = [{"id": i} for i in range(10)]
    train, val = SessionExporter.train_val_split(items, val_ratio=0.2, seed=42)
    assert len(train) == 8
    assert len(val) == 2
    assert set(t["id"] for t in train).isdisjoint(set(v["id"] for v in val))


def test_save_jsonl(tmp_path):
    data = [{"messages": [{"role": "user", "content": "hi"}]}]
    out_file = tmp_path / "dataset.jsonl"
    SessionExporter.save_jsonl(data, out_file)

    assert out_file.exists()
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["messages"][0]["content"] == "hi"
