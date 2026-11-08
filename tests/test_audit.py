"""Tests for append-only JSONL audit logging."""

import json
from pathlib import Path
import pytest
from nino.security.audit import AuditLogger


def test_audit_logger_creates_file_and_appends(tmp_path: Path):
    log_file = tmp_path / "sub" / "audit.jsonl"
    logger = AuditLogger(log_path=str(log_file), log_content=False)

    logger.log(
        endpoint="/health",
        method="GET",
        status_code=200,
        duration_ms=4.52,
        client_ip="127.0.0.1",
    )

    logger.log(
        endpoint="/chat",
        method="POST",
        status_code=401,
        duration_ms=1.12,
        client_ip="127.0.0.1",
    )

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    entry1 = json.loads(lines[0])
    assert entry1["endpoint"] == "/health"
    assert entry1["status_code"] == 200
    assert entry1["duration_ms"] == 4.52
    assert "timestamp" in entry1

    entry2 = json.loads(lines[1])
    assert entry2["endpoint"] == "/chat"
    assert entry2["status_code"] == 401


def test_audit_logger_omits_content_by_default(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=str(log_file), log_content=False)

    logger.log(
        endpoint="/chat",
        method="POST",
        status_code=200,
        duration_ms=25.0,
        client_ip="127.0.0.1",
        request_content={"messages": [{"content": "classified prompt"}]},
        response_content={"reply": "classified answer"},
    )

    entry = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert entry["content"] == "[CONTENT_OMITTED]"
    assert "request_content" not in entry
    assert "response_content" not in entry
    assert "classified" not in json.dumps(entry)


def test_audit_logger_logs_content_when_enabled(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=str(log_file), log_content=True)

    logger.log(
        endpoint="/chat",
        method="POST",
        status_code=200,
        duration_ms=15.0,
        client_ip="127.0.0.1",
        request_content={"prompt": "test prompt"},
        response_content={"reply": "test reply"},
    )

    entry = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert entry["request_content"] == {"prompt": "test prompt"}
    assert entry["response_content"] == {"reply": "test reply"}
