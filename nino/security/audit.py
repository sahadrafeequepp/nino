"""Append-only JSONL audit logger for request tracing, compliance, and memory operations."""

import datetime
import json
import os
from pathlib import Path
from typing import Any


class AuditLogger:
    """Append-only audit logger writing structured JSONL entries."""

    def __init__(self, log_path: str = "logs/audit.jsonl", log_content: bool = False):
        self.log_path = Path(log_path)
        self.log_content = log_content
        # Ensure the logs directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        client_ip: str = "127.0.0.1",
        request_content: Any = None,
        response_content: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an audit entry in JSONL format."""
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "client_ip": client_ip,
        }

        if extra:
            entry.update(extra)

        if self.log_content:
            entry["request_content"] = request_content
            entry["response_content"] = response_content
        else:
            entry["content"] = "[CONTENT_OMITTED]"

        line = json.dumps(entry) + "\n"
        with open(self.log_path, mode="a", encoding="utf-8") as f:
            f.write(line)

        return entry

    def log_tool_execution(
        self,
        tool_name: str,
        tier: str,
        decision: str,
        duration_ms: float,
        arguments: Any = None,
        result: Any = None,
    ) -> dict[str, Any]:
        """Record a structured tool execution event in JSONL format."""
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event_type": "tool_execution",
            "endpoint": f"/tool/{tool_name}",
            "tool_name": tool_name,
            "tier": tier,
            "decision": decision,
            "duration_ms": round(duration_ms, 2),
        }

        if self.log_content:
            entry["arguments"] = arguments
            entry["result"] = result
        else:
            entry["content"] = "[CONTENT_OMITTED]"

        line = json.dumps(entry) + "\n"
        with open(self.log_path, mode="a", encoding="utf-8") as f:
            f.write(line)

        return entry


    def log_memory_operation(
        self,
        action: str,
        entity_type: str,
        entity_id: str | None = None,
        count: int | None = None,
        duration_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Record a memory operation (read, write, delete) strictly as metadata without content."""
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event_type": "memory_operation",
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "count": count,
            "duration_ms": round(duration_ms, 2),
            "content_included": False,
        }

        line = json.dumps(entry) + "\n"
        with open(self.log_path, mode="a", encoding="utf-8") as f:
            f.write(line)

        return entry

    def log_image_capture(
        self,
        capture_type: str,
        status: str = "success",
        duration_ms: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record that an image capture happened (type and timestamp), never the image content."""
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event_type": "image_capture",
            "capture_type": capture_type,  # "upload", "screenshot", "camera"
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "content": "[IMAGE_CONTENT_NEVER_LOGGED]",
        }
        if extra:
            entry.update(extra)

        line = json.dumps(entry) + "\n"
        with open(self.log_path, mode="a", encoding="utf-8") as f:
            f.write(line)

        return entry
