"""Confirmation manager with argument-bound hashes, expiry, and single-use validation."""

import hashlib
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any
from pydantic import BaseModel, Field

from nino.memory.db import MemoryDB
from nino.tools.permissions import PermissionTier


def compute_arguments_hash(tool_name: str, arguments: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash for tool name and canonical arguments."""
    canonical_json = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    payload = f"{tool_name}:{canonical_json}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Confirmation(BaseModel):
    """Model representing an interactive confirmation request for tool execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: dict[str, Any]
    arguments_hash: str
    summary: str
    tier: str
    status: str = "pending"  # pending, approved, denied, used, expired
    created_at: str
    expires_at: str


class ConfirmationManager:
    """Manages creation, approval, and cryptographic validation of tool confirmations."""

    def __init__(self, memory_db: MemoryDB, default_expiry_minutes: int = 5):
        self.db = memory_db
        self.default_expiry_minutes = default_expiry_minutes

    def create_confirmation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        summary: str,
        tier: PermissionTier | str,
        expiry_minutes: int | None = None,
    ) -> Confirmation:
        """Create and store a pending confirmation bound to exact arguments."""
        tier_str = tier.value if isinstance(tier, PermissionTier) else str(tier)
        arg_hash = compute_arguments_hash(tool_name, arguments)
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(minutes=expiry_minutes or self.default_expiry_minutes)

        conf = Confirmation(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            arguments=arguments,
            arguments_hash=arg_hash,
            summary=summary,
            tier=tier_str,
            status="pending",
            created_at=now.isoformat(),
            expires_at=expiry.isoformat(),
        )

        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO confirmations
                (id, tool_name, arguments_json, arguments_hash, summary, tier, status, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conf.id,
                    conf.tool_name,
                    json.dumps(conf.arguments),
                    conf.arguments_hash,
                    conf.summary,
                    conf.tier,
                    conf.status,
                    conf.created_at,
                    conf.expires_at,
                ),
            )
        return conf

    def get_confirmation(self, confirmation_id: str) -> Confirmation | None:
        """Retrieve confirmation and check expiration status."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM confirmations WHERE id = ?", (confirmation_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            now_iso = datetime.now(timezone.utc).isoformat()
            status = row["status"]
            if status in ("pending", "approved") and row["expires_at"] < now_iso:
                status = "expired"
                conn.execute(
                    "UPDATE confirmations SET status = 'expired' WHERE id = ?",
                    (confirmation_id,),
                )


            return Confirmation(
                id=row["id"],
                tool_name=row["tool_name"],
                arguments=json.loads(row["arguments_json"]),
                arguments_hash=row["arguments_hash"],
                summary=row["summary"],
                tier=row["tier"],
                status=status,
                created_at=row["created_at"],
                expires_at=row["expires_at"],
            )

    def list_pending(self) -> list[Confirmation]:
        """List all currently pending and unexpired confirmations."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            # Mark expired ones first
            conn.execute(
                "UPDATE confirmations SET status = 'expired' WHERE status = 'pending' AND expires_at < ?",
                (now_iso,),
            )
            cursor = conn.execute(
                "SELECT * FROM confirmations WHERE status = 'pending' ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()

            return [
                Confirmation(
                    id=r["id"],
                    tool_name=r["tool_name"],
                    arguments=json.loads(r["arguments_json"]),
                    arguments_hash=r["arguments_hash"],
                    summary=r["summary"],
                    tier=r["tier"],
                    status=r["status"],
                    created_at=r["created_at"],
                    expires_at=r["expires_at"],
                )
                for r in rows
            ]

    def approve(self, confirmation_id: str) -> tuple[bool, str]:
        """Approve a pending confirmation."""
        conf = self.get_confirmation(confirmation_id)
        if not conf:
            return False, f"Confirmation '{confirmation_id}' not found."
        if conf.status == "expired":
            return False, f"Confirmation '{confirmation_id}' has expired."
        if conf.status != "pending":
            return False, f"Confirmation '{confirmation_id}' is already {conf.status}."

        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE confirmations SET status = 'approved' WHERE id = ?",
                (confirmation_id,),
            )
        return True, "Approved successfully."

    def deny(self, confirmation_id: str) -> tuple[bool, str]:
        """Deny a pending confirmation."""
        conf = self.get_confirmation(confirmation_id)
        if not conf:
            return False, f"Confirmation '{confirmation_id}' not found."
        if conf.status in ("used", "denied"):
            return False, f"Confirmation '{confirmation_id}' is already {conf.status}."

        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE confirmations SET status = 'denied' WHERE id = ?",
                (confirmation_id,),
            )
        return True, "Denied successfully."

    def validate_and_consume(
        self,
        confirmation_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[bool, str]:
        """Validate confirmation token against tool name, exact argument hash, expiry, and single-use."""
        conf = self.get_confirmation(confirmation_id)
        if not conf:
            return False, f"Confirmation '{confirmation_id}' not found."

        if conf.status == "expired":
            return False, "Confirmation token has expired."

        if conf.status == "used":
            return False, "Confirmation token has already been used (single-use policy)."

        if conf.status == "denied":
            return False, "Confirmation token was explicitly denied by the user."

        if conf.status != "approved":
            return False, f"Confirmation token is not approved (current status: {conf.status})."

        # Check tool name
        if conf.tool_name != tool_name:
            return False, f"Confirmation token tool mismatch: expected '{conf.tool_name}', got '{tool_name}'."

        # Cryptographically check argument hash
        current_hash = compute_arguments_hash(tool_name, arguments)
        if current_hash != conf.arguments_hash:
            return (
                False,
                "Confirmation rejected: tool arguments have been modified since approval (argument binding violation).",
            )

        # Single-use consumption
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE confirmations SET status = 'used' WHERE id = ?",
                (confirmation_id,),
            )

        return True, "Validation successful."
