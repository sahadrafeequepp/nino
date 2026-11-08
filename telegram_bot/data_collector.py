"""Data collection, auditing, and media archiving module for Nino Telegram Bot.

Saves user info, interactions, and media into the 'bot_saved_data' directory:
- bot_saved_data/users.json: registry of all users (owner + guests), usernames, phone numbers, timestamps
- bot_saved_data/interactions.jsonl: chronological log of every message, command, and interaction
- bot_saved_data/media/{user_id}/: segregated media storage for photos, voice notes, and documents

Note on IP Addresses:
Telegram's Bot API architecture communicates exclusively via Telegram's MTProto cloud servers.
Client IP addresses are never exposed or transmitted to bots.
"""

import datetime
import json
import logging
import os
from pathlib import Path
import re
import socket
from typing import Any, Optional
import urllib.request

logger = logging.getLogger("nino.telegram.data_collector")

IP_NOTE = "Unavailable (Telegram Bot API proxies all client traffic via MTProto servers)"

# Indian Standard Time (IST / Asia/Kolkata)
try:
    from zoneinfo import ZoneInfo
    _tz_name = os.getenv("USER_TIMEZONE") or os.getenv("NINO_TIMEZONE") or "Asia/Kolkata"
    IST = ZoneInfo(_tz_name)
except Exception:
    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30), name="IST")


def get_now_ist() -> datetime.datetime:
    """Return current datetime in Indian Standard Time (IST)."""
    return datetime.datetime.now(IST)


def get_now_ist_iso() -> str:
    """Return current ISO 8601 timestamp in Indian Standard Time (IST)."""
    return datetime.datetime.now(IST).isoformat()


class DataCollector:
    """Manages recording of bot users, interaction logs, and media archives."""

    def __init__(self, base_dir: str | Path = "bot_saved_data"):
        self.base_dir = Path(base_dir).resolve()
        self.media_dir = self.base_dir / "media"
        self.users_file = self.base_dir / "users.json"
        self.interactions_file = self.base_dir / "interactions.jsonl"
        self.replies_file = self.base_dir / "reply.jsonl"

        # Ensure directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)

        # Initialize users file if missing
        if not self.users_file.exists():
            self._write_json(self.users_file, {})

        # Initialize interactions file if missing
        if not self.interactions_file.exists():
            self.interactions_file.touch()

        # Initialize reply file if missing
        if not self.replies_file.exists():
            self.replies_file.touch()

    def _read_users(self) -> dict[str, Any]:
        """Read users dictionary safely."""
        try:
            if self.users_file.exists():
                return json.loads(self.users_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not read users.json (%s), creating fresh dict", e)
        return {}

    def _write_json(self, path: Path, data: Any) -> None:
        """Write JSON safely with pretty formatting."""
        try:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to write JSON to %s: %s", path, e)

    def record_user(
        self,
        user: Any,
        is_owner: bool = False,
        phone_number: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record or update a user entry in bot_saved_data/users.json automatically."""
        if not user:
            return {}

        user_id_val = getattr(user, "id", None)
        if isinstance(user_id_val, (int, str)):
            user_id_str = str(user_id_val)
        else:
            user_id_str = "unknown"
            user_id_val = None

        uname = getattr(user, "username", None)
        fname = getattr(user, "first_name", None)
        lname = getattr(user, "last_name", None)
        raw_phone = phone_number or getattr(user, "phone_number", None)

        username = str(uname) if isinstance(uname, str) else None
        first_name = str(fname) if isinstance(fname, str) else None
        last_name = str(lname) if isinstance(lname, str) else None
        phone = str(raw_phone) if isinstance(raw_phone, str) else None

        now = get_now_ist_iso()

        # Automatic IP resolution for owner/local user
        auto_ip = None
        if is_owner:
            ips = fetch_host_ips()
            auto_ip = ips.get("public") or ips.get("local")

        users = self._read_users()
        user_record = users.get(user_id_str)

        if not user_record:
            user_record = {
                "user_id": user_id_val,
                "username": username,
                "first_name": first_name,
                "last_name": last_name,
                "phone_number": phone,
                "ip_address": auto_ip or IP_NOTE,
                "role": "owner" if is_owner else "guest",
                "access_status": "approved" if is_owner else "pending",
                "first_seen": now,
                "last_seen": now,
                "total_messages": 0,
                "media_count": 0,
            }
        else:
            # Update fields
            user_record["username"] = username or user_record.get("username")
            user_record["first_name"] = first_name or user_record.get("first_name")
            user_record["last_name"] = last_name or user_record.get("last_name")
            user_record["last_seen"] = now
            if is_owner:
                user_record["role"] = "owner"
                user_record["access_status"] = "approved"
                if auto_ip and (not user_record.get("ip_address") or user_record.get("ip_address") == IP_NOTE):
                    user_record["ip_address"] = auto_ip
            elif "access_status" not in user_record:
                user_record["access_status"] = "pending"
            if phone:
                user_record["phone_number"] = phone

        users[user_id_str] = user_record
        self._write_json(self.users_file, users)
        return user_record

    def update_phone(self, user_id: int, phone_number: str) -> bool:
        """Update or set phone number for a user in bot_saved_data/users.json."""
        return self.update_contact(user_id=user_id, phone_number=phone_number)

    def update_contact(
        self,
        user_id: int,
        phone_number: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> bool:
        """Update verified contact details (phone, first name, last name) in users.json."""
        user_id_str = str(user_id)
        users = self._read_users()
        now = get_now_ist_iso()
        clean_phone = phone_number.strip() if phone_number else None

        if user_id_str not in users:
            users[user_id_str] = {
                "user_id": user_id,
                "username": None,
                "first_name": first_name.strip() if first_name else None,
                "last_name": last_name.strip() if last_name else None,
                "phone_number": clean_phone,
                "ip_address": IP_NOTE,
                "role": "guest",
                "access_status": "pending",
                "first_seen": now,
                "last_seen": now,
                "total_messages": 0,
                "media_count": 0,
            }
        else:
            if clean_phone:
                users[user_id_str]["phone_number"] = clean_phone
            if first_name:
                users[user_id_str]["first_name"] = first_name.strip()
            if last_name:
                users[user_id_str]["last_name"] = last_name.strip()
            users[user_id_str]["last_seen"] = now

        self._write_json(self.users_file, users)
        logger.info("Saved contact details for user %s: phone=%s, name=%s %s", user_id, clean_phone, first_name, last_name)
        return True

    def update_ip(self, user_id: int, ip_address: str) -> bool:
        """Update or set IP address for a user in bot_saved_data/users.json."""
        user_id_str = str(user_id)
        users = self._read_users()
        now = get_now_ist_iso()
        clean_ip = ip_address.strip()

        if user_id_str not in users:
            users[user_id_str] = {
                "user_id": user_id,
                "username": None,
                "first_name": None,
                "last_name": None,
                "phone_number": None,
                "ip_address": clean_ip,
                "role": "guest",
                "first_seen": now,
                "last_seen": now,
                "total_messages": 0,
                "media_count": 0,
            }
        else:
            users[user_id_str]["ip_address"] = clean_ip
            users[user_id_str]["last_seen"] = now

        self._write_json(self.users_file, users)
        logger.info("Saved IP address for user %s: %s", user_id, clean_ip)
        return True

    def get_user(self, user_id: int) -> Optional[dict[str, Any]]:
        """Retrieve user record from users.json."""
        users = self._read_users()
        return users.get(str(user_id))

    def get_access_status(self, user_id: int, authorized_owner_id: Optional[int] = None) -> str:
        """Return access status: 'approved', 'pending', 'denied', or 'unknown'."""
        if authorized_owner_id is not None and user_id == authorized_owner_id:
            return "approved"
        user_record = self.get_user(user_id)
        if not user_record:
            return "unknown"
        if user_record.get("role") == "owner":
            return "approved"
        return user_record.get("access_status", "pending")

    def set_access_status(self, user_id: int, status: str) -> bool:
        """Set user access status ('approved', 'denied', 'pending')."""
        if status not in ("approved", "denied", "pending"):
            raise ValueError(f"Invalid access status: {status}")
        users = self._read_users()
        user_id_str = str(user_id)
        now = get_now_ist_iso()

        if user_id_str not in users:
            users[user_id_str] = {
                "user_id": user_id,
                "username": None,
                "first_name": None,
                "last_name": None,
                "phone_number": None,
                "ip_address": IP_NOTE,
                "role": "guest",
                "access_status": status,
                "first_seen": now,
                "last_seen": now,
                "total_messages": 0,
                "media_count": 0,
            }
        else:
            users[user_id_str]["access_status"] = status
            users[user_id_str]["last_seen"] = now

        self._write_json(self.users_file, users)
        logger.info("Updated access status for user %s to: %s", user_id, status)
        return True

    def is_approved(self, user_id: Optional[int], authorized_owner_id: Optional[int] = None) -> bool:
        """Check if user has approved access."""
        if user_id is None:
            return False
        if authorized_owner_id is not None and user_id == authorized_owner_id:
            return True
        if authorized_owner_id is None:
            return True
        return self.get_access_status(user_id, authorized_owner_id) == "approved"

    def list_users(self) -> dict[str, Any]:
        """Return all users from users.json."""
        return self._read_users()

    def get_pending_users(self) -> list[dict[str, Any]]:
        """Return all users with pending access status."""
        users = self._read_users()
        pending = []
        for uid_str, data in users.items():
            if data.get("role") != "owner" and data.get("access_status", "pending") == "pending":
                pending.append(data)
        return pending

    def log_interaction(
        self,
        user: Any,
        message_type: str,
        content: str = "",
        media_path: Optional[str] = None,
        is_owner: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Log a message/action into interactions.jsonl and increment user stats."""
        now = get_now_ist_iso()
        user_id = getattr(user, "id", None) if user else None
        if not isinstance(user_id, (int, str)):
            user_id = None
        username = getattr(user, "username", None) if user else None
        if not isinstance(username, str):
            username = None

        # Update user record count
        if user_id:
            user_id_str = str(user_id)
            users = self._read_users()
            if user_id_str in users:
                users[user_id_str]["total_messages"] = users[user_id_str].get("total_messages", 0) + 1
                if media_path:
                    users[user_id_str]["media_count"] = users[user_id_str].get("media_count", 0) + 1
                users[user_id_str]["last_seen"] = now
                self._write_json(self.users_file, users)
            else:
                self.record_user(user, is_owner=is_owner)

        entry = {
            "timestamp": now,
            "user_id": user_id,
            "username": username,
            "role": "owner" if is_owner else "guest",
            "type": message_type,
            "content": content[:2000] if content else "",
            "media_path": media_path,
            "metadata": metadata or {},
        }

        try:
            with open(self.interactions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Failed to append interaction log: %s", e)

        return entry

    def log_reply(
        self,
        user: Any,
        content: str = "",
        reply_to_type: str = "text",
        media_path: Optional[str] = None,
        voice_sent: bool = False,
        is_owner: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Log the bot's outgoing reply into bot_saved_data/reply.jsonl."""
        now = get_now_ist_iso()
        user_id = getattr(user, "id", None) if user else None
        if not isinstance(user_id, (int, str)):
            user_id = None
        username = getattr(user, "username", None) if user else None
        if not isinstance(username, str):
            username = None

        meta = dict(metadata or {})
        if voice_sent:
            meta["voice_sent"] = True

        entry = {
            "timestamp": now,
            "user_id": user_id,
            "username": username,
            "role": "owner" if is_owner else "guest",
            "type": "reply",
            "reply_to_type": reply_to_type,
            "content": content if content else "",
            "media_path": media_path,
            "metadata": meta,
        }

        try:
            with open(self.replies_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Failed to append reply log: %s", e)

        return entry

    def get_recent_replies(self, limit: int = 50, user_id: Optional[int] = None) -> list[dict[str, Any]]:
        """Return the most recent replies from reply.jsonl."""
        if not self.replies_file.exists():
            return []
        replies = []
        try:
            with open(self.replies_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            if user_id is None or record.get("user_id") == user_id:
                                replies.append(record)
                        except Exception:
                            continue
        except Exception as e:
            logger.warning("Could not read reply.jsonl: %s", e)
        return replies[-limit:]

    def get_recent_interactions(self, limit: int = 50, user_id: Optional[int] = None) -> list[dict[str, Any]]:
        """Return the most recent interactions from interactions.jsonl."""
        if not self.interactions_file.exists():
            return []
        interactions = []
        try:
            with open(self.interactions_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            if user_id is None or record.get("user_id") == user_id:
                                interactions.append(record)
                        except Exception:
                            continue
        except Exception as e:
            logger.warning("Could not read interactions.jsonl: %s", e)
        return interactions[-limit:]

    def save_media(
        self,
        user_id: int,
        file_bytes: bytes,
        original_filename: str,
    ) -> str:
        """Save a user-uploaded media file into bot_saved_data/media/{user_id}/.

        Returns relative path string from workspace root.
        """
        user_media_dir = self.media_dir / str(user_id)
        user_media_dir.mkdir(parents=True, exist_ok=True)

        now_str = get_now_ist().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^\w\-.]", "_", original_filename).strip("_") or "file"
        filename = f"{now_str}_{safe_name}"

        destination = user_media_dir / filename
        destination.write_bytes(file_bytes)

        # Return path relative to base or workspace
        try:
            rel_path = destination.relative_to(Path.cwd()).as_posix()
        except ValueError:
            rel_path = destination.as_posix()

        logger.info("Saved media file for user %s: %s (%d bytes)", user_id, rel_path, len(file_bytes))
        return rel_path


# Regex patterns for free-form text extraction
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}|\+?\d{10,14}")
IPV4_REGEX = re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b")


def extract_phone_from_text(text: str) -> Optional[str]:
    """Extract a phone number from conversational text."""
    if not text:
        return None
    matches = PHONE_REGEX.findall(text)
    for m in matches:
        digits_only = re.sub(r"[^\d]", "", m)
        if 10 <= len(digits_only) <= 15:
            return m.strip()
    return None


def extract_ip_from_text(text: str) -> Optional[str]:
    """Extract an IPv4 address from conversational text."""
    if not text:
        return None
    match = IPV4_REGEX.search(text)
    if match:
        candidate = match.group(0).strip()
        # Filter out obvious false positives like 0.0.0.0
        if candidate not in ("0.0.0.0", "255.255.255.255"):
            return candidate
    return None


def fetch_host_ips() -> dict[str, str]:
    """Fetch local network IP and public WAN IP for the host machine."""
    ips = {"local": "127.0.0.1", "public": None}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips["local"] = s.getsockname()[0]
        s.close()
    except Exception as e:
        logger.debug("Could not resolve local network IP: %s", e)

    try:
        req = urllib.request.Request("https://api.ipify.org?format=json", headers={"User-Agent": "NinoBot/1.0"})
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
            ips["public"] = data.get("ip")
    except Exception as e:
        logger.debug("ipify fetch failed: %s, trying ifconfig.me", e)
        try:
            req = urllib.request.Request("https://ifconfig.me/ip", headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                ips["public"] = response.read().decode("utf-8").strip()
        except Exception as e2:
            logger.debug("ifconfig.me fetch failed: %s", e2)
    return ips


