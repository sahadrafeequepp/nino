"""Notification channel adapters and dispatcher for Nino proactive alerts."""

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Any, Optional
import httpx

from nino.config import Settings

logger = logging.getLogger("nino.automation.notifications")


class NotificationChannel(ABC):
    """Abstract base class for delivering proactive notifications."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel identifier."""
        pass

    @abstractmethod
    async def send(self, message: str, title: str = "Nino Notification", is_urgent: bool = False) -> bool:
        """Send notification. Returns True if delivered successfully."""
        pass


class ConsoleNotificationChannel(NotificationChannel):
    """Prints proactive notifications cleanly to console/log."""

    @property
    def name(self) -> str:
        return "console"

    async def send(self, message: str, title: str = "Nino Notification", is_urgent: bool = False) -> bool:
        prefix = "🚨 [URGENT ALERT]" if is_urgent else "🔔 [NINO ALERT]"
        print(f"\n{prefix} {title}: {message}\n")
        logger.info("Delivered notification via console: '%s'", message)
        return True


class WebAvatarNotificationChannel(NotificationChannel):
    """Delivers proactive notification events to connected avatar WebSockets."""

    def __init__(self):
        self._connected_sockets: set[Any] = set()

    @property
    def name(self) -> str:
        return "web_avatar"

    def register_socket(self, ws: Any) -> None:
        self._connected_sockets.add(ws)

    def unregister_socket(self, ws: Any) -> None:
        self._connected_sockets.discard(ws)

    async def send(self, message: str, title: str = "Nino Notification", is_urgent: bool = False) -> bool:
        if not self._connected_sockets:
            return False

        payload = {
            "type": "proactive_notification",
            "title": title,
            "message": message,
            "is_urgent": is_urgent,
        }
        dead_sockets = set()
        delivered = False

        for ws in list(self._connected_sockets):
            try:
                await ws.send_json(payload)
                delivered = True
            except Exception:
                dead_sockets.add(ws)

        for ws in dead_sockets:
            self._connected_sockets.discard(ws)

        return delivered


class NtfyNotificationChannel(NotificationChannel):
    """Delivers private push notifications to a self-hosted or Tailscale ntfy server."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def name(self) -> str:
        return "ntfy"

    async def send(self, message: str, title: str = "Nino Notification", is_urgent: bool = False) -> bool:
        ntfy_url = (self.settings.ntfy_url or "").rstrip("/")
        topic = self.settings.ntfy_topic or ""
        if not ntfy_url or not topic:
            return False

        target_endpoint = f"{ntfy_url}/{topic}"
        headers = {"Title": title}
        if is_urgent:
            headers["Priority"] = "urgent"
            headers["Tags"] = "warning"
        if self.settings.ntfy_token:
            headers["Authorization"] = f"Bearer {self.settings.ntfy_token}"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(target_endpoint, content=message.encode("utf-8"), headers=headers)
                return res.status_code == 200
        except Exception as e:
            logger.warning("Failed to deliver push notification via ntfy: %s", e)
            return False


class TelegramNotificationChannel(NotificationChannel):
    """Delivers proactive notifications directly to Telegram Bot user."""

    def __init__(self, bot_token: Optional[str] = None, user_id: Optional[int] = None):
        import os
        self.bot_token = (bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        uid = user_id or os.getenv("MY_TELEGRAM_USER_ID", "")
        try:
            self.user_id: Optional[int] = int(uid) if uid else None
        except ValueError:
            self.user_id = None

    @property
    def name(self) -> str:
        return "telegram"

    async def send(self, message: str, title: str = "Nino Notification", is_urgent: bool = False) -> bool:
        if not self.bot_token or not self.user_id:
            return False

        header = "🚨 *URGENT NOTIFICATION*" if is_urgent else "🔔 *NINO NOTIFICATION*"
        payload = {
            "chat_id": self.user_id,
            "text": f"{header}\n\n*{title}*\n{message}",
            "parse_mode": "Markdown",
        }
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.post(url, json=payload)
                return res.status_code == 200
        except Exception as e:
            logger.warning("Failed to deliver notification via Telegram: %s", e)
            return False


class NotificationDispatcher:
    """Coordinates delivery across registered channels."""

    def __init__(self, channels: Optional[list[NotificationChannel]] = None):
        self.channels: list[NotificationChannel] = channels or []

    def add_channel(self, channel: NotificationChannel) -> None:
        self.channels.append(channel)

    async def broadcast(self, message: str, title: str = "Nino Notification", is_urgent: bool = False) -> dict[str, bool]:
        """Broadcast message to all channels. Returns dict of {channel_name: success}."""
        results = {}
        for channel in self.channels:
            try:
                res = await channel.send(message=message, title=title, is_urgent=is_urgent)
                results[channel.name] = res
            except Exception as e:
                logger.error("Error in channel '%s': %s", channel.name, e)
                results[channel.name] = False
        return results
