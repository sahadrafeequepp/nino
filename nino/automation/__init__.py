"""Automation, scheduling, and proactive messaging package for Nino."""

from nino.automation.briefing import DailyBriefingGenerator
from nino.automation.notifications import (
    ConsoleNotificationChannel,
    NotificationChannel,
    NotificationDispatcher,
    NtfyNotificationChannel,
    WebAvatarNotificationChannel,
)
from nino.automation.policy import ProactivePolicyEngine
from nino.automation.scheduler import SQLiteJobScheduler

__all__ = [
    "ProactivePolicyEngine",
    "SQLiteJobScheduler",
    "DailyBriefingGenerator",
    "NotificationChannel",
    "ConsoleNotificationChannel",
    "WebAvatarNotificationChannel",
    "NtfyNotificationChannel",
    "NotificationDispatcher",
]
