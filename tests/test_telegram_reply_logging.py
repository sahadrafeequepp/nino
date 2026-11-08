"""Unit tests for bot reply logging into bot_saved_data/reply.jsonl."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

class MockBotCommand:
    def __init__(self, command, description=""):
        self.command = command
        self.description = description

class MockInlineKeyboardButton:
    def __init__(self, text, callback_data=None):
        self.text = text
        self.callback_data = callback_data

class MockInlineKeyboardMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard

try:
    import telegram
    from telegram.ext import filters
except ImportError:
    mock_telegram = MagicMock()
    mock_filters = MagicMock()
    class DummyBaseFilter:
        pass
    mock_filters.BaseFilter = DummyBaseFilter
    mock_telegram.constants.ParseMode.MARKDOWN = "Markdown"
    mock_telegram.BotCommand = MockBotCommand
    mock_telegram.InlineKeyboardButton = MockInlineKeyboardButton
    mock_telegram.InlineKeyboardMarkup = MockInlineKeyboardMarkup
    sys.modules["telegram"] = mock_telegram
    sys.modules["telegram.ext"] = MagicMock()
    sys.modules["telegram.ext"].filters = mock_filters
    sys.modules["telegram.constants"] = mock_telegram.constants
    sys.modules["telegram.request"] = MagicMock()

if "telegram" in sys.modules and not isinstance(sys.modules["telegram"].BotCommand, type):
    sys.modules["telegram"].BotCommand = MockBotCommand

from telegram_bot import bot
from telegram_bot.data_collector import DataCollector


class InstantAwaitable:
    """Helper to mock async methods synchronously."""
    def __init__(self, return_value=None):
        self.return_value = return_value

    def __await__(self):
        async def _coro():
            return self.return_value
        return _coro().__await__()


def sync_run(coro):
    return asyncio.run(coro)


@pytest.fixture
def temp_collector(tmp_path: Path):
    """Provide a fresh isolated DataCollector instance."""
    dc = DataCollector(tmp_path)
    return dc


def test_reply_file_created_on_init(temp_collector):
    """Verify reply.jsonl is created when DataCollector initializes."""
    assert temp_collector.replies_file.exists()
    assert temp_collector.replies_file.name == "reply.jsonl"


def test_log_reply_format(temp_collector):
    """Verify log_reply properly appends a formatted JSON record to reply.jsonl."""
    mock_user = MagicMock()
    mock_user.id = 123456
    mock_user.username = "test_user"

    entry = temp_collector.log_reply(
        user=mock_user,
        content="Hello! How can I help you today?",
        reply_to_type="text",
        voice_sent=True,
        is_owner=False,
        metadata={"engine": "ollama (gemma2:2b)"},
    )

    assert entry["user_id"] == 123456
    assert entry["username"] == "test_user"
    assert entry["role"] == "guest"
    assert entry["type"] == "reply"
    assert entry["reply_to_type"] == "text"
    assert entry["content"] == "Hello! How can I help you today?"
    assert entry["metadata"].get("voice_sent") is True
    assert entry["metadata"].get("engine") == "ollama (gemma2:2b)"

    # Verify physical file on disk
    lines = temp_collector.replies_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    disk_record = json.loads(lines[0])
    assert disk_record["content"] == "Hello! How can I help you today?"
    assert disk_record["user_id"] == 123456

    # Test get_recent_replies
    recent = temp_collector.get_recent_replies()
    assert len(recent) == 1
    assert recent[0]["user_id"] == 123456


def test_handle_text_logs_to_reply_file(monkeypatch, temp_collector):
    """Verify handle_text writes the generated AI response into reply.jsonl."""
    owner_id = 1334848825
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", temp_collector)

    mock_update = MagicMock()
    mock_owner = MagicMock()
    mock_owner.id = owner_id
    mock_owner.username = "sahad_rafeeque"
    mock_update.effective_user = mock_owner

    mock_msg = MagicMock()
    mock_msg.text = "Tell me a joke"
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    # Mock agent chat response
    async def mock_chat(*args, **kwargs):
        return ("Why did the chicken cross the road? To get to the other side!", "ollama (nino-manglish)")

    monkeypatch.setattr(bot.agent, "chat", mock_chat)

    mock_context = MagicMock()
    mock_context.bot.send_chat_action = MagicMock(return_value=InstantAwaitable())
    sync_run(bot.handle_text(mock_update, mock_context))

    assert mock_msg.reply_text.called

    # Check that reply was recorded in reply.jsonl
    replies = temp_collector.get_recent_replies(user_id=owner_id)
    assert len(replies) == 1
    assert "Why did the chicken cross the road?" in replies[0]["content"]
    assert replies[0]["reply_to_type"] == "text"
    assert replies[0]["role"] == "owner"


def test_start_and_help_commands_log_to_reply_file(monkeypatch, temp_collector):
    """Verify command responses (/start, /help) are logged into reply.jsonl."""
    owner_id = 1334848825
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", temp_collector)

    mock_update = MagicMock()
    mock_owner = MagicMock()
    mock_owner.id = owner_id
    mock_owner.username = "sahad_rafeeque"
    mock_update.effective_user = mock_owner

    mock_msg = MagicMock()
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()

    # 1. Run /start
    sync_run(bot.start_command(mock_update, mock_context))
    replies = temp_collector.get_recent_replies()
    assert len(replies) == 1
    assert replies[0]["reply_to_type"] == "command:/start"
    assert "Master. I'm Nino" in replies[0]["content"]

    # 2. Run /help
    sync_run(bot.help_command(mock_update, mock_context))
    replies = temp_collector.get_recent_replies()
    assert len(replies) == 2
    assert replies[1]["reply_to_type"] == "command:/help"
    assert "Nino Owner Command Guide" in replies[1]["content"]


def test_time_weather_and_location_log_to_reply_file(monkeypatch, temp_collector):
    """Verify /time, /weather, and location handler responses are logged into reply.jsonl."""
    owner_id = 1334848825
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", temp_collector)

    mock_update = MagicMock()
    mock_owner = MagicMock()
    mock_owner.id = owner_id
    mock_owner.username = "sahad_rafeeque"
    mock_update.effective_user = mock_owner

    mock_msg = MagicMock()
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.bot.send_chat_action = MagicMock(return_value=InstantAwaitable())

    # 1. /time command
    sync_run(bot.time_command(mock_update, mock_context))
    replies = temp_collector.get_recent_replies()
    assert len(replies) == 1
    assert replies[0]["reply_to_type"] == "command:/time"
    assert "Current Local Time" in replies[0]["content"]

    # 2. /weather command
    async def mock_weather(*args, **kwargs):
        return "⛅ Weather for Vadakara, Kerala:\n• Condition: Sunny\n• Temperature: 28°C"

    monkeypatch.setattr(bot, "fetch_weather_report", mock_weather)
    sync_run(bot.weather_command(mock_update, mock_context))
    replies = temp_collector.get_recent_replies()
    assert len(replies) == 2
    assert replies[1]["reply_to_type"] == "command:/weather"
    assert "Weather for Vadakara" in replies[1]["content"]

    # 3. Location pin handler
    mock_loc = MagicMock()
    mock_loc.latitude = 11.6085
    mock_loc.longitude = 75.5917
    mock_msg.location = mock_loc

    sync_run(bot.handle_location(mock_update, mock_context))
    replies = temp_collector.get_recent_replies()
    assert len(replies) == 3
    assert replies[2]["reply_to_type"] == "location"
    assert "Location Received & Clock Synchronized" in replies[2]["content"]

