"""Unit tests for Telegram Bot voice reply default (OFF) and toggle functionality."""

import sys
from unittest.mock import MagicMock
import pytest

class MockInlineKeyboardButton:
    def __init__(self, text, callback_data=None):
        self.text = text
        self.callback_data = callback_data

class MockInlineKeyboardMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard

# Mock telegram if not installed in local test environment
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
    mock_telegram.BotCommand = lambda command, description="": MagicMock(command=command, description=description)
    mock_telegram.InlineKeyboardButton = MockInlineKeyboardButton
    mock_telegram.InlineKeyboardMarkup = MockInlineKeyboardMarkup
    sys.modules["telegram"] = mock_telegram
    sys.modules["telegram.ext"] = MagicMock()
    sys.modules["telegram.ext"].filters = mock_filters
    sys.modules["telegram.constants"] = mock_telegram.constants
    sys.modules["telegram.request"] = MagicMock()

from telegram_bot import bot

# Ensure bot uses our mock classes if telegram was mocked
if hasattr(bot, "InlineKeyboardButton") and isinstance(bot.InlineKeyboardButton, MagicMock):
    bot.InlineKeyboardButton = MockInlineKeyboardButton
    bot.InlineKeyboardMarkup = MockInlineKeyboardMarkup


def test_voice_reply_default_is_off():
    """Verify that voice replies are disabled (False) by default."""
    assert bot.DEFAULT_VOICE_REPLY is False
    assert bot.get_user_voice_pref() is False
    assert bot.get_user_voice_pref(user_id=987654321) is False


def test_user_voice_preference_toggle():
    """Verify that user voice preference can be toggled on and off per user."""
    test_user_id = 5551234

    # Initial state should be False (off)
    assert bot.get_user_voice_pref(test_user_id) is False

    # Toggle ON
    bot.set_user_voice_pref(test_user_id, True)
    assert bot.get_user_voice_pref(test_user_id) is True

    # Toggle OFF
    bot.set_user_voice_pref(test_user_id, False)
    assert bot.get_user_voice_pref(test_user_id) is False


def test_keyboards_reflect_voice_state():
    """Verify that owner and guest keyboards show the current voice state."""
    user_id = 7770001
    bot.set_user_voice_pref(user_id, False)

    ctrl_kb = bot.get_control_keyboard(user_id)
    buttons = [btn.text for row in ctrl_kb.inline_keyboard for btn in row]
    assert "🔇 Voice: OFF" in buttons
    assert "🔊 Voice: ON" not in buttons

    # Toggle on
    bot.set_user_voice_pref(user_id, True)
    ctrl_kb_on = bot.get_control_keyboard(user_id)
    buttons_on = [btn.text for row in ctrl_kb_on.inline_keyboard for btn in row]
    assert "🔊 Voice: ON" in buttons_on
    assert "🔇 Voice: OFF" not in buttons_on

    # Clean up
    bot.set_user_voice_pref(user_id, False)


class InstantAwaitable:
    def __await__(self):
        if False:
            yield
        return None

def sync_run(coro):
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value


def test_voice_toggle_command_on_off(monkeypatch):
    """Verify that /voice on and /voice off command updates preference and sends confirmation."""
    mock_dc = MagicMock()
    monkeypatch.setattr(bot, "data_collector", mock_dc)
    user_id = 999111
    mock_update = MagicMock()
    mock_user = MagicMock()
    mock_user.id = user_id
    mock_user.username = "tester"
    mock_user.first_name = "Test"
    mock_update.effective_user = mock_user

    reply_mock = MagicMock(return_value=InstantAwaitable())
    mock_update.message.reply_text = reply_mock

    mock_context = MagicMock()

    # 1. Turn voice ON with '/voice on'
    mock_context.args = ["on"]
    sync_run(bot.voice_toggle_command(mock_update, mock_context))
    assert bot.get_user_voice_pref(user_id) is True
    last_call = reply_mock.call_args
    assert "enabled" in last_call[0][0]

    # 2. Turn voice OFF with '/voice off'
    mock_context.args = ["off"]
    sync_run(bot.voice_toggle_command(mock_update, mock_context))
    assert bot.get_user_voice_pref(user_id) is False
    last_call = reply_mock.call_args
    assert "disabled" in last_call[0][0]

    # 3. Toggle voice with no arguments '/voice'
    mock_context.args = []
    sync_run(bot.voice_toggle_command(mock_update, mock_context))
    assert bot.get_user_voice_pref(user_id) is True

    sync_run(bot.voice_toggle_command(mock_update, mock_context))
    assert bot.get_user_voice_pref(user_id) is False
