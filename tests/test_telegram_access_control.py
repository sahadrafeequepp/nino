"""Unit tests for Telegram Bot User Access Control, Owner Approvals, and Denials."""

import asyncio
import sys
from unittest.mock import MagicMock
from pathlib import Path
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
def clean_data_collector(tmp_path):
    """Provide a fresh, isolated DataCollector instance for testing."""
    dc = DataCollector(tmp_path)
    return dc


def test_owner_always_authorized(monkeypatch, clean_data_collector):
    """Verify the configured bot owner is always authorized regardless of state."""
    owner_id = 1334848825
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    assert bot.is_owner(owner_id) is True
    assert bot.is_authorized(owner_id) is True
    assert clean_data_collector.is_approved(owner_id, owner_id) is True


def test_unapproved_user_is_pending_and_not_authorized(monkeypatch, clean_data_collector):
    """Verify an unknown or newly registered user defaults to pending and is not authorized."""
    owner_id = 1334848825
    guest_id = 99887766
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    # Initial state
    assert bot.is_owner(guest_id) is False
    assert bot.is_authorized(guest_id) is False
    assert clean_data_collector.get_access_status(guest_id, owner_id) == "unknown"

    # Register visitor
    mock_guest = MagicMock()
    mock_guest.id = guest_id
    mock_guest.first_name = "Alice"
    mock_guest.username = "alice_test"
    clean_data_collector.record_user(mock_guest, is_owner=False)

    assert clean_data_collector.get_access_status(guest_id, owner_id) == "pending"
    assert bot.is_authorized(guest_id) is False


def test_owner_approves_and_denies_user(monkeypatch, clean_data_collector):
    """Verify owner can approve and deny a user, transitioning their authorization."""
    owner_id = 1334848825
    guest_id = 1977731223
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    # 1. Initially pending
    clean_data_collector.set_access_status(guest_id, "pending")
    assert bot.is_authorized(guest_id) is False
    assert clean_data_collector.get_access_status(guest_id, owner_id) == "pending"

    # 2. Owner approves
    clean_data_collector.set_access_status(guest_id, "approved")
    assert bot.is_authorized(guest_id) is True
    assert clean_data_collector.get_access_status(guest_id, owner_id) == "approved"

    # 3. Owner denies / revokes
    clean_data_collector.set_access_status(guest_id, "denied")
    assert bot.is_authorized(guest_id) is False
    assert clean_data_collector.get_access_status(guest_id, owner_id) == "denied"


def test_approve_command_by_owner(monkeypatch, clean_data_collector):
    """Verify /approve <user_id> executed by the owner approves the user and sends notification."""
    owner_id = 1334848825
    guest_id = 554433
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    # Register pending guest
    mock_guest = MagicMock()
    mock_guest.id = guest_id
    mock_guest.first_name = "Bob"
    clean_data_collector.record_user(mock_guest, is_owner=False)

    # Mock owner update & context
    mock_update = MagicMock()
    mock_owner = MagicMock()
    mock_owner.id = owner_id
    mock_update.effective_user = mock_owner

    mock_msg = MagicMock()
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.args = [str(guest_id)]
    mock_context.bot.send_message = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.approve_command(mock_update, mock_context))

    # Verify user is now approved
    assert clean_data_collector.is_approved(guest_id, owner_id) is True
    assert mock_msg.reply_text.called
    assert "User Approved" in mock_msg.reply_text.call_args[0][0]

    # Verify user received welcome notification
    assert mock_context.bot.send_message.called
    call_args = mock_context.bot.send_message.call_args
    assert call_args.kwargs.get("chat_id") == guest_id
    assert "Access Approved" in call_args.kwargs.get("text")


def test_deny_command_by_owner(monkeypatch, clean_data_collector):
    """Verify /deny <user_id> executed by the owner denies the user."""
    owner_id = 1334848825
    guest_id = 554433
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    # Pre-approve guest
    clean_data_collector.set_access_status(guest_id, "approved")
    assert clean_data_collector.is_approved(guest_id, owner_id) is True

    # Owner invokes /deny
    mock_update = MagicMock()
    mock_owner = MagicMock()
    mock_owner.id = owner_id
    mock_update.effective_user = mock_owner

    mock_msg = MagicMock()
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.args = [str(guest_id)]
    mock_context.bot.send_message = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.deny_command(mock_update, mock_context))

    assert clean_data_collector.is_approved(guest_id, owner_id) is False
    assert clean_data_collector.get_access_status(guest_id, owner_id) == "denied"
    assert "User Denied" in mock_msg.reply_text.call_args[0][0]


def test_non_owner_cannot_approve_or_deny(monkeypatch, clean_data_collector):
    """Verify non-owners are strictly blocked from using /approve, /deny, /users, or /pending."""
    owner_id = 1334848825
    intruder_id = 778899
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    mock_update = MagicMock()
    mock_user = MagicMock()
    mock_user.id = intruder_id
    mock_update.effective_user = mock_user

    mock_msg = MagicMock()
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.args = ["12345"]

    # Test /approve
    sync_run(bot.approve_command(mock_update, mock_context))
    assert "Owner Only" in mock_msg.reply_text.call_args[0][0]

    # Test /deny
    mock_msg.reply_text.reset_mock()
    sync_run(bot.deny_command(mock_update, mock_context))
    assert "Owner Only" in mock_msg.reply_text.call_args[0][0]

    # Test /users
    mock_msg.reply_text.reset_mock()
    sync_run(bot.users_command(mock_update, mock_context))
    assert "Owner Only" in mock_msg.reply_text.call_args[0][0]

    # Test /pending
    mock_msg.reply_text.reset_mock()
    sync_run(bot.pending_command(mock_update, mock_context))
    assert "Owner Only" in mock_msg.reply_text.call_args[0][0]


def test_callback_query_approve_and_deny(monkeypatch, clean_data_collector):
    """Verify inline button callbacks for approving and denying users."""
    owner_id = 1334848825
    guest_id = 332211
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    # 1. Test approve callback
    mock_update = MagicMock()
    mock_query = MagicMock()
    mock_owner = MagicMock()
    mock_owner.id = owner_id
    mock_query.from_user = mock_owner
    mock_query.data = f"approve_user:{guest_id}"
    mock_query.answer = MagicMock(return_value=InstantAwaitable())
    mock_query.edit_message_text = MagicMock(return_value=InstantAwaitable())
    mock_update.callback_query = mock_query

    mock_context = MagicMock()
    mock_context.bot.send_message = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.callback_query_handler(mock_update, mock_context))

    assert clean_data_collector.is_approved(guest_id, owner_id) is True
    assert "User Approved" in mock_query.edit_message_text.call_args[0][0]
    assert mock_context.bot.send_message.called

    # 2. Test deny callback
    mock_query.data = f"deny_user:{guest_id}"
    mock_query.edit_message_text.reset_mock()
    mock_context.bot.send_message.reset_mock()

    sync_run(bot.callback_query_handler(mock_update, mock_context))

    assert clean_data_collector.is_approved(guest_id, owner_id) is False
    assert clean_data_collector.get_access_status(guest_id, owner_id) == "denied"
    assert "User Denied" in mock_query.edit_message_text.call_args[0][0]


def test_pending_user_message_triggers_owner_notification(monkeypatch, clean_data_collector):
    """Verify when an unapproved user messages Nino, an approval alert is sent to the owner."""
    owner_id = 1334848825
    guest_id = 887766
    bot._pending_notifications_sent.clear()
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    mock_update = MagicMock()
    mock_guest = MagicMock()
    mock_guest.id = guest_id
    mock_guest.first_name = "Charlie"
    mock_guest.last_name = "Brown"
    mock_guest.username = "charlie_b"
    mock_update.effective_user = mock_guest

    mock_msg = MagicMock()
    mock_msg.text = "Hello, can I use this bot?"
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.bot.send_message = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.handle_text(mock_update, mock_context))

    # User received pending response
    assert mock_msg.reply_text.called
    assert "Access Pending" in mock_msg.reply_text.call_args[0][0]

    # Owner received notification with approve/deny buttons
    assert mock_context.bot.send_message.called
    call_args = mock_context.bot.send_message.call_args
    assert call_args.kwargs.get("chat_id") == owner_id
    assert "New User Access Request" in call_args.kwargs.get("text")
    assert "Charlie Brown" in call_args.kwargs.get("text")
    assert str(guest_id) in call_args.kwargs.get("text")
    assert call_args.kwargs.get("reply_markup") is not None


def test_scoped_commands_separation(monkeypatch):
    """Verify owner administrative commands are omitted from the default user menu and present in owner menu."""
    owner_id = 1334848825
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)

    user_cmds = {cmd.command for cmd in bot.USER_COMMANDS}
    owner_cmds = {cmd.command for cmd in bot.OWNER_COMMANDS}

    # Commands that must be owner-only
    owner_only = {"status", "mode", "remember", "memories", "vault", "users", "user", "pending", "approve", "deny", "requestcontact", "whois", "send", "broadcast"}

    # None of the owner-only commands should exist in the default user menu
    for cmd in owner_only:
        assert cmd not in user_cmds, f"Owner command '{cmd}' must NOT be in default USER_COMMANDS!"

    # All owner-only commands must exist in the owner menu
    for cmd in owner_only:
        assert cmd in owner_cmds, f"Owner command '{cmd}' must be in OWNER_COMMANDS!"

    # Verify registration with scoped BotCommandScope
    mock_bot = MagicMock()
    mock_bot.set_my_commands = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.register_scoped_commands(mock_bot))

    assert mock_bot.set_my_commands.call_count == 2
    # First call: default scope for regular users
    first_call = mock_bot.set_my_commands.call_args_list[0]
    assert first_call.args[0] == bot.USER_COMMANDS

    # Second call: scoped to owner's chat
    second_call = mock_bot.set_my_commands.call_args_list[1]
    assert second_call.args[0] == bot.OWNER_COMMANDS


def test_unapproved_user_prompted_for_contact_approval(monkeypatch, clean_data_collector):
    """Verify an unapproved user is explicitly prompted to approve sharing contact details and name."""
    owner_id = 1334848825
    guest_id = 554433
    bot._pending_notifications_sent.clear()
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    mock_update = MagicMock()
    mock_guest = MagicMock()
    mock_guest.id = guest_id
    mock_guest.first_name = "David"
    mock_guest.last_name = "Smith"
    mock_guest.username = "davidsmith"
    mock_update.effective_user = mock_guest

    mock_msg = MagicMock()
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.bot.send_message = MagicMock(return_value=InstantAwaitable())

    # 1. Test /start for pending user
    sync_run(bot.start_command(mock_update, mock_context))

    assert mock_msg.reply_text.called
    start_reply = mock_msg.reply_text.call_args[0][0]
    assert "approve sharing your contact details" in start_reply.lower() or "contact details" in start_reply.lower()
    assert mock_msg.reply_text.call_args.kwargs.get("reply_markup") is not None

    # 2. Test regular message access check for pending user
    mock_msg.reply_text.reset_mock()
    mock_msg.text = "Hello Nino"
    sync_run(bot.handle_text(mock_update, mock_context))

    assert mock_msg.reply_text.called
    msg_reply = mock_msg.reply_text.call_args[0][0]
    assert "approve sharing your contact details" in msg_reply.lower() or "contact details" in msg_reply.lower()
    assert mock_msg.reply_text.call_args.kwargs.get("reply_markup") is not None


def test_handle_contact_saves_details_and_notifies_owner(monkeypatch, clean_data_collector):
    """Verify when contact details are shared, they are saved and forwarded to the owner."""
    owner_id = 1334848825
    guest_id = 776655
    bot._pending_notifications_sent.clear()
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    mock_update = MagicMock()
    mock_guest = MagicMock()
    mock_guest.id = guest_id
    mock_guest.first_name = "Emma"
    mock_guest.last_name = "Watson"
    mock_guest.username = "emmaw"
    mock_update.effective_user = mock_guest

    mock_contact = MagicMock()
    mock_contact.phone_number = "+15551234567"
    mock_contact.first_name = "Emma"
    mock_contact.last_name = "Watson"
    mock_contact.user_id = guest_id

    mock_msg = MagicMock()
    mock_msg.contact = mock_contact
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.bot.send_message = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.handle_contact(mock_update, mock_context))

    # Check user record in data_collector
    user_record = clean_data_collector.get_user(guest_id)
    assert user_record is not None
    assert user_record.get("phone_number") == "+15551234567"
    assert user_record.get("first_name") == "Emma"

    # User received confirmation
    assert mock_msg.reply_text.called
    user_reply = mock_msg.reply_text.call_args[0][0]
    assert "Contact Details Approved" in user_reply or "Contact Received" in user_reply
    assert "+15551234567" in user_reply

    # Owner received notification with contact details
    assert mock_context.bot.send_message.called
    owner_call = mock_context.bot.send_message.call_args
    assert owner_call.kwargs.get("chat_id") == owner_id
    owner_text = owner_call.kwargs.get("text")
    assert "Emma Watson" in owner_text
    assert "+15551234567" in owner_text


def test_owner_request_contact_command_and_callback(monkeypatch, clean_data_collector):
    """Verify owner can send a contact sharing request to a pending user via command and inline button."""
    owner_id = 1334848825
    guest_id = 112233
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    # Register guest user without phone
    mock_guest = MagicMock()
    mock_guest.id = guest_id
    mock_guest.first_name = "Frank"
    mock_guest.last_name = "Miller"
    mock_guest.username = "fmiller"
    clean_data_collector.record_user(mock_guest, is_owner=False)

    # 1. Test /requestcontact command from owner
    mock_update = MagicMock()
    mock_owner = MagicMock()
    mock_owner.id = owner_id
    mock_update.effective_user = mock_owner

    mock_msg = MagicMock()
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.args = [str(guest_id)]
    mock_context.bot.send_message = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.request_contact_command(mock_update, mock_context))

    # User received contact request
    assert mock_context.bot.send_message.called
    user_call = mock_context.bot.send_message.call_args
    assert user_call.kwargs.get("chat_id") == guest_id
    assert "Contact Verification Required" in user_call.kwargs.get("text")
    assert user_call.kwargs.get("reply_markup") is not None

    # Owner received confirmation
    assert mock_msg.reply_text.called
    assert "Contact Request Sent" in mock_msg.reply_text.call_args[0][0]

    # 2. Test ask_contact callback query
    mock_query = MagicMock()
    mock_query.from_user = mock_owner
    mock_query.data = f"ask_contact:{guest_id}"
    mock_query.answer = MagicMock(return_value=InstantAwaitable())
    mock_query.message.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.callback_query = mock_query

    mock_context.bot.send_message.reset_mock()
    sync_run(bot.callback_query_handler(mock_update, mock_context))

    assert mock_context.bot.send_message.called
    assert mock_context.bot.send_message.call_args.kwargs.get("chat_id") == guest_id
    assert mock_query.message.reply_text.called
    assert "Contact Request Sent" in mock_query.message.reply_text.call_args[0][0]


def test_dismiss_contact_keyboard(monkeypatch, clean_data_collector):
    """Verify typing 'Dismiss' removes the reply keyboard."""
    owner_id = 1334848825
    guest_id = 445566
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    mock_update = MagicMock()
    mock_guest = MagicMock()
    mock_guest.id = guest_id
    mock_update.effective_user = mock_guest
    mock_msg = MagicMock()
    mock_msg.text = "Dismiss"
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()

    sync_run(bot.handle_text(mock_update, mock_context))

    assert mock_msg.reply_text.called
    assert "Contact sharing dismissed" in mock_msg.reply_text.call_args[0][0]


def test_whois_command_owner_and_guest(monkeypatch, clean_data_collector):
    """Verify /whois is restricted to the owner and formats user details properly."""
    owner_id = 1334848825
    guest_id = 889900
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    # 1. Test guest execution is blocked
    mock_guest_update = MagicMock()
    mock_guest = MagicMock()
    mock_guest.id = guest_id
    mock_guest.username = "guest_user"
    mock_guest_update.effective_user = mock_guest
    mock_guest_msg = MagicMock()
    mock_guest_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_guest_update.message = mock_guest_msg

    mock_context = MagicMock()
    sync_run(bot.whois_command(mock_guest_update, mock_context))

    assert mock_guest_msg.reply_text.called
    assert "Owner Only" in mock_guest_msg.reply_text.call_args[0][0]

    # 2. Test owner inspecting a user by ID
    mock_owner_update = MagicMock()
    mock_owner = MagicMock()
    mock_owner.id = owner_id
    mock_owner.first_name = "Sahad"
    mock_owner.username = "sahad_rafeeque"
    mock_owner_update.effective_user = mock_owner

    mock_owner_msg = MagicMock()
    mock_owner_msg.reply_to_message = None
    mock_owner_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_owner_msg.reply_photo = MagicMock(return_value=InstantAwaitable())
    mock_owner_update.message = mock_owner_msg

    mock_context.args = [str(owner_id)]
    mock_context.bot.get_chat = MagicMock(return_value=InstantAwaitable(mock_owner))
    mock_context.bot.get_user_profile_photos = MagicMock(return_value=InstantAwaitable(None))

    sync_run(bot.whois_command(mock_owner_update, mock_context))

    assert mock_owner_msg.reply_text.called
    reply_content = mock_owner_msg.reply_text.call_args[1].get("text", "") or mock_owner_msg.reply_text.call_args[0][0]
    assert "User Information" in reply_content
    assert str(owner_id) in reply_content
    assert "Sahad" in reply_content
    assert "OWNER" in reply_content


def test_send_message_command_owner_and_guest(monkeypatch, clean_data_collector):
    """Verify owner can send messages to users, while guests are blocked."""
    owner_id = 1334848825
    guest_id = 889900
    target_user_id = 1977731223

    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    # Pre-register target user
    mock_target = MagicMock()
    mock_target.id = target_user_id
    mock_target.username = "anjana_sree"
    mock_target.first_name = "Anjana"
    clean_data_collector.record_user(mock_target)

    # 1. Guest attempt should be rejected
    mock_guest_update = MagicMock()
    mock_guest = MagicMock()
    mock_guest.id = guest_id
    mock_guest_update.effective_user = mock_guest
    mock_guest_msg = MagicMock()
    mock_guest_msg.reply_to_message = None
    mock_guest_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_guest_update.message = mock_guest_msg

    mock_context = MagicMock()
    mock_context.args = [str(target_user_id), "Hello", "Anjana!"]

    sync_run(bot.send_message_command(mock_guest_update, mock_context))
    assert mock_guest_msg.reply_text.called
    assert "Owner Only" in mock_guest_msg.reply_text.call_args[0][0]

    # 2. Owner sending via ID
    mock_owner_update = MagicMock()
    mock_owner = MagicMock()
    mock_owner.id = owner_id
    mock_owner_update.effective_user = mock_owner
    mock_owner_msg = MagicMock()
    mock_owner_msg.reply_to_message = None
    mock_owner_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_owner_update.message = mock_owner_msg

    mock_context.bot.send_message = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.send_message_command(mock_owner_update, mock_context))

    assert mock_context.bot.send_message.called
    send_args = mock_context.bot.send_message.call_args[1]
    assert send_args["chat_id"] == target_user_id
    assert "Hello Anjana!" in send_args["text"]
    assert "Message from Bot Owner" in send_args["text"]

    # Check confirmation to owner
    assert mock_owner_msg.reply_text.called
    confirm_text = mock_owner_msg.reply_text.call_args[0][0]
    assert "Message Delivered" in confirm_text
    assert str(target_user_id) in confirm_text

    # 3. Owner sending via reply
    mock_reply_msg = MagicMock()
    mock_reply_user = MagicMock()
    mock_reply_user.id = target_user_id
    mock_reply_user.first_name = "Anjana"
    mock_reply_msg.from_user = mock_reply_user
    mock_owner_msg.reply_to_message = mock_reply_msg
    mock_context.args = ["Reply", "text", "here"]

    mock_context.bot.send_message.reset_mock()
    sync_run(bot.send_message_command(mock_owner_update, mock_context))

    assert mock_context.bot.send_message.called
    send_args = mock_context.bot.send_message.call_args[1]
    assert send_args["chat_id"] == target_user_id
    assert "Reply text here" in send_args["text"]


def test_broadcast_command_owner_and_guest(monkeypatch, clean_data_collector):
    """Verify owner can broadcast announcements to all users."""
    owner_id = 1334848825
    guest_id = 889900

    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", owner_id)
    monkeypatch.setattr(bot, "data_collector", clean_data_collector)

    # Register two guest users
    u1 = MagicMock()
    u1.id = 111111
    u1.username = "u1"
    clean_data_collector.record_user(u1)

    u2 = MagicMock()
    u2.id = 222222
    u2.username = "u2"
    clean_data_collector.record_user(u2)

    # 1. Guest blocked
    mock_guest_update = MagicMock()
    mock_guest = MagicMock()
    mock_guest.id = guest_id
    mock_guest_update.effective_user = mock_guest
    mock_guest_msg = MagicMock()
    mock_guest_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_guest_update.message = mock_guest_msg

    mock_context = MagicMock()
    mock_context.args = ["System", "upgrade", "tonight!"]

    sync_run(bot.broadcast_command(mock_guest_update, mock_context))
    assert mock_guest_msg.reply_text.called
    assert "Owner Only" in mock_guest_msg.reply_text.call_args[0][0]

    # 2. Owner broadcast
    mock_owner_update = MagicMock()
    mock_owner = MagicMock()
    mock_owner.id = owner_id
    mock_owner_update.effective_user = mock_owner
    mock_owner_msg = MagicMock()
    mock_status_msg = MagicMock()
    mock_status_msg.edit_text = MagicMock(return_value=InstantAwaitable())
    mock_owner_msg.reply_text = MagicMock(return_value=InstantAwaitable(mock_status_msg))
    mock_owner_update.message = mock_owner_msg

    mock_context.bot.send_message = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.broadcast_command(mock_owner_update, mock_context))

    assert mock_context.bot.send_message.call_count >= 2
    assert mock_status_msg.edit_text.called
    report = mock_status_msg.edit_text.call_args[0][0]
    assert "Broadcast Complete" in report
    assert f"Delivered</b>: {mock_context.bot.send_message.call_count}" in report

    # Verify interactions.jsonl records whom the broadcast was sent to
    recent_interactions = clean_data_collector.get_recent_interactions(limit=10)
    broadcast_entries = [i for i in recent_interactions if i.get("type") == "owner_broadcast"]
    assert len(broadcast_entries) > 0
    b_entry = broadcast_entries[-1]
    assert "delivered_to" in b_entry.get("metadata", {})
    assert len(b_entry["metadata"]["delivered_to"]) >= 2
    assert "recipient_ids" in b_entry["metadata"]
    assert 111111 in b_entry["metadata"]["recipient_ids"]
    assert 222222 in b_entry["metadata"]["recipient_ids"]
    assert "111111" in b_entry["content"]
    assert "222222" in b_entry["content"]



