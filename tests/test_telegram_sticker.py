"""Unit tests for Telegram Bot sticker reading, emotion detection, and accurate sticker replies."""

import io
import sys
from unittest.mock import MagicMock
import pytest
from PIL import Image

# Mock telegram if not installed in local test environment
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
    mock_telegram.BotCommand = lambda command, description="": MagicMock(command=command, description=description)
    mock_telegram.InlineKeyboardButton = MockInlineKeyboardButton
    mock_telegram.InlineKeyboardMarkup = MockInlineKeyboardMarkup
    sys.modules["telegram"] = mock_telegram
    sys.modules["telegram.ext"] = MagicMock()
    sys.modules["telegram.ext"].filters = mock_filters
    sys.modules["telegram.constants"] = mock_telegram.constants
    sys.modules["telegram.request"] = MagicMock()

from telegram_bot import bot
from telegram_bot.sticker_helper import (
    StickerCache,
    detect_emotion,
    get_target_emotions,
    pick_matching_sticker,
)


class InstantAwaitable:
    def __init__(self, val=None):
        self.val = val

    def __await__(self):
        if False:
            yield
        return self.val


def sync_run(coro):
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


class DummySticker:
    def __init__(self, file_id: str, file_unique_id: str, emoji: str, set_name: str = "TestPack"):
        self.file_id = file_id
        self.file_unique_id = file_unique_id
        self.emoji = emoji
        self.set_name = set_name
        self.is_animated = False
        self.is_video = False
        self.thumbnail = None


def test_detect_emotion_by_emoji():
    """Verify emotion detection from standard Telegram emoji tags."""
    assert detect_emotion("😂") == "joy"
    assert detect_emotion("🤣") == "joy"
    assert detect_emotion("❤️") == "love"
    assert detect_emotion("🥰") == "love"
    assert detect_emotion("😎") == "cool"
    assert detect_emotion("👍") == "agree"
    assert detect_emotion("😭") == "sad"
    assert detect_emotion("😡") == "angry"
    assert detect_emotion("😱") == "surprised"
    assert detect_emotion("🤔") == "thinking"
    assert detect_emotion("👋") == "greeting"
    assert detect_emotion("🎉") == "celebrate"


def test_detect_emotion_by_visual_context():
    """Verify emotion detection falls back to visual description keywords if emoji is missing/generic."""
    assert detect_emotion("🎨", "The character is crying with tears flowing") == "sad"
    assert detect_emotion(None, "A cartoon laughing hysterically haha") == "joy"
    assert detect_emotion(None, "A cute character holding a red heart love") == "love"
    assert detect_emotion(None, "An unknown abstract shape") == "joy"  # Default fallback


def test_get_target_emotions():
    """Verify prioritized complementary reaction list for various emotions."""
    assert "joy" in get_target_emotions("joy")
    assert "love" in get_target_emotions("love")
    assert "surprised" in get_target_emotions("angry")
    assert "love" in get_target_emotions("sad")


def test_pick_matching_sticker_from_pack():
    """Verify matching sticker selection prefers target emotions and excludes incoming sticker."""
    s1 = DummySticker(file_id="id_joy_1", file_unique_id="uid_1", emoji="😂")
    s2 = DummySticker(file_id="id_joy_2", file_unique_id="uid_2", emoji="🤣")
    s3 = DummySticker(file_id="id_cool_1", file_unique_id="uid_3", emoji="😎")
    s4 = DummySticker(file_id="id_sad_1", file_unique_id="uid_4", emoji="😢")

    pack = [s1, s2, s3, s4]

    # When incoming sticker is s1 ("😂", uid_1), target emotions ["joy", "cool"] should pick s2 or s3 (not s1)
    picked = pick_matching_sticker(pack, ["joy", "cool"], incoming_file_unique_id="uid_1")
    assert picked is not None
    assert picked.file_id in ["id_joy_2", "id_cool_1"]
    assert picked.file_unique_id != "uid_1"


def test_pick_matching_sticker_fallback():
    """Verify fallback behavior when pack has no direct matches for target emotion."""
    s1 = DummySticker(file_id="id_misc_1", file_unique_id="uid_1", emoji="🚗")
    s2 = DummySticker(file_id="id_misc_2", file_unique_id="uid_2", emoji="🚀")
    pack = [s1, s2]

    # Neither is in 'sad' or 'love', so should pick another sticker from the pack
    picked = pick_matching_sticker(pack, ["sad", "love"], incoming_file_unique_id="uid_1")
    assert picked is not None
    assert picked.file_id == "id_misc_2"

    # Empty pack
    assert pick_matching_sticker([], ["joy"]) is None


def test_sticker_cache(tmp_path):
    """Verify sticker cache persistence and retrieval."""
    cache_file = tmp_path / "sticker_cache.json"
    cache = StickerCache(cache_file)

    # Empty cache returns None
    assert cache.get_sticker_for_emotions(["joy"]) is None

    # Record stickers
    cache.record_sticker("file_joy_123", "joy")
    cache.record_sticker("file_love_456", "love")

    assert cache.get_sticker_for_emotions(["joy"]) == "file_joy_123"
    assert cache.get_sticker_for_emotions(["love"]) == "file_love_456"

    # Verify persistence to disk
    fresh_cache = StickerCache(cache_file)
    assert fresh_cache.get_sticker_for_emotions(["joy"]) == "file_joy_123"

    # Record sticker set
    dummy_set = MagicMock()
    dummy_set.stickers = [
        DummySticker(file_id="file_cool_789", file_unique_id="uid_c", emoji="😎"),
    ]
    fresh_cache.record_sticker_set(dummy_set)
    assert fresh_cache.get_sticker_for_emotions(["cool"]) == "file_cool_789"


def test_handle_sticker_authorized(monkeypatch, tmp_path):
    """Verify handle_sticker receives sticker, fetches matching sticker from pack, and replies."""
    user_id = 111222
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", user_id)
    monkeypatch.setattr(bot, "data_collector", MagicMock())
    monkeypatch.setattr(bot, "security_scanner", MagicMock())
    monkeypatch.setattr(bot, "media_processor", MagicMock())

    # Create dummy WebP image bytes
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    fake_webp_bytes = buf.getvalue()

    # Mock security scan result
    mock_scan_res = MagicMock()
    mock_scan_res.is_safe = True
    bot.security_scanner.scan_file_bytes = MagicMock(return_value=InstantAwaitable(mock_scan_res))
    bot.media_processor.extract_image = MagicMock(return_value=InstantAwaitable("A happy smiling character"))

    # Mock Update and Message
    mock_update = MagicMock()
    mock_user = MagicMock()
    mock_user.id = user_id
    mock_user.username = "owner"
    mock_user.first_name = "Owner"
    mock_update.effective_user = mock_user
    mock_update.effective_chat.id = 12345

    mock_msg = MagicMock()
    mock_msg.caption = None
    mock_msg.reply_sticker = MagicMock(return_value=InstantAwaitable())
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())

    # Incoming sticker
    incoming_sticker = DummySticker(file_id="in_file_1", file_unique_id="in_uid_1", emoji="😂", set_name="LaughPack")
    mock_file = MagicMock()
    async def fake_download(ba):
        ba.extend(fake_webp_bytes)
    mock_file.download_as_bytearray = fake_download
    incoming_sticker.get_file = MagicMock(return_value=InstantAwaitable(mock_file))
    mock_msg.sticker = incoming_sticker
    mock_update.message = mock_msg

    # Mock Context and Sticker Pack
    mock_context = MagicMock()
    mock_context.bot.send_chat_action = MagicMock(return_value=InstantAwaitable())

    matched_sticker = DummySticker(file_id="reply_file_99", file_unique_id="rep_uid_99", emoji="🤣", set_name="LaughPack")
    mock_set = MagicMock()
    mock_set.stickers = [incoming_sticker, matched_sticker]
    mock_context.bot.get_sticker_set = MagicMock(return_value=InstantAwaitable(mock_set))

    # Execute handler
    sync_run(bot.handle_sticker(mock_update, mock_context))

    # Verify sticker was replied with the matched sticker ID!
    assert mock_msg.reply_sticker.called
    call_args = mock_msg.reply_sticker.call_args
    assert call_args.kwargs.get("sticker") == "reply_file_99"


def test_handle_sticker_unauthorized(monkeypatch):
    """Verify handle_sticker rejects unauthorized users."""
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", 111111)
    mock_update = MagicMock()
    mock_user = MagicMock()
    mock_user.id = 999999
    mock_update.effective_user = mock_user

    mock_msg = MagicMock()
    mock_msg.sticker = DummySticker(file_id="f1", file_unique_id="u1", emoji="❤️")
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()

    sync_run(bot.handle_sticker(mock_update, mock_context))

    assert mock_msg.reply_text.called
    assert "Access Denied" in mock_msg.reply_text.call_args[0][0]
    assert not mock_msg.reply_sticker.called


def test_handle_sticker_pack_fetch_error_fallback(monkeypatch):
    """Verify handle_sticker falls back gracefully to sticker cache or mirroring if pack fetch fails."""
    user_id = 444333
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", user_id)
    monkeypatch.setattr(bot, "data_collector", MagicMock())
    monkeypatch.setattr(bot, "security_scanner", MagicMock())
    monkeypatch.setattr(bot, "media_processor", MagicMock())

    # Pre-populate cache with a love sticker
    bot.sticker_cache.cache = {}
    bot.sticker_cache.record_sticker("cached_love_id", "love")

    mock_scan_res = MagicMock()
    mock_scan_res.is_safe = True
    bot.security_scanner.scan_file_bytes = MagicMock(return_value=InstantAwaitable(mock_scan_res))
    bot.media_processor.extract_image = MagicMock(return_value=InstantAwaitable("A loving cat"))

    mock_update = MagicMock()
    mock_user = MagicMock()
    mock_user.id = user_id
    mock_update.effective_user = mock_user

    mock_msg = MagicMock()
    mock_msg.caption = None
    mock_msg.reply_sticker = MagicMock(return_value=InstantAwaitable())
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())

    incoming_sticker = DummySticker(file_id="in_file_love", file_unique_id="in_uid_love", emoji="❤️", set_name="BrokenPack")
    mock_file = MagicMock()
    async def fake_download(ba):
        ba.extend(b"dummy")
    mock_file.download_as_bytearray = fake_download
    incoming_sticker.get_file = MagicMock(return_value=InstantAwaitable(mock_file))
    mock_msg.sticker = incoming_sticker
    mock_update.message = mock_msg

    # Mock Context with failed get_sticker_set
    mock_context = MagicMock()
    mock_context.bot.send_chat_action = MagicMock(return_value=InstantAwaitable())
    mock_context.bot.get_sticker_set = MagicMock(side_effect=Exception("Sticker set not found"))

    # Execute handler
    sync_run(bot.handle_sticker(mock_update, mock_context))

    # Should still succeed and reply with a sticker!
    assert mock_msg.reply_sticker.called
    replied_id = mock_msg.reply_sticker.call_args.kwargs.get("sticker")
    assert replied_id in ["cached_love_id", "in_file_love"]


def test_handle_sticker_with_caption(monkeypatch):
    """Verify handle_sticker generates and sends text when message contains a caption."""
    user_id = 555666
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", user_id)
    monkeypatch.setattr(bot, "data_collector", MagicMock())
    monkeypatch.setattr(bot, "security_scanner", MagicMock())
    monkeypatch.setattr(bot, "media_processor", MagicMock())

    mock_scan_res = MagicMock()
    mock_scan_res.is_safe = True
    bot.security_scanner.scan_file_bytes = MagicMock(return_value=InstantAwaitable(mock_scan_res))
    bot.media_processor.extract_image = MagicMock(return_value=InstantAwaitable("A laughing face"))

    mock_agent = MagicMock()
    mock_agent.chat = MagicMock(return_value=InstantAwaitable(("That is hilarious!", "ollama")))
    mock_agent.set_media_context = MagicMock()
    monkeypatch.setattr(bot, "agent", mock_agent)

    mock_update = MagicMock()
    mock_user = MagicMock()
    mock_user.id = user_id
    mock_update.effective_user = mock_user

    mock_msg = MagicMock()
    mock_msg.caption = "What do you think of this?"
    mock_msg.reply_sticker = MagicMock(return_value=InstantAwaitable())
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())

    incoming_sticker = DummySticker(file_id="in_file_cap", file_unique_id="in_uid_cap", emoji="😂", set_name=None)
    mock_file = MagicMock()
    async def fake_download(ba):
        pass
    mock_file.download_as_bytearray = fake_download
    incoming_sticker.get_file = MagicMock(return_value=InstantAwaitable(mock_file))
    mock_msg.sticker = incoming_sticker
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.bot.send_chat_action = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.handle_sticker(mock_update, mock_context))

    assert mock_msg.reply_sticker.called
    assert mock_msg.reply_text.called
    assert mock_msg.reply_text.call_args[0][0] == "That is hilarious!"

