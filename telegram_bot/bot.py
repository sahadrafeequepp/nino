"""Secure, Multi-User Telegram Bot Interface for Nino AI Assistant.

Features:
- Role separation: Owner administrative access vs. Sandboxed Guest chat access.
- Audit & Storage: Automatically logs user details and saves media to 'bot_saved_data/'.
- IP & Phone handling:
  * IP Addresses: Not exposed by Telegram's server architecture (documented and audited).
  * Phone Numbers: Captured via native Telegram Contact share requests and saved to users.json.
- Long polling architecture (zero open ports or webhooks required).
- 100% Local LLM engine: Local Ollama primary with zero cloud dependencies.
- Asymmetric shared brain: Bot shares Nino's database, but private PC memories are shielded.
- Voice pipeline: Inbound faster-whisper STT + Outbound edge-tts voice notes.
- Multimodal: Vision (photo analysis), document inspection, live weather.
- Google Antigravity denial policies & confirmation-gated vault access.
"""

import asyncio
import html
import inspect
import io
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import httpx

from telegram import (
    BotCommand,
    BotCommandScopeChat,
    BotCommandScopeDefault,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
    constants,
)
from telegram.constants import ChatAction, ParseMode
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

try:
    from telegram_bot.agent import DENIAL_POLICIES, NinoAgent
    from telegram_bot.checkin_server import CheckinServer, build_checkin_url
    from telegram_bot.clock import clock
    from telegram_bot.data_collector import (
        DataCollector,
        extract_ip_from_text,
        extract_phone_from_text,
        fetch_host_ips,
        IST,
    )
    from telegram_bot.file_reader import inspect_and_read_file
    from telegram_bot.link_inspector import LinkInspector
    from telegram_bot.reminders import extract_reminder_from_text, parse_time_expression, reminder_mgr
    from telegram_bot.video_inspector import generate_video_storyboard, inspect_video_file
    from telegram_bot.voice_utils import synthesize_voice, transcribe_voice
    from nino.security.antivirus import antivirus_scanner
except ImportError:
    from agent import DENIAL_POLICIES, NinoAgent
    from checkin_server import CheckinServer, build_checkin_url
    from clock import clock
    from data_collector import (
        DataCollector,
        extract_ip_from_text,
        extract_phone_from_text,
        fetch_host_ips,
        IST,
    )
    from file_reader import inspect_and_read_file
    from link_inspector import LinkInspector
    from reminders import extract_reminder_from_text, parse_time_expression, reminder_mgr
    from video_inspector import generate_video_storyboard, inspect_video_file
    from voice_utils import synthesize_voice, transcribe_voice
    from nino.security.antivirus import antivirus_scanner


# Load environment configuration
load_dotenv(Path(__file__).parent / ".env", override=True)
load_dotenv(Path.cwd() / ".env")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("nino.telegram.bot")

# Environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
MY_TELEGRAM_USER_ID_STR = os.getenv("MY_TELEGRAM_USER_ID", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("MODEL_NAME", "nino-manglish")).strip()
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision").strip()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en").strip()
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural").strip()
VAULT_DIR = os.getenv("VAULT_DIR", "./vault").strip()
DB_PATH = os.getenv("DB_PATH", "data/nino.db").strip()
NINO_MODE = os.getenv("NINO_MODE", "companion").strip()
DATA_DIR = os.getenv("DATA_DIR", "bot_saved_data").strip()

DEFAULT_LATITUDE = float(os.getenv("DEFAULT_LATITUDE", "11.6085"))
DEFAULT_LONGITUDE = float(os.getenv("DEFAULT_LONGITUDE", "75.5917"))
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Vadakara, Kerala")

# Parse and validate authorized owner user ID from MY_TELEGRAM_USER_ID
try:
    AUTHORIZED_USER_ID = int(os.getenv("MY_TELEGRAM_USER_ID", "1334848825").strip())
except ValueError:
    AUTHORIZED_USER_ID = None


def get_owner_id() -> Optional[int]:
    """Retrieve the configured owner Telegram ID from AUTHORIZED_USER_ID or MY_TELEGRAM_USER_ID."""
    global AUTHORIZED_USER_ID
    if AUTHORIZED_USER_ID is not None:
        return AUTHORIZED_USER_ID
    env_id = os.getenv("MY_TELEGRAM_USER_ID", "").strip()
    if env_id:
        try:
            return int(env_id)
        except ValueError:
            logger.error("Invalid MY_TELEGRAM_USER_ID: '%s' must be an integer.", env_id)
    return None


# Initialize Nino Local Agent (100% Local on Ollama)
agent = NinoAgent(
    base_url=OLLAMA_BASE_URL,
    model_name=OLLAMA_MODEL,
    vision_model=OLLAMA_VISION_MODEL,
    vault_dir=VAULT_DIR,
    db_path=DB_PATH,
    mode=NINO_MODE,
)

# Initialize Data Collector for user registry, interactions, and media archiving
data_collector = DataCollector(DATA_DIR)

# Initialize Checkin Server for IP Address verification
CHECKIN_PORT = int(os.getenv("CHECKIN_PORT", "8088"))
checkin_server = CheckinServer(
    data_collector=data_collector,
    host="0.0.0.0",
    port=CHECKIN_PORT,
)

# Initialize Safe Link Inspector with SSRF Firewall & Antivirus
link_inspector = LinkInspector()

# User preference flag: whether Nino replies with voice to messages (default: False / OFF)
DEFAULT_VOICE_REPLY = os.getenv("TELEGRAM_VOICE_REPLY_DEFAULT", "false").strip().lower() in ("true", "1", "yes")
user_voice_reply_preference = DEFAULT_VOICE_REPLY
user_voice_preferences: dict[int, bool] = {}


def get_user_voice_pref(user_id: Optional[int] = None) -> bool:
    """Return whether voice replies are enabled for a user (default: False / OFF)."""
    if user_id is not None and user_id in user_voice_preferences:
        return user_voice_preferences[user_id]
    return user_voice_reply_preference


def set_user_voice_pref(user_id: Optional[int], enabled: bool) -> None:
    """Set voice reply preference for a user and sync the global default fallback."""
    global user_voice_reply_preference
    if user_id is not None:
        user_voice_preferences[user_id] = enabled
    user_voice_reply_preference = enabled


try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from telegram_bot.security_scanner import security_scanner
    from telegram_bot.media_processor import media_processor
    from telegram_bot.sticker_helper import (
        StickerCache,
        detect_emotion,
        get_target_emotions,
        pick_matching_sticker,
    )
except ImportError:
    from security_scanner import security_scanner
    from media_processor import media_processor
    from sticker_helper import (
        StickerCache,
        detect_emotion,
        get_target_emotions,
        pick_matching_sticker,
    )

# Sticker settings & persistence cache
TELEGRAM_DEFAULT_STICKER_SET = os.getenv("TELEGRAM_DEFAULT_STICKER_SET", "").strip()
TELEGRAM_STICKER_REPLY_TEXT = os.getenv("TELEGRAM_STICKER_REPLY_TEXT", "false").strip().lower() in ("true", "1", "yes")
sticker_cache = StickerCache(Path(DATA_DIR) / "sticker_cache.json")


# ==============================================================================
# Role & Access Helper
# ==============================================================================
def is_owner(user_id: Optional[int]) -> bool:
    """Strictly verify if the given Telegram user ID matches MY_TELEGRAM_USER_ID."""
    if user_id is None:
        return False
    owner_id = get_owner_id()
    return owner_id is not None and user_id == owner_id


def is_authorized(user_id: Optional[int]) -> bool:
    """Gate access to the bot owner and owner-approved users.

    Permits the configured MY_TELEGRAM_USER_ID as well as users marked as
    'approved' by the owner in the user registry.
    """
    owner_id = get_owner_id()
    if owner_id is None:
        return True
    if user_id is None:
        return False
    if user_id == owner_id:
        return True
    return data_collector.is_approved(user_id, owner_id)


_pending_notifications_sent: set[int] = set()


async def notify_owner_of_access_request(
    bot_instance,
    user: Any,
    force: bool = False,
    phone: Optional[str] = None,
) -> None:
    """Notify the owner about a user requesting access with interactive inline buttons."""
    owner_id = get_owner_id()
    if owner_id is None or not bot_instance:
        return

    user_id = getattr(user, "id", None)
    if not user_id or user_id == owner_id:
        return

    if not force and user_id in _pending_notifications_sent:
        logger.info("Owner already notified for pending user %s, skipping duplicate alert.", user_id)
        return

    user_record = data_collector.get_user(user_id) or {}
    first_name = getattr(user, "first_name", None) or user_record.get("first_name") or "Guest"
    last_name = getattr(user, "last_name", None) or user_record.get("last_name") or ""
    full_name = f"{first_name} {last_name}".strip()
    username = getattr(user, "username", None) or user_record.get("username")
    username_str = f"@{username}" if username else "None"
    phone_val = phone or user_record.get("phone_number")
    phone_str = f"`{phone_val}`" if phone_val else "Not shared"

    text = (
        "🔔 *New User Access Request*\n\n"
        f"• *Name*: {full_name}\n"
        f"• *User ID*: `{user_id}`\n"
        f"• *Username*: {username_str}\n"
        f"• *Phone*: {phone_str}\n\n"
        "Would you like to grant this user access to Nino?"
    )

    action_buttons = [
        InlineKeyboardButton("✅ Approve", callback_data=f"approve_user:{user_id}"),
        InlineKeyboardButton("❌ Deny", callback_data=f"deny_user:{user_id}"),
    ]
    keyboard = [action_buttons]
    if not phone_val:
        keyboard.append([
            InlineKeyboardButton("📱 Request Contact Details", callback_data=f"ask_contact:{user_id}")
        ])

    try:
        await bot_instance.send_message(
            chat_id=owner_id,
            text=text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN,
        )
        _pending_notifications_sent.add(user_id)
        logger.info("Sent access request notification for user %s to owner %s", user_id, owner_id)
    except Exception as e:
        logger.warning("Could not send access request notification to owner %s: %s", owner_id, e)


async def check_user_access_or_prompt(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    """Check user authorization.

    If authorized (owner or approved user), returns True.
    If pending or denied, sends appropriate response to user,
    alerts the owner if pending, and returns False.
    """
    user = update.effective_user
    user_id = user.id if user else None

    # Owner is always permitted immediately
    if is_owner(user_id):
        return True

    if is_authorized(user_id):
        return True

    target_msg = update.message if getattr(update, "message", None) is not None else getattr(update, "effective_message", None)

    if user_id is None:
        if target_msg:
            res = target_msg.reply_text("⛔ Access Denied.")
            if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                await res
        return False

    owner_id = get_owner_id()
    status = data_collector.get_access_status(user_id, owner_id)

    if status == "denied":
        denied_msg = "⛔ *Access Denied*: You do not have permission to use this bot."
        data_collector.log_interaction(user, message_type="blocked", content="access_denied", is_owner=False)
        data_collector.log_reply(user=user, content=denied_msg, reply_to_type="access_check", is_owner=False)
        if target_msg:
            res = target_msg.reply_text(
                denied_msg,
                parse_mode=ParseMode.MARKDOWN,
            )
            if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                await res
        return False

    # Status is "pending" or new visitor
    data_collector.record_user(user, is_owner=False)
    data_collector.set_access_status(user_id, "pending")
    data_collector.log_interaction(user, message_type="pending_request", content="awaiting_approval", is_owner=False)

    user_record = data_collector.get_user(user_id) or {}
    has_contact = bool(user_record.get("phone_number"))

    if target_msg:
        if has_contact:
            pending_msg = (
                "⏳ *Access Pending Approval*: Your request and contact details have been submitted to the bot owner.\n\n"
                "You will receive a notification here once your access is approved!"
            )
            data_collector.log_reply(user=user, content=pending_msg, reply_to_type="access_check", is_owner=False)
            res = target_msg.reply_text(
                pending_msg,
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            pending_msg = (
                "⏳ *Access Pending Approval*: Access Denied until approved by the bot owner.\n\n"
                "To request access to this private bot, please approve sharing your contact details and name with the owner for verification.\n\n"
                "Tap the button below to approve and share your contact card:"
            )
            data_collector.log_reply(user=user, content=pending_msg, reply_to_type="access_check", is_owner=False)
            res = target_msg.reply_text(
                pending_msg,
                reply_markup=get_contact_request_reply_markup(),
                parse_mode=ParseMode.MARKDOWN,
            )
        if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
            await res

    await notify_owner_of_access_request(context.bot, user)
    return False


class OwnerFilter(filters.BaseFilter):
    """Filter that matches strictly the authorized owner (MY_TELEGRAM_USER_ID)."""

    def filter(self, message) -> bool:
        if not message or not message.from_user:
            return False
        return is_owner(message.from_user.id)


owner_filter = OwnerFilter()


# ==============================================================================
# Weather Utility
# ==============================================================================
WMO_WEATHER_CODES = {
    0: "Clear sky ☀️",
    1: "Mainly clear 🌤️",
    2: "Partly cloudy ⛅",
    3: "Overcast ☁️",
    45: "Fog 🌫️",
    48: "Depositing rime fog 🌫️",
    51: "Light drizzle 🌦️",
    53: "Moderate drizzle 🌧️",
    55: "Dense drizzle 🌧️",
    61: "Slight rain 🌦️",
    63: "Moderate rain 🌧️",
    65: "Heavy rain ⛈️",
    71: "Slight snow 🌨️",
    73: "Moderate snow 🌨️",
    75: "Heavy snow ❄️",
    80: "Slight rain showers 🌦️",
    81: "Moderate rain showers 🌧️",
    82: "Violent rain showers ⛈️",
    95: "Thunderstorm ⚡",
}


async def fetch_weather_report(lat: float, lon: float, location_name: str) -> str:
    """Fetch current weather from Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json().get("current_weather", {})
                temp = data.get("temperature", "--")
                wind = data.get("windspeed", "--")
                code = data.get("weathercode", 0)
                cond = WMO_WEATHER_CODES.get(code, "Clear")
                return (
                    f"⛅ *Weather for {location_name}*:\n"
                    f"• *Condition*: {cond}\n"
                    f"• *Temperature*: {temp}°C\n"
                    f"• *Wind Speed*: {wind} km/h\n"
                    f"• *Coordinates*: `{lat}, {lon}`"
                )
    except Exception as e:
        logger.warning("Weather fetch error: %s", e)
    return f"⚠️ Could not fetch current weather for {location_name}."


# ==============================================================================
# UI Keyboards (Owner Dashboard, Guest Menu, Contact Request)
# ==============================================================================
def get_control_keyboard(user_id: Optional[int] = None) -> InlineKeyboardMarkup:
    """Create interactive control buttons for bot owner."""
    voice_label = "🔊 Voice: ON" if get_user_voice_pref(user_id) else "🔇 Voice: OFF"
    keyboard = [
        [
            InlineKeyboardButton("🎭 Switch Mode", callback_data="btn_mode"),
            InlineKeyboardButton(voice_label, callback_data="btn_voice"),
        ],
        [
            InlineKeyboardButton("⛅ Live Weather", callback_data="btn_weather"),
            InlineKeyboardButton("🕒 Inbuilt Clock", callback_data="btn_time"),
        ],
        [
            InlineKeyboardButton("⏰ Reminders", callback_data="btn_reminders"),
            InlineKeyboardButton("🧠 View Memories", callback_data="btn_memories"),
        ],
        [
            InlineKeyboardButton("👥 Manage Users", callback_data="btn_manage_users"),
            InlineKeyboardButton("⚡ System Status", callback_data="btn_status"),
        ],
        [
            InlineKeyboardButton("🧹 Clear Chat", callback_data="btn_clear"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_guest_keyboard(user_id: Optional[int] = None) -> InlineKeyboardMarkup:
    """Create interactive buttons for guest visitors."""
    voice_label = "🔊 Voice: ON" if get_user_voice_pref(user_id) else "🔇 Voice: OFF"
    keyboard = [
        [
            InlineKeyboardButton("⛅ Live Weather", callback_data="btn_weather"),
            InlineKeyboardButton("🕒 Inbuilt Clock", callback_data="btn_time"),
        ],
        [
            InlineKeyboardButton("⏰ Reminders", callback_data="btn_reminders"),
            InlineKeyboardButton(voice_label, callback_data="btn_voice"),
        ],
        [
            InlineKeyboardButton("❓ Help Guide", callback_data="btn_help"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_contact_request_reply_markup() -> ReplyKeyboardMarkup:
    """Native Telegram keyboard requesting contact card."""
    keyboard = [
        [KeyboardButton("📱 Approve & Share Contact Details (Name & Phone)", request_contact=True)],
        [KeyboardButton("Dismiss")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)


# ==============================================================================
# Command Handlers
# ==============================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command with role-appropriate onboarding."""
    user = update.effective_user
    user_id = user.id if user else None
    owner = is_owner(user_id)

    data_collector.record_user(user, is_owner=owner)
    data_collector.log_interaction(user, message_type="command", content="/start", is_owner=owner)

    if owner:
        greeting = (
            "✨ *Hey there, Master. I'm Nino.*\n\n"
            "Your personal AI assistant is active, connected, and ready.\n\n"
            "Tap a quick action below or message me directly:"
        )
        data_collector.log_reply(user=user, content=greeting, reply_to_type="command:/start", is_owner=True)
        await update.message.reply_text(
            greeting,
            reply_markup=get_control_keyboard(user_id),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Check non-owner authorization
    status = data_collector.get_access_status(user_id, AUTHORIZED_USER_ID)
    if status == "denied":
        denied_msg = "⛔ *Access Denied*: You do not have permission to use this bot."
        data_collector.log_reply(user=user, content=denied_msg, reply_to_type="command:/start", is_owner=False)
        await update.message.reply_text(
            denied_msg,
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    if is_authorized(user_id):
        first_name = user.first_name if user else "Friend"
        greeting = (
            f"✨ *Hey {first_name}, I'm Nino!*\n\n"
            "I'm an AI assistant. You can chat with me, send voice notes, share pictures for analysis, "
            "send documents, or check live weather.\n\n"
            "Tap a quick action below or message me directly:"
        )
        data_collector.log_reply(user=user, content=greeting, reply_to_type="command:/start", is_owner=False)
        await update.message.reply_text(
            greeting,
            reply_markup=get_guest_keyboard(user_id),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Pending or new visitor
    first_name = user.first_name if user else "Friend"
    data_collector.set_access_status(user_id, "pending")

    user_record = data_collector.get_user(user_id) or {}
    has_contact = bool(user_record.get("phone_number"))

    if has_contact:
        greeting = (
            f"👋 *Hey {first_name}, I'm Nino!*\n\n"
            "⏳ *Access Request Pending*: Your request and verified contact details have been submitted to the bot owner for approval.\n\n"
            "You will receive a notification here as soon as your access is approved!"
        )
        data_collector.log_reply(user=user, content=greeting, reply_to_type="command:/start", is_owner=False)
        await update.message.reply_text(
            greeting,
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        greeting = (
            f"👋 *Hey {first_name}, I'm Nino!*\n\n"
            "⏳ *Access Request Pending*: Access to Nino is private and requires owner approval.\n\n"
            "To request access, please approve sharing your contact details (phone number) and name with the owner for verification.\n\n"
            "Tap the button below to approve and share your contact details:"
        )
        data_collector.log_reply(user=user, content=greeting, reply_to_type="command:/start", is_owner=False)
        await update.message.reply_text(
            greeting,
            reply_markup=get_contact_request_reply_markup(),
            parse_mode=ParseMode.MARKDOWN,
        )
    await notify_owner_of_access_request(context.bot, user)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command with role-specific command directory."""
    user = update.effective_user
    owner = is_owner(user.id if user else None)
    data_collector.log_interaction(user, message_type="command", content="/help", is_owner=owner)

    if owner:
        help_text = (
            "🤖 *Nino Owner Command Guide*:\n\n"
            "💬 *Chat & Multimodal*:\n"
            "• Type any message or question directly\n"
            "• Send a voice note for audio transcription and speech reply\n"
            "• Send a photo/screenshot with a question for visual analysis\n"
            "• Send a `.txt` or `.md` document for summaries and review\n"
            "• Send your location pin for local weather and clock sync\n\n"
            "⏰ *Reminders & Alerts*:\n"
            "• `/remind <time> <task>` - Schedule reminder (e.g. `/remind 10m call mom`)\n"
            "• `/reminders` - View active scheduled reminders with cancel/snooze\n"
            "• Say in chat: _\"remind me in 15 minutes to drink water\"_\n\n"
            "👥 *User & Access Control*:\n"
            "• `/users` - View all registered users and their access status\n"
            "• `/pending` - View and manage pending user access requests\n"
            "• `/approve <user_id>` - Grant access to a user\n"
            "• `/deny <user_id>` - Revoke or deny access to a user\n"
            "• `/requestcontact <user_id>` - Ask user to share contact & name\n"
            "• `/whois [user_id|@username]` - Inspect user Telegram profile & info\n"
            "• `/send <id|@username> <message>` - Send direct message to a user\n"
            "• `/broadcast <message>` - Broadcast announcement to all users\n\n"
            "⚙️ *Owner Controls & Memory*:\n"
            "• `/status` - Check Hybrid engine & brain memory stats\n"
            "• `/time [city]` - Inbuilt clock & location time sync\n"
            "• `/mode <companion|work|focus>` - Switch personality style\n"
            "• `/voice [on|off]` - Toggle automatic voice replies\n"
            "• `/remember <fact>` - Record a permanent memory into Nino's brain\n"
            "• `/memories` - List memories accessible to Telegram\n"
            "• `/weather [city]` - Current weather report\n"
            "• `/vault [filename]` - Inspect authorized private vault notes\n"
            "• `/clear` - Flush current conversation thread"
        )
    else:
        help_text = (
            "🤖 *Nino Assistant Guide*:\n\n"
            "💬 *What you can do*:\n"
            "• Type any question or message directly to chat\n"
            "• Send a voice note to talk to me with audio replies\n"
            "• Send a photo to have me describe and analyze it\n"
            "• Send a document (`.pdf`, `.docx`, `.txt`) for review\n"
            "• Send your location pin for local clock sync & weather\n\n"
            "⏰ *Reminders*:\n"
            "• `/remind <time> <task>` - Schedule a reminder\n"
            "• `/reminders` - Check scheduled reminders\n\n"
            "📋 *Available Commands*:\n"
            "• `/start` - Launch main menu\n"
            "• `/help` - Show this guide\n"
            "• `/time [city]` - Inbuilt clock & exact time\n"
            "• `/weather` - Check current weather\n"
            "• `/voice [on|off]` - Toggle audio voice replies\n"
            "• `/clear` - Reset your chat conversation\n"
            "• `/clearmedia` - Clear active media session memory\n"
            "• `/contact` - Share your contact/phone number"
        )
    data_collector.log_reply(user=user, content=help_text, reply_to_type="command:/help", is_owner=owner)
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def contact_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Prompt user to approve sharing their contact details and name."""
    user = update.effective_user
    owner = is_owner(user.id if user else None)
    data_collector.log_interaction(
        user,
        message_type="command",
        content="/contact",
        is_owner=owner,
    )
    prompt_text = (
        "📱 *Share Contact Details & Name*:\n\n"
        "Please tap the button below to approve sharing your contact details and name with the owner:"
    )
    data_collector.log_reply(user=user, content=prompt_text, reply_to_type="command:/contact", is_owner=owner)
    await update.message.reply_text(
        prompt_text,
        reply_markup=get_contact_request_reply_markup(),
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_contact(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process contact cards shared by users to capture verified phone number and name."""
    message = update.message
    if not message or not message.contact:
        return

    user = update.effective_user
    contact = message.contact
    phone_number = contact.phone_number
    owner = is_owner(user.id if user else None)

    first_name = contact.first_name or (user.first_name if user else "Friend")
    last_name = contact.last_name or (user.last_name if user else "")

    data_collector.record_user(user, is_owner=owner, phone_number=phone_number)
    data_collector.update_contact(
        user.id if user else 0,
        phone_number=phone_number,
        first_name=contact.first_name,
        last_name=contact.last_name,
    )
    data_collector.log_interaction(
        user,
        message_type="contact",
        content=f"Shared contact: {first_name} {last_name} ({phone_number})",
        is_owner=owner,
        metadata={
            "phone_number": phone_number,
            "first_name": contact.first_name,
            "last_name": contact.last_name,
            "contact_user_id": contact.user_id,
        },
    )

    if not owner and not is_authorized(user.id if user else None):
        reply_msg = (
            f"✅ *Contact Details Approved & Received*\n\n"
            f"Thank you, *{first_name}*! Your name and contact details (`{phone_number}`) have been submitted to the bot owner for approval.\n\n"
            f"You will receive a notification here once the owner approves your access."
        )
        data_collector.log_reply(user=user, content=reply_msg, reply_to_type="contact", is_owner=False)
        await message.reply_text(
            reply_msg,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=ParseMode.MARKDOWN,
        )
        await notify_owner_of_access_request(context.bot, user, force=True, phone=phone_number)
    else:
        reply_msg = f"📱 *Contact Verified*: Thank you, {first_name}! Your contact details (`{phone_number}`) have been securely saved."
        data_collector.log_reply(user=user, content=reply_msg, reply_to_type="contact", is_owner=True)
        await message.reply_text(
            reply_msg,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=ParseMode.MARKDOWN,
        )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check hybrid engine status, Ollama availability, malware scanner, and media session state."""
    user = update.effective_user
    user_id = user.id if user else None
    if not is_owner(user_id):
        data_collector.log_interaction(user, message_type="blocked", content="/status", is_owner=False)
        await update.message.reply_text("🔒 *Owner Only*: System diagnostics are restricted to the bot owner.")
        return

    data_collector.log_interaction(user, message_type="command", content="/status", is_owner=True)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    ollama_online = await agent.is_ollama_online()

    ollama_status = "🟢 ONLINE" if ollama_online else "🔴 OFFLINE (PC Sleeping / Stopped)"
    vt_status = "🟢 ACTIVE (VirusTotal v3)" if security_scanner.has_vt_api else "🛡️ Local Magic-Byte & Header Guard"

    active_engine = f"Ollama ({agent.model_name}) 100% Local on PC" if ollama_online else "🔴 Offline"
    v_pref = get_user_voice_pref(user_id)
    voice_status = "Enabled 🔊" if v_pref else "Disabled 🔇 (Default: text-only)"

    memories = agent.brain.list_accessible_memories()

    media_ctx = agent.get_media_context(user_id or 0)
    if media_ctx:
        media_status = f"Active ({media_ctx['type'].upper()}: `{media_ctx['filename_or_url']}`)"
    else:
        media_status = "None (ready for uploads)"

    status_report = (
        "⚡ *Nino Local System Status*:\n\n"
        f"• *Current Active Engine*: {active_engine}\n"
        f"• *Local Ollama ({agent.model_name})*: {ollama_status}\n"
        f"• *Architecture*: 100% Local & Private (Zero Cloud Fallback)\n"
        f"• *Security & Malware Guard*: {vt_status}\n"
        f"• *Active Media Memory*: {media_status}\n"
        f"• *Active Persona*: `{agent.mode.upper()}`\n"
        f"• *Voice Output*: {voice_status} (`{TTS_VOICE}`)\n"
        f"• *Shared Brain Memories*: {len(memories)} facts accessible to bot\n"
        f"• *Antigravity Security Policies*: Active (OS, Raw Disk, and External URLs blocked)\n\n"
        "_Tip: Nino runs 100% locally on your PC via Ollama for absolute privacy._"
    )
    data_collector.log_reply(user=user, content=status_report, reply_to_type="command:/status", is_owner=True)
    await update.message.reply_text(
        status_report,
        reply_markup=get_control_keyboard(user_id),
        parse_mode=ParseMode.MARKDOWN,
    )


async def clearmedia_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Release cached media context without clearing general chat memory."""
    user = update.effective_user
    user_id = user.id if user else None
    if not is_authorized(user_id):
        if update.effective_message:
            await update.effective_message.reply_text("⛔ Access Denied.")
        return

    agent.clear_media_context(user_id or 0)
    data_collector.log_interaction(user, message_type="command", content="/clearmedia", is_owner=is_owner(user_id))
    reply_msg = "🧹 Media context cleared. Your general chat memory is still preserved."
    data_collector.log_reply(user=user, content=reply_msg, reply_to_type="command:/clearmedia", is_owner=is_owner(user_id))
    await update.message.reply_text(reply_msg)


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Switch Nino's persona mode (Owner only)."""
    user = update.effective_user
    if not is_owner(user.id if user else None):
        data_collector.log_interaction(user, message_type="blocked", content="/mode", is_owner=False)
        await update.message.reply_text("🔒 *Owner Only*: Persona configuration is restricted to the bot owner.")
        return

    data_collector.log_interaction(user, message_type="command", content="/mode", is_owner=True)
    args = context.args
    if not args or args[0].lower() not in ("companion", "work", "focus"):
        keyboard = [
            [
                InlineKeyboardButton("✨ Companion (The Cool One)", callback_data="set_mode:companion"),
            ],
            [
                InlineKeyboardButton("💼 Work (Direct & Task-first)", callback_data="set_mode:work"),
            ],
            [
                InlineKeyboardButton("🎯 Focus (Minimalist Bullets)", callback_data="set_mode:focus"),
            ],
        ]
        msg = f"Current mode is: *{agent.mode.upper()}*.\nSelect a mode below:"
        data_collector.log_reply(user=user, content=msg, reply_to_type="command:/mode", is_owner=True)
        await update.message.reply_text(
            msg,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    new_mode = args[0].lower()
    agent.mode = new_mode
    reply_msg = f"🎭 Switched to *{new_mode.upper()}* mode."
    data_collector.log_reply(user=user, content=reply_msg, reply_to_type="command:/mode", is_owner=True)
    await update.message.reply_text(reply_msg, parse_mode=ParseMode.MARKDOWN)


async def voice_toggle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle voice replies for messages."""
    user = update.effective_user
    user_id = user.id if user else None
    if not await check_user_access_or_prompt(update, context):
        return
    owner = is_owner(user_id)
    data_collector.log_interaction(
        user,
        message_type="command",
        content="/voice",
        is_owner=owner,
    )
    args = context.args
    current = get_user_voice_pref(user_id)
    if args and args[0].lower() in ("off", "disable", "no", "false", "0"):
        set_user_voice_pref(user_id, False)
        reply_msg = "🔇 Voice replies are now *disabled* (text-only mode)."
    elif args and args[0].lower() in ("on", "enable", "yes", "true", "1"):
        set_user_voice_pref(user_id, True)
        reply_msg = "🔊 Voice replies are now *enabled*. Nino will speak responses."
    else:
        new_state = not current
        set_user_voice_pref(user_id, new_state)
        status = "enabled 🔊" if new_state else "disabled 🔇 (text-only mode)"
        reply_msg = f"Voice replies are now *{status}*."
    data_collector.log_reply(user=user, content=reply_msg, reply_to_type="command:/voice", is_owner=owner)
    await update.message.reply_text(reply_msg, parse_mode=ParseMode.MARKDOWN)


async def remember_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Save an explicit fact/memory into Nino's shared brain (Owner only)."""
    user = update.effective_user
    if not is_owner(user.id if user else None):
        data_collector.log_interaction(user, message_type="blocked", content="/remember", is_owner=False)
        await update.message.reply_text("🔒 *Owner Only*: Memory database storage is restricted to the bot owner.")
        return

    if not context.args:
        await update.message.reply_text(
            "🧠 *How to store a memory*:\n"
            "Use `/remember <fact>`\n"
            "Example: `/remember I love black coffee with no sugar`\n\n"
            "_This memory will be stored in Nino's brain and will be remembered on both Telegram and your PC._",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    fact_text = " ".join(context.args).strip()
    data_collector.log_interaction(user, message_type="command", content=f"/remember {fact_text}", is_owner=True)
    success = agent.brain.save_telegram_fact(fact_text, category="general")
    if success:
        await update.message.reply_text(
            f"🧠 *Remembered*:\n\"{fact_text}\"\n\n_Saved to shared brain — PC Nino will also know this._",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await update.message.reply_text("⚠️ Could not store memory in database.")


async def memories_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List memories accessible to Telegram (Owner only)."""
    user = update.effective_user
    if not is_owner(user.id if user else None):
        data_collector.log_interaction(user, message_type="blocked", content="/memories", is_owner=False)
        await update.message.reply_text("🔒 *Owner Only*: Memory inspection is restricted to the bot owner.")
        return

    data_collector.log_interaction(user, message_type="command", content="/memories", is_owner=True)
    memories = agent.brain.list_accessible_memories()
    if not memories:
        await update.message.reply_text(
            "🧠 *Brain Memory*: No public or Telegram memories found yet.\n"
            "Use `/remember <fact>` to add one!",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    lines = ["🧠 *Accessible Memories* (Profile & Telegram):"]
    for m in memories[:15]:
        origin_tag = "[Telegram]" if m.get("origin") == "telegram" else "[Shared]"
        lines.append(f"• `{origin_tag}` {m['text']}")

    lines.append("\n_Note: Private PC conversations are safely shielded from Telegram._")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def weather_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch current weather for default city or user location."""
    user = update.effective_user
    owner = is_owner(user.id if user else None)
    data_collector.log_interaction(
        user,
        message_type="command",
        content="/weather",
        is_owner=owner,
    )
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    report = await fetch_weather_report(DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_CITY)
    data_collector.log_reply(user=user, content=report, reply_to_type="command:/weather", is_owner=owner)
    await update.message.reply_text(report, parse_mode=ParseMode.MARKDOWN)


async def time_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display exact current time of the location Nino is located at, or query another city."""
    user = update.effective_user
    owner = is_owner(user.id if user else None)
    data_collector.log_interaction(
        user,
        message_type="command",
        content="/time",
        is_owner=owner,
    )

    # Optional argument: query time for a specific city / place
    if context.args:
        city_query = " ".join(context.args).strip()
        try:
            from nino.tools.weather import geocode_place_name
            geo = await geocode_place_name(city_query)
            if geo and "timezone" in geo:
                tz_name = geo["timezone"]
                dt = clock.now(tz_name)
                period = clock.period_of_day(dt.hour)
                place_name = f"{geo.get('name', city_query)}, {geo.get('admin1', '')}, {geo.get('country', '')}".strip(", ")
                msg = (
                    f"🕒 *Inbuilt Clock — Local Time for {place_name}*:\n\n"
                    f"• *Current Time*: `{dt.strftime('%I:%M:%S %p')}` ({period})\n"
                    f"• *24-Hour Clock*: `{dt.strftime('%H:%M:%S')}`\n"
                    f"• *Date*: {dt.strftime('%A, %B %d, %Y')}\n"
                    f"• *Timezone*: `{tz_name}` ({dt.strftime('%Z, %z')})\n"
                    f"• *Coordinates*: `{geo.get('latitude', '')}, {geo.get('longitude', '')}`"
                )
                data_collector.log_reply(user=user, content=msg, reply_to_type="command:/time", is_owner=owner)
                await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
                return
        except Exception as e:
            logger.debug("Failed to geocode city for time query: %s", e)

    # Fast Instant Timestamp & Full Inbuilt Clock Report
    now_str = clock.now().strftime("%I:%M:%S %p (%A, %d %b %Y)")
    clock_report = clock.get_clock_report()
    msg = f"🕒 *Current Local Time:*\n`{now_str}`\n\n{clock_report}"
    data_collector.log_reply(user=user, content=msg, reply_to_type="command:/time", is_owner=owner)
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear conversation history and active media context for the current user."""
    user = update.effective_user
    user_id = user.id if user else None
    if not await check_user_access_or_prompt(update, context):
        return

    agent.reset_conversation(user_id=user_id)
    agent.clear_media_context(user_id or 0)
    owner = is_owner(user_id)
    data_collector.log_interaction(user, message_type="command", content="/clear", is_owner=owner)
    clear_msg = "🧹 *Memory Cleared*: Starting a fresh conversation thread."
    data_collector.log_reply(user=user, content=clear_msg, reply_to_type="command:/clear", is_owner=owner)
    await update.message.reply_text(clear_msg, parse_mode=ParseMode.MARKDOWN)


async def remind_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Schedule a reminder via command: /remind <time> <message>."""
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return

    owner = is_owner(user.id)
    data_collector.record_user(user, is_owner=owner)
    data_collector.log_interaction(user, message_type="command", content="/remind", is_owner=owner)

    args = context.args
    if not args or len(args) < 2:
        help_text = (
            "⏰ *Nino Reminder Command*:\n\n"
            "Use `/remind <time> <task>`\n\n"
            "*Examples*:\n"
            "• `/remind 5m Call mom`\n"
            "• `/remind 30s Check the oven`\n"
            "• `/remind 1h Drink water`\n"
            "• `/remind 9:30pm Take medicine`\n"
            "• `/remind tomorrow Submit project report`\n\n"
            "_You can also simply say:_ `\"remind me in 10 minutes to take a break\"`"
        )
        data_collector.log_reply(user=user, content=help_text, reply_to_type="command:/remind", is_owner=owner)
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
        return

    time_arg = args[0]
    message_arg = " ".join(args[1:]).strip()

    try:
        reminder = reminder_mgr.create_reminder(
            message=message_arg,
            time_expression=time_arg,
            chat_id=chat.id,
            user_id=user.id,
        )
        reply = (
            f"⏰ *Reminder Scheduled!*\n\n"
            f"📌 *Task*: {reminder['message']}\n"
            f"🕒 *Alert at*: `{reminder['formatted_time']}` ({reminder['time_label']})\n\n"
            f"_Nino will send you a Telegram alert right here when it's due!_"
        )
        data_collector.log_reply(user=user, content=reply, reply_to_type="command:/remind", is_owner=owner)
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error("Failed to create reminder: %s", e)
        err_msg = f"⚠️ Could not set reminder: {e}"
        data_collector.log_reply(user=user, content=err_msg, reply_to_type="command:/remind", is_owner=owner)
        await update.message.reply_text(err_msg)


async def reminders_list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all pending reminders with inline actions to cancel or snooze."""
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return

    data_collector.log_interaction(user, message_type="command", content="/reminders", is_owner=is_owner(user.id))
    reminders = reminder_mgr.list_reminders(chat_id=chat.id, user_id=user.id)

    if not reminders:
        await update.message.reply_text(
            "⏰ *No Active Reminders*\n\n"
            "You have no pending reminders scheduled.\n"
            "Use `/remind 10m <task>` or say _\"remind me in 15 minutes to call John\"_ to schedule one!",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    lines = ["⏰ *Your Active Reminders*:\n"]
    keyboard = []
    for r in reminders[:10]:
        try:
            dt = datetime.fromisoformat(r["trigger_time"])
            disp = dt.strftime("%I:%M %p (%A, %d %b)")
        except Exception:
            disp = r["trigger_time"]

        lines.append(f"• **#{r['id']}**: {r['message']}\n  🕒 `{disp}`\n")
        keyboard.append([
            InlineKeyboardButton(f"❌ Cancel #{r['id']}", callback_data=f"cancel_rem:{r['id']}"),
            InlineKeyboardButton(f"⏰ +15m #{r['id']}", callback_data=f"snooze_rem:{r['id']}:15"),
        ])

    await update.message.reply_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN,
    )


async def vault_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inspect or request authorized access to a vault file (Owner only)."""
    user = update.effective_user
    if not is_owner(user.id if user else None):
        data_collector.log_interaction(user, message_type="blocked", content="/vault", is_owner=False)
        await update.message.reply_text("🔒 *Owner Only*: Vault storage is strictly restricted to the bot owner.")
        return

    data_collector.log_interaction(user, message_type="command", content="/vault", is_owner=True)
    args = context.args
    if not args:
        files = agent.vault.list_files()
        if not files:
            await update.message.reply_text(
                "📂 *Vault Empty*: No files found in `./vault`.\n"
                "Place text or markdown notes in the vault folder to inspect them here.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        file_list = "\n".join(f"• `{f}`" for f in files)
        await update.message.reply_text(
            f"📂 *Authorized Vault Notes*:\n{file_list}\n\nUse `/vault <filename>` to read a note.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    target_file = args[0].strip()
    if not agent.vault.is_safe_path(target_file):
        await update.message.reply_text("⛔ *Security Refusal*: Path traversal outside the vault directory is blocked.")
        return

    keyboard = [
        [
            InlineKeyboardButton("✅ Confirm & Read", callback_data=f"vault_read:{target_file}"),
            InlineKeyboardButton("❌ Cancel", callback_data="vault_cancel"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        f"🔐 *Vault Access Confirmation Required*:\n"
        f"Do you authorize reading `{target_file}` from your local vault?",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN,
    )


async def approve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Approve a user to grant them access to Nino (Owner only)."""
    user = update.effective_user
    if not is_owner(user.id if user else None):
        await update.message.reply_text("🔒 *Owner Only*: User management is restricted to the bot owner.")
        return

    args = context.args
    if not args:
        pending = data_collector.get_pending_users()
        if not pending:
            await update.message.reply_text(
                "Usage: `/approve <user_id>`\n\nNo pending users at this time. Use `/users` to view all users.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        lines = ["👥 *Pending Access Requests*:\n"]
        keyboard = []
        for p in pending:
            p_uid = p.get("user_id")
            p_name = f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip() or f"User {p_uid}"
            phone = p.get("phone_number") or "No phone"
            lines.append(f"• *{p_name}* (`{p_uid}`) - Phone: `{phone}`")
            keyboard.append([
                InlineKeyboardButton(f"✅ Approve {p_name[:12]}", callback_data=f"approve_user:{p_uid}"),
                InlineKeyboardButton(f"❌ Deny", callback_data=f"deny_user:{p_uid}"),
            ])

        await update.message.reply_text(
            "\n".join(lines) + "\n\nTap a button below or type `/approve <user_id>`:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    try:
        target_id = int(args[0])
    except ValueError:
        await update.message.reply_text("⚠️ Invalid User ID. Must be an integer (e.g. `/approve 1977731223`).", parse_mode=ParseMode.MARKDOWN)
        return

    if is_owner(target_id):
        await update.message.reply_text("👑 User is already the bot owner.")
        return

    data_collector.set_access_status(target_id, "approved")
    _pending_notifications_sent.discard(target_id)
    target_record = data_collector.get_user(target_id) or {}
    name = f"{target_record.get('first_name') or ''} {target_record.get('last_name') or ''}".strip() or f"User {target_id}"

    await update.message.reply_text(
        f"✅ *User Approved*: {name} (`{target_id}`) now has access to chat with Nino.",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        await context.bot.send_message(
            chat_id=target_id,
            text=(
                f"🎉 *Access Approved!*\n\n"
                f"Hey {target_record.get('first_name') or 'there'}, the owner has approved your access to Nino!\n"
                f"You can now send messages, voice notes, photos, and files. Tap /start to begin."
            ),
            reply_markup=get_guest_keyboard(target_id),
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.warning("Could not send approval message to %d: %s", target_id, e)


async def deny_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Deny or revoke a user's access to Nino (Owner only)."""
    user = update.effective_user
    if not is_owner(user.id if user else None):
        await update.message.reply_text("🔒 *Owner Only*: User management is restricted to the bot owner.")
        return

    args = context.args
    if not args:
        await update.message.reply_text("Usage: `/deny <user_id>`\nExample: `/deny 1977731223`", parse_mode=ParseMode.MARKDOWN)
        return

    try:
        target_id = int(args[0])
    except ValueError:
        await update.message.reply_text("⚠️ Invalid User ID. Must be an integer (e.g. `/deny 1977731223`).", parse_mode=ParseMode.MARKDOWN)
        return

    if is_owner(target_id):
        await update.message.reply_text("⚠️ You cannot deny access to the bot owner.")
        return

    data_collector.set_access_status(target_id, "denied")
    _pending_notifications_sent.discard(target_id)
    target_record = data_collector.get_user(target_id) or {}
    name = f"{target_record.get('first_name') or ''} {target_record.get('last_name') or ''}".strip() or f"User {target_id}"

    await update.message.reply_text(
        f"❌ *User Denied*: Access revoked for {name} (`{target_id}`).",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        await context.bot.send_message(
            chat_id=target_id,
            text="⛔ *Access Revoked*: Your access to this Nino bot has been denied by the owner.",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.warning("Could not send denial message to %d: %s", target_id, e)


async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all registered bot users, or inspect a specific user if argument/reply provided (Owner only)."""
    user = update.effective_user
    user_id = user.id if user else None
    cmd_name = "/user" if (update.message and update.message.text and update.message.text.startswith("/user") and not update.message.text.startswith("/users")) else "/users"

    if not is_owner(user_id):
        data_collector.log_interaction(user, message_type="blocked", content=cmd_name, is_owner=False)
        reply_msg = f"🔒 *Owner Only*: The {cmd_name} command is restricted to the bot owner."
        data_collector.log_reply(user=user, content=reply_msg, reply_to_type=f"command:{cmd_name}", is_owner=False)
        await update.message.reply_text(reply_msg, parse_mode=ParseMode.MARKDOWN)
        return

    # If argument (e.g. /user 1977731223 or /user @username) or reply message provided, delegate to whois inspection
    if (context.args and len(context.args) > 0) or (update.message and update.message.reply_to_message):
        await whois_command(update, context)
        return

    data_collector.log_interaction(user, message_type="command", content=cmd_name, is_owner=True)

    users = data_collector.list_users()
    if not users:
        empty_msg = "👥 <b>No users recorded yet.</b>"
        data_collector.log_reply(user=user, content=empty_msg, reply_to_type=f"command:{cmd_name}", is_owner=True)
        await update.message.reply_text(empty_msg, parse_mode=ParseMode.HTML)
        return

    owner_list = []
    approved_list = []
    pending_list = []
    denied_list = []
    keyboard = []

    for uid_str, data in users.items():
        try:
            uid = int(uid_str)
        except ValueError:
            uid = uid_str
        raw_name = f"{data.get('first_name') or ''} {data.get('last_name') or ''}".strip() or f"User {uid}"
        name = html.escape(raw_name)
        raw_uname = data.get("username")
        uname = f"@{html.escape(raw_uname)}" if raw_uname else "No username"
        phone = html.escape(str(data.get("phone_number") or "No phone"))
        msgs = data.get("total_messages", 0)

        if is_owner(uid) or data.get("role") == "owner":
            owner_list.append(f"• 👑 <b>{name}</b> (<code>{uid}</code>) - {uname} | 📞 <code>{phone}</code> ({msgs} msgs)")
        else:
            status = data.get("access_status", "pending")
            entry = f"• <b>{name}</b> (<code>{uid}</code>) - {uname} | 📞 <code>{phone}</code> ({msgs} msgs)"
            if status == "approved":
                approved_list.append(entry)
                keyboard.append([
                    InlineKeyboardButton(f"❌ Revoke {raw_name[:12]}", callback_data=f"deny_user:{uid}"),
                ])
            elif status == "denied":
                denied_list.append(entry)
                keyboard.append([
                    InlineKeyboardButton(f"✅ Unban {raw_name[:12]}", callback_data=f"approve_user:{uid}"),
                ])
            else:  # pending
                pending_list.append(entry)
                keyboard.append([
                    InlineKeyboardButton(f"✅ Approve {raw_name[:12]}", callback_data=f"approve_user:{uid}"),
                    InlineKeyboardButton(f"❌ Deny", callback_data=f"deny_user:{uid}"),
                ])
                if not data.get("phone_number"):
                    keyboard.append([
                        InlineKeyboardButton(f"📱 Ask Contact ({raw_name[:12]})", callback_data=f"ask_contact:{uid}")
                    ])

    text_parts = ["👥 <b>Nino User Directory</b>:\n"]
    if owner_list:
        text_parts.append("<b>Owner</b>:\n" + "\n".join(owner_list) + "\n")
    if pending_list:
        text_parts.append("⏳ <b>Pending Approval</b>:\n" + "\n".join(pending_list) + "\n")
    if approved_list:
        text_parts.append("✅ <b>Approved Users</b>:\n" + "\n".join(approved_list) + "\n")
    if denied_list:
        text_parts.append("❌ <b>Denied Users</b>:\n" + "\n".join(denied_list) + "\n")

    text_parts.append("💡 <i>Use <code>/approve &lt;id&gt;</code> or <code>/deny &lt;id&gt;</code> to change access anytime.</i>")
    message_text = "\n".join(text_parts)
    markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    data_collector.log_reply(user=user, content=message_text, reply_to_type=f"command:{cmd_name}", is_owner=True)

    try:
        await update.message.reply_text(message_text, reply_markup=markup, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.warning("Failed sending HTML users list, falling back to plain text: %s", e)
        clean_text = re.sub(r"<[^>]+>", "", message_text)
        await update.message.reply_text(clean_text, reply_markup=markup)


async def pending_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View and manage pending user access requests (Owner only)."""
    user = update.effective_user
    user_id = user.id if user else None
    if not is_owner(user_id):
        data_collector.log_interaction(user, message_type="blocked", content="/pending", is_owner=False)
        reply_msg = "🔒 *Owner Only*: User management is restricted to the bot owner."
        data_collector.log_reply(user=user, content=reply_msg, reply_to_type="command:/pending", is_owner=False)
        await update.message.reply_text(reply_msg, parse_mode=ParseMode.MARKDOWN)
        return

    data_collector.log_interaction(user, message_type="command", content="/pending", is_owner=True)
    pending = data_collector.get_pending_users()
    if not pending:
        msg = "✅ <b>No Pending Requests</b>: All users have been approved or processed."
        data_collector.log_reply(user=user, content=msg, reply_to_type="command:/pending", is_owner=True)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        return

    lines = ["⏳ <b>Pending User Requests</b>:\n"]
    keyboard = []
    for p in pending:
        p_uid = p.get("user_id")
        raw_name = f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip() or f"User {p_uid}"
        p_name = html.escape(raw_name)
        raw_uname = p.get("username")
        uname = f"@{html.escape(raw_uname)}" if raw_uname else "No username"
        phone = html.escape(str(p.get("phone_number") or "No phone"))
        lines.append(f"• <b>{p_name}</b> (<code>{p_uid}</code>) - {uname} | 📞 <code>{phone}</code>")
        keyboard.append([
            InlineKeyboardButton(f"✅ Approve {raw_name[:12]}", callback_data=f"approve_user:{p_uid}"),
            InlineKeyboardButton(f"❌ Deny", callback_data=f"deny_user:{p_uid}"),
        ])
        if not p.get("phone_number"):
            keyboard.append([
                InlineKeyboardButton(f"📱 Ask Contact ({raw_name[:12]})", callback_data=f"ask_contact:{p_uid}")
            ])

    msg_text = "\n".join(lines)
    data_collector.log_reply(user=user, content=msg_text, reply_to_type="command:/pending", is_owner=True)
    try:
        await update.message.reply_text(
            msg_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.warning("Failed sending HTML pending list, falling back: %s", e)
        await update.message.reply_text(
            re.sub(r"<[^>]+>", "", msg_text),
            reply_markup=InlineKeyboardMarkup(keyboard),
        )


async def request_contact_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a request to a user asking them to approve sharing their contact details and name (Owner only)."""
    user = update.effective_user
    if not is_owner(user.id if user else None):
        await update.message.reply_text("🔒 *Owner Only*: User management is restricted to the bot owner.")
        return

    args = context.args
    if not args:
        pending = data_collector.get_pending_users()
        unverified = [p for p in pending if not p.get("phone_number")]
        if not unverified:
            await update.message.reply_text(
                "Usage: `/requestcontact <user_id>`\n\nAll pending users have already shared their contact details.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        lines = ["📱 *Pending Users Without Contact Details*:\n"]
        keyboard = []
        for p in unverified:
            p_uid = p.get("user_id")
            p_name = f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip() or f"User {p_uid}"
            uname = f"@{p.get('username')}" if p.get("username") else "No username"
            lines.append(f"• *{p_name}* (`{p_uid}`) - {uname}")
            keyboard.append([
                InlineKeyboardButton(f"📱 Request from {p_name[:12]}", callback_data=f"ask_contact:{p_uid}")
            ])

        await update.message.reply_text(
            "\n".join(lines) + "\n\nTap a button below or type `/requestcontact <user_id>`:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    try:
        target_id = int(args[0])
    except ValueError:
        await update.message.reply_text("⚠️ Invalid User ID. Must be an integer (e.g. `/requestcontact 1977731223`).", parse_mode=ParseMode.MARKDOWN)
        return

    target_record = data_collector.get_user(target_id) or {}
    name = f"{target_record.get('first_name') or ''} {target_record.get('last_name') or ''}".strip() or f"User {target_id}"

    try:
        await context.bot.send_message(
            chat_id=target_id,
            text=(
                "👋 *Contact Verification Required by Owner*\n\n"
                "The bot owner has requested that you approve sharing your contact details (phone number) and name to proceed with your access request.\n\n"
                "Please tap the button below to approve and share your contact details:"
            ),
            reply_markup=get_contact_request_reply_markup(),
            parse_mode=ParseMode.MARKDOWN,
        )
        await update.message.reply_text(
            f"📨 *Contact Request Sent*: Successfully asked {name} (`{target_id}`) to approve sharing their contact details and name.",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.warning("Could not send contact request to %d: %s", target_id, e)
        await update.message.reply_text(
            f"⚠️ Could not send contact request to {name} (`{target_id}`): {e}",
            parse_mode=ParseMode.MARKDOWN,
        )


async def send_message_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a direct message to a specific user on behalf of the owner (Owner only).

    Usage:
      /send <user_id|@username> <message text>
      Or reply to any user message with: /send <message text>
    """
    user = update.effective_user
    user_id = user.id if user else None
    if not is_owner(user_id):
        data_collector.log_interaction(user, message_type="blocked", content="/send", is_owner=False)
        reply_msg = "🔒 *Owner Only*: Sending messages to users via the bot is restricted to the bot owner."
        data_collector.log_reply(user=user, content=reply_msg, reply_to_type="command:/send", is_owner=False)
        await update.message.reply_text(reply_msg, parse_mode=ParseMode.MARKDOWN)
        return

    reply_msg = update.message.reply_to_message if update.message else None
    target_id: Optional[int] = None
    target_name: str = ""
    message_text: str = ""

    if reply_msg and reply_msg.from_user:
        target_id = reply_msg.from_user.id
        raw_name = f"{reply_msg.from_user.first_name or ''} {reply_msg.from_user.last_name or ''}".strip() or reply_msg.from_user.username or f"User {target_id}"
        target_name = raw_name
        message_text = " ".join(context.args).strip() if context.args else ""
    elif context.args and len(context.args) >= 2:
        raw_target = context.args[0].strip()
        message_text = " ".join(context.args[1:]).strip()

        if raw_target.startswith("@"):
            uname = raw_target[1:].lower()
            for uid_str, udata in data_collector.list_users().items():
                if (udata.get("username") or "").lower() == uname:
                    try:
                        target_id = int(uid_str)
                    except ValueError:
                        pass
                    break
        elif raw_target.lstrip("-").isdigit():
            target_id = int(raw_target)
        else:
            uname = raw_target.lower()
            for uid_str, udata in data_collector.list_users().items():
                if (udata.get("username") or "").lower() == uname:
                    try:
                        target_id = int(uid_str)
                    except ValueError:
                        pass
                    break

    if not target_id or not message_text:
        usage = (
            "ℹ️ <b>Usage of /send:</b>\n\n"
            "• <code>/send &lt;user_id&gt; &lt;message&gt;</code>\n"
            "• <code>/send @username &lt;message&gt;</code>\n"
            "• Or reply to any user's message with <code>/send &lt;message&gt;</code>\n\n"
            "<b>Example:</b>\n"
            "<code>/send 1977731223 Hey, your access has been approved!</code>"
        )
        data_collector.log_reply(user=user, content=usage, reply_to_type="command:/send", is_owner=True)
        await update.message.reply_text(usage, parse_mode=ParseMode.HTML)
        return

    target_record = data_collector.get_user(target_id) or {}
    if not target_name:
        t_fname = target_record.get("first_name") or ""
        t_lname = target_record.get("last_name") or ""
        target_name = f"{t_fname} {t_lname}".strip() or (f"@{target_record['username']}" if target_record.get("username") else f"User {target_id}")

    recipient_text = (
        "💬 <b>Message from Bot Owner:</b>\n\n"
        f"{html.escape(message_text)}\n\n"
        "<i>(You can reply directly to this message)</i>"
    )

    try:
        await context.bot.send_message(
            chat_id=target_id,
            text=recipient_text,
            parse_mode=ParseMode.HTML,
        )

        data_collector.log_interaction(
            user,
            message_type="owner_send",
            content=f"To {target_id} ({target_name}): {message_text}",
            is_owner=True,
            metadata={"target_id": target_id, "target_name": target_name},
        )

        class DummyRecipient:
            def __init__(self, uid: int, uname: Optional[str]):
                self.id = uid
                self.username = uname

        data_collector.log_reply(
            user=DummyRecipient(target_id, target_record.get("username")),
            content=message_text,
            reply_to_type="owner_direct_message",
            is_owner=True,
            metadata={"sent_by_owner": True, "target_id": target_id},
        )

        confirm_msg = (
            f"✅ <b>Message Delivered</b> to <b>{html.escape(target_name)}</b> (<code>{target_id}</code>):\n\n"
            f"<i>\"{html.escape(message_text)}\"</i>"
        )
        await update.message.reply_text(confirm_msg, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.warning("Failed sending owner message to %s: %s", target_id, e)
        err_msg = f"⚠️ <b>Delivery Failed</b> to {html.escape(target_name)} (<code>{target_id}</code>):\n<code>{html.escape(str(e))}</code>"
        await update.message.reply_text(err_msg, parse_mode=ParseMode.HTML)


async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Broadcast an announcement to all registered users (Owner only).

    Usage:
      /broadcast <message text>
    """
    user = update.effective_user
    user_id = user.id if user else None
    if not is_owner(user_id):
        data_collector.log_interaction(user, message_type="blocked", content="/broadcast", is_owner=False)
        reply_msg = "🔒 *Owner Only*: Broadcasting is restricted to the bot owner."
        data_collector.log_reply(user=user, content=reply_msg, reply_to_type="command:/broadcast", is_owner=False)
        await update.message.reply_text(reply_msg, parse_mode=ParseMode.MARKDOWN)
        return

    message_text = " ".join(context.args).strip() if context.args else ""
    if not message_text:
        usage = (
            "📢 <b>Usage of /broadcast:</b>\n\n"
            "<code>/broadcast &lt;message text&gt;</code>\n\n"
            "<b>Example:</b>\n"
            "<code>/broadcast Nino will undergo a quick 5-minute update tonight!</code>"
        )
        data_collector.log_reply(user=user, content=usage, reply_to_type="command:/broadcast", is_owner=True)
        await update.message.reply_text(usage, parse_mode=ParseMode.HTML)
        return

    users = data_collector.list_users()
    recipients = [
        uid_str for uid_str, data in users.items()
        if not is_owner(int(uid_str) if uid_str.lstrip("-").isdigit() else None)
        and data.get("role") != "owner"
    ]

    if not recipients:
        await update.message.reply_text("👥 No guest users registered to receive broadcasts.", parse_mode=ParseMode.HTML)
        return

    status_msg = await update.message.reply_text(f"🚀 Broadcasting message to {len(recipients)} user(s)...", parse_mode=ParseMode.HTML)

    broadcast_text = (
        "📢 <b>Announcement from Bot Owner:</b>\n\n"
        f"{html.escape(message_text)}"
    )

    sent_count = 0
    fail_count = 0
    delivered_users: list[dict[str, Any]] = []
    failed_users: list[dict[str, Any]] = []
    delivered_labels: list[str] = []

    class DummyRecipient:
        def __init__(self, uid: int, uname: Optional[str]):
            self.id = uid
            self.username = uname

    for uid_str in recipients:
        u_data = users.get(uid_str, {})
        raw_name = f"{u_data.get('first_name') or ''} {u_data.get('last_name') or ''}".strip() or (f"@{u_data['username']}" if u_data.get("username") else f"User {uid_str}")
        try:
            target_id = int(uid_str)
            await context.bot.send_message(
                chat_id=target_id,
                text=broadcast_text,
                parse_mode=ParseMode.HTML,
            )
            sent_count += 1
            user_entry = {
                "user_id": target_id,
                "name": raw_name,
                "username": u_data.get("username"),
            }
            delivered_users.append(user_entry)
            delivered_labels.append(f"{target_id} ({raw_name})")

            # Log to reply.jsonl
            data_collector.log_reply(
                user=DummyRecipient(target_id, u_data.get("username")),
                content=message_text,
                reply_to_type="owner_broadcast",
                is_owner=True,
                metadata={"sent_by_owner": True, "target_id": target_id, "name": raw_name},
            )

            # Log to interactions.jsonl for the recipient
            data_collector.log_interaction(
                DummyRecipient(target_id, u_data.get("username")),
                message_type="broadcast_received",
                content=f"Received broadcast from owner: {message_text}",
                is_owner=False,
                metadata={"broadcast_by": user_id, "sent_by_owner": True},
            )
        except Exception as e:
            logger.debug("Broadcast failed for %s: %s", uid_str, e)
            fail_count += 1
            failed_users.append({
                "user_id": int(uid_str) if uid_str.lstrip("-").isdigit() else uid_str,
                "name": raw_name,
                "username": u_data.get("username"),
                "error": str(e),
            })

    delivered_summary = ", ".join(delivered_labels) if delivered_labels else "None"
    broadcast_log_content = f"Broadcast to {sent_count} user(s) [{delivered_summary}]: {message_text}"

    data_collector.log_interaction(
        user,
        message_type="owner_broadcast",
        content=broadcast_log_content,
        is_owner=True,
        metadata={
            "sent_count": sent_count,
            "fail_count": fail_count,
            "total": len(recipients),
            "delivered_to": delivered_users,
            "failed_to": failed_users,
            "recipient_ids": [d["user_id"] for d in delivered_users],
        },
    )

    report_lines = [
        "📢 <b>Broadcast Complete</b>\n",
        f"• <b>Delivered</b>: {sent_count} user(s)",
    ]
    if delivered_labels:
        report_lines.append(f"• <b>Recipients</b>: {html.escape(delivered_summary)}")
    if failed_users:
        fail_summary = ", ".join(f"{f['user_id']} ({f['name']})" for f in failed_users)
        report_lines.append(f"• <b>Failed</b>: {fail_count} ({html.escape(fail_summary)})")

    report_lines.append(f"\n<i>Message sent:</i>\n\"{html.escape(message_text)}\"")
    await status_msg.edit_text("\n".join(report_lines), parse_mode=ParseMode.HTML)


def format_display_time(iso_str: Optional[str]) -> str:
    """Format an ISO timestamp to readable Indian Standard Time (IST) representation."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is not None:
            dt = dt.astimezone(IST)
        else:
            dt = dt.replace(tzinfo=IST)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_str[:19].replace("T", " ")


async def whois_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Extract and display user profile information (Owner only)."""
    user = update.effective_user
    user_id = user.id if user else None
    if not is_owner(user_id):
        data_collector.log_interaction(user, message_type="blocked", content="/whois", is_owner=False)
        reply_msg = "🔒 *Owner Only*: The /whois command is restricted to the bot owner."
        data_collector.log_reply(user=user, content=reply_msg, reply_to_type="command:/whois", is_owner=False)
        await update.message.reply_text(reply_msg, parse_mode=ParseMode.MARKDOWN)
        return

    data_collector.log_interaction(user, message_type="command", content="/whois", is_owner=True)

    # 1. Determine target user ID
    target_id: Optional[int] = None
    target_user: Any = None

    reply_msg = update.message.reply_to_message if update.message else None
    if reply_msg and reply_msg.from_user:
        target_user = reply_msg.from_user
        target_id = target_user.id
    elif context.args:
        raw_arg = context.args[0].strip()
        if raw_arg.startswith("@"):
            uname = raw_arg[1:].lower()
            for uid_str, udata in data_collector.list_users().items():
                if (udata.get("username") or "").lower() == uname:
                    try:
                        target_id = int(uid_str)
                    except ValueError:
                        pass
                    break
        elif raw_arg.lstrip("-").isdigit():
            target_id = int(raw_arg)
        else:
            uname = raw_arg.lower()
            for uid_str, udata in data_collector.list_users().items():
                if (udata.get("username") or "").lower() == uname:
                    try:
                        target_id = int(uid_str)
                    except ValueError:
                        pass
                    break
    else:
        # Default to caller
        target_user = user
        target_id = user_id

    if not target_id:
        usage_msg = (
            "ℹ️ <b>Usage of /whois</b>:\n\n"
            "• Reply to any user message with <code>/whois</code>\n"
            "• Or specify an ID or username: <code>/whois 1334848825</code> or <code>/whois @username</code>"
        )
        data_collector.log_reply(user=user, content=usage_msg, reply_to_type="command:/whois", is_owner=True)
        await update.message.reply_text(usage_msg, parse_mode=ParseMode.HTML)
        return

    # 2. Fetch Chat / User details from Telegram
    tg_chat = None
    try:
        tg_chat = await context.bot.get_chat(target_id)
    except Exception as e:
        logger.debug("get_chat failed for %s: %s", target_id, e)

    user_record = data_collector.get_user(target_id) or {}

    first_name = (
        getattr(tg_chat, "first_name", None)
        or getattr(target_user, "first_name", None)
        or user_record.get("first_name")
        or "Unknown"
    )
    last_name = (
        getattr(tg_chat, "last_name", None)
        or getattr(target_user, "last_name", None)
        or user_record.get("last_name")
        or ""
    )
    username = (
        getattr(tg_chat, "username", None)
        or getattr(target_user, "username", None)
        or user_record.get("username")
    )
    bio = getattr(tg_chat, "bio", None)

    is_owner_target = is_owner(target_id) or user_record.get("role") == "owner"
    access_status = "owner" if is_owner_target else user_record.get("access_status", "unknown")
    phone = user_record.get("phone_number")
    total_msgs = user_record.get("total_messages", 0)
    first_seen = user_record.get("first_seen")
    last_seen = user_record.get("last_seen")

    status_icon = "👑" if is_owner_target else ("✅" if access_status == "approved" else ("⏳" if access_status == "pending" else "⛔"))
    out_lines = [
        "👤 <b>User Information</b>\n",
        f"<b>Name:</b> <a href='tg://user?id={target_id}'>{html.escape(first_name)}</a>",
    ]
    if last_name:
        out_lines.append(f"<b>Suffix:</b> {html.escape(last_name)}")
    if username:
        out_lines.append(f"<b>Username:</b> @{html.escape(username)}")
    out_lines.append(f"<b>User ID:</b> <code>{target_id}</code>")

    if username:
        out_lines.append(f"<b>User Link:</b> <a href='https://t.me/{username}'>@{html.escape(username)}</a>")

    if bio:
        out_lines.append(f"<b>Bio:</b> {html.escape(bio)}")

    out_lines.append(f"\n<b>Nino Access:</b> {status_icon} <code>{access_status.upper()}</code>")
    if phone:
        out_lines.append(f"<b>Phone:</b> <code>{html.escape(phone)}</code>")
    if total_msgs:
        out_lines.append(f"<b>Messages Sent:</b> <code>{total_msgs}</code>")
    if first_seen:
        out_lines.append(f"<b>First Seen:</b> <code>{html.escape(format_display_time(first_seen))}</code>")
    if last_seen:
        out_lines.append(f"<b>Last Active:</b> <code>{html.escape(format_display_time(last_seen))}</code>")

    eff_chat = update.effective_chat
    if eff_chat and eff_chat.type in ("supergroup", "group", "channel"):
        try:
            member = await context.bot.get_chat_member(eff_chat.id, target_id)
            if member:
                out_lines.append(f"<b>Chat Role:</b> <code>{html.escape(str(member.status).capitalize())}</code>")
                joined_dt = getattr(member, "joined_date", None)
                if joined_dt:
                    if isinstance(joined_dt, (int, float)):
                        j_dt = datetime.fromtimestamp(joined_dt, tz=IST)
                    elif isinstance(joined_dt, datetime):
                        j_dt = joined_dt.astimezone(IST) if joined_dt.tzinfo else joined_dt.replace(tzinfo=IST)
                    else:
                        j_dt = None
                    if j_dt:
                        out_lines.append(f"<b>Joined on:</b> <code>{html.escape(j_dt.strftime('%Y-%m-%d %H:%M:%S'))}</code>")
        except Exception:
            pass

    message_out_str = "\n".join(out_lines)

    # 3. Check for profile photo
    photo_file_id: Optional[str] = None
    try:
        photos = await context.bot.get_user_profile_photos(target_id, limit=1)
        if photos and photos.total_count > 0 and photos.photos:
            photo_file_id = photos.photos[0][-1].file_id
    except Exception as e:
        logger.debug("Could not retrieve profile photos for %s: %s", target_id, e)

    data_collector.log_reply(user=user, content=message_out_str, reply_to_type="command:/whois", is_owner=True)

    if photo_file_id:
        try:
            await update.message.reply_photo(
                photo=photo_file_id,
                caption=message_out_str,
                parse_mode=ParseMode.HTML,
            )
            return
        except Exception as e:
            logger.debug("Failed sending profile photo reply, falling back to text: %s", e)

    await update.message.reply_text(
        text=message_out_str,
        parse_mode=ParseMode.HTML,
    )


# ==============================================================================
# Interactive Callback Query Handler
# ==============================================================================
async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button clicks for dashboard, confirmations, and quick actions."""
    query = update.callback_query
    await query.answer()

    user = query.from_user
    user_id = user.id if user else None
    owner = is_owner(user_id)
    data = query.data

    if data == "btn_weather":
        report = await fetch_weather_report(DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_CITY)
        await query.message.reply_text(report, parse_mode=ParseMode.MARKDOWN)
        return

    if data == "btn_time":
        await query.message.reply_text(clock.get_clock_report(), parse_mode=ParseMode.MARKDOWN)
        return

    if data == "btn_reminders":
        chat_id = query.message.chat_id if query.message else None
        reminders = reminder_mgr.list_reminders(chat_id=chat_id, user_id=user_id)
        if not reminders:
            await query.message.reply_text(
                "⏰ *No Active Reminders*\n\n"
                "You have no pending reminders.\n"
                "Schedule one with `/remind <time> <task>` or simply say _\"remind me in 5 minutes to call Alex\"_.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        lines = ["⏰ *Your Active Reminders*:\n"]
        keyboard = []
        for r in reminders[:10]:
            try:
                dt = datetime.fromisoformat(r["trigger_time"])
                disp = dt.strftime("%I:%M %p (%d %b)")
            except Exception:
                disp = r["trigger_time"]
            lines.append(f"• **#{r['id']}**: {r['message']}\n  🕒 `{disp}`\n")
            keyboard.append([
                InlineKeyboardButton(f"❌ Cancel #{r['id']}", callback_data=f"cancel_rem:{r['id']}"),
                InlineKeyboardButton(f"⏰ +15m #{r['id']}", callback_data=f"snooze_rem:{r['id']}:15"),
            ])
        await query.message.reply_text(
            "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Reminder Callback Actions (Cancel / Snooze / Dismiss)
    if data.startswith("cancel_rem:"):
        rem_id = int(data.split(":")[1])
        success = reminder_mgr.cancel_reminder(rem_id, user_id=user_id if not owner else None)
        if success:
            await query.edit_message_text(f"✅ Reminder #{rem_id} has been cancelled.")
        else:
            await query.message.reply_text(f"⚠️ Could not cancel reminder #{rem_id}.")
        return

    if data.startswith("snooze_rem:"):
        parts = data.split(":")
        rem_id = int(parts[1])
        mins = int(parts[2]) if len(parts) > 2 else 15
        new_dt = reminder_mgr.snooze_reminder(rem_id, minutes=mins, user_id=user_id if not owner else None)
        if new_dt:
            disp = new_dt.strftime("%I:%M %p")
            await query.message.reply_text(f"⏰ Reminder #{rem_id} snoozed by {mins} minutes (Due at `{disp}`).", parse_mode=ParseMode.MARKDOWN)
        else:
            await query.message.reply_text(f"⚠️ Could not snooze reminder #{rem_id}.")
        return

    if data.startswith("dismiss_rem:"):
        rem_id = int(data.split(":")[1])
        reminder_mgr.mark_delivered(rem_id)
        await query.edit_message_text("✅ Reminder dismissed.")
        return

    if data == "btn_voice":
        current = get_user_voice_pref(user_id)
        new_state = not current
        set_user_voice_pref(user_id, new_state)
        status = "enabled 🔊" if new_state else "disabled 🔇 (text-only)"
        keyboard = get_control_keyboard(user_id) if owner else get_guest_keyboard(user_id)
        try:
            await query.edit_message_reply_markup(reply_markup=keyboard)
        except Exception:
            pass
        await query.message.reply_text(
            f"Voice replies are now *{status}*.\nUse `/voice on` or `/voice off` to toggle anytime.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    if data == "btn_help":
        await help_command(update, context)
        return

    if data in ("btn_request_contact", "btn_phone"):
        await query.message.reply_text(
            "📱 Please tap the button below to share your contact information:",
            reply_markup=get_contact_request_reply_markup(),
        )
        return

    # User Access Approvals & Denials
    if data.startswith("approve_user:"):
        if not owner:
            await query.message.reply_text("🔒 *Owner Only*: Only the owner can approve users.")
            return

        target_id = int(data.split(":")[1])
        data_collector.set_access_status(target_id, "approved")
        _pending_notifications_sent.discard(target_id)
        target_record = data_collector.get_user(target_id) or {}
        first_name = target_record.get("first_name") or f"User {target_id}"

        await query.edit_message_text(
            f"✅ *User Approved*\n\n"
            f"• *Name*: {first_name}\n"
            f"• *User ID*: `{target_id}`\n\n"
            f"This user is now authorized to chat and use Nino.",
            parse_mode=ParseMode.MARKDOWN,
        )

        try:
            await context.bot.send_message(
                chat_id=target_id,
                text=(
                    f"🎉 *Access Approved!*\n\n"
                    f"Hey {first_name}, the owner has approved your access to Nino!\n"
                    f"You can now send messages, voice notes, photos, and files. Tap /start to begin."
                ),
                reply_markup=get_guest_keyboard(target_id),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            logger.warning("Could not notify approved user %s: %s", target_id, e)
        return

    if data.startswith("deny_user:"):
        if not owner:
            await query.message.reply_text("🔒 *Owner Only*: Only the owner can deny users.")
            return

        target_id = int(data.split(":")[1])
        data_collector.set_access_status(target_id, "denied")
        _pending_notifications_sent.discard(target_id)
        target_record = data_collector.get_user(target_id) or {}
        first_name = target_record.get("first_name") or f"User {target_id}"

        await query.edit_message_text(
            f"❌ *User Denied*\n\n"
            f"• *Name*: {first_name}\n"
            f"• *User ID*: `{target_id}`\n\n"
            f"Access has been denied for this user.",
            parse_mode=ParseMode.MARKDOWN,
        )

        try:
            await context.bot.send_message(
                chat_id=target_id,
                text="⛔ *Access Request Denied*\n\nYour request to access Nino was not approved by the owner.",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            logger.warning("Could not notify denied user %s: %s", target_id, e)
        return

    if data.startswith("ask_contact:"):
        if not owner:
            await query.message.reply_text("🔒 *Owner Only*: Only the owner can request user contact details.")
            return

        target_id = int(data.split(":")[1])
        target_record = data_collector.get_user(target_id) or {}
        first_name = target_record.get("first_name") or f"User {target_id}"

        try:
            await context.bot.send_message(
                chat_id=target_id,
                text=(
                    "👋 *Contact Verification Required by Owner*\n\n"
                    "The bot owner has requested that you approve sharing your contact details (phone number) and name to proceed with your access request.\n\n"
                    "Please tap the button below to approve and share your contact card:"
                ),
                reply_markup=get_contact_request_reply_markup(),
                parse_mode=ParseMode.MARKDOWN,
            )
            await query.message.reply_text(
                f"📨 *Contact Request Sent*: Successfully asked {first_name} (`{target_id}`) to approve sharing their contact details and name.",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            logger.warning("Could not send contact request to %s: %s", target_id, e)
            await query.message.reply_text(
                f"⚠️ Could not deliver contact request to {first_name} (`{target_id}`): {e}",
                parse_mode=ParseMode.MARKDOWN,
            )
        return

    if data == "btn_manage_users":
        if not owner:
            await query.message.reply_text("🔒 *Owner Only*: User management is restricted to the bot owner.")
            return
        await users_command(update, context)
        return

    # Owner-only actions guard
    if not owner:
        await query.message.reply_text("🔒 *Owner Only*: This action is restricted to the bot owner.")
        return

    # Mode Selector
    if data == "btn_mode":
        keyboard = [
            [InlineKeyboardButton("✨ Companion", callback_data="set_mode:companion")],
            [InlineKeyboardButton("💼 Work", callback_data="set_mode:work")],
            [InlineKeyboardButton("🎯 Focus", callback_data="set_mode:focus")],
        ]
        await query.edit_message_text(
            f"Select persona mode (Current: *{agent.mode.upper()}*):",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    if data.startswith("set_mode:"):
        new_mode = data.split(":", 1)[1]
        agent.mode = new_mode
        await query.edit_message_text(f"🎭 Switched to *{new_mode.upper()}* mode.", parse_mode=ParseMode.MARKDOWN)
        return

    # Memories
    if data == "btn_memories":
        memories = agent.brain.list_accessible_memories()
        if not memories:
            await query.edit_message_text("🧠 No memories stored yet. Use `/remember <fact>` to save one!")
            return
        lines = ["🧠 *Accessible Memories* (Profile & Telegram):"]
        for m in memories[:10]:
            lines.append(f"• {m['text']}")
        await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        return

    # Status
    if data == "btn_status":
        ollama_online = await agent.is_ollama_online()
        active = "Ollama (Local)" if ollama_online else "Gemini (Cloud Fallback)"
        await query.edit_message_text(
            f"⚡ *System Check*:\n• Engine: {active}\n• Mode: {agent.mode.upper()}\n• Voice: {'ON' if get_user_voice_pref(user_id) else 'OFF (Text-Only)'}",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Clear
    if data == "btn_clear":
        agent.reset_conversation(user_id=user_id)
        await query.edit_message_text("🧹 *Chat Cleared*: Ready for a fresh topic.", parse_mode=ParseMode.MARKDOWN)
        return

    # Vault Confirmations
    if data == "vault_cancel":
        await query.edit_message_text("❌ Operation cancelled.")
        return

    if data.startswith("vault_read:"):
        filename = data.split(":", 1)[1]
        result = agent.vault.read_file(filename)
        if "error" in result:
            await query.edit_message_text(f"⚠️ Error: {result['error']}")
        else:
            snippet = result["content"]
            if len(snippet) > 3000:
                snippet = snippet[:3000] + "\n\n...[content truncated]"
            await query.edit_message_text(
                f"📄 *Vault File*: `{filename}`\n\n```\n{snippet}\n```",
                parse_mode=ParseMode.MARKDOWN,
            )


# ==============================================================================
# Message Handlers (Text, Voice, Photo, Document, Location)
# ==============================================================================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming text messages, inspect and scan links, log interaction, and reply."""
    if not update.message or not update.message.text:
        return

    user = update.effective_user
    user_id = user.id if user else None
    user_text = update.message.text.strip()
    chat_id = update.effective_chat.id

    # Handle dismiss button from contact reply keyboard before blocking
    if user_text.lower() in ("dismiss", "dismiss ❌", "cancel"):
        dismiss_text = "Contact sharing dismissed."
        data_collector.log_reply(user=user, content=dismiss_text, reply_to_type="dismiss", is_owner=is_owner(user_id))
        await update.message.reply_text(dismiss_text, reply_markup=ReplyKeyboardRemove())
        return

    if not await check_user_access_or_prompt(update, context):
        return

    owner = is_owner(user_id)

    # Check for URLs in text message
    urls = link_inspector.extract_urls(user_text)
    augmented_prompt = user_text
    raw_display: Optional[str] = None

    if urls:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        web_snippets = []
        clean_user_query = user_text
        for u in urls:
            clean_user_query = clean_user_query.replace(u, "").strip()

        for url in urls[:2]:  # Check up to 2 links per message for safety
            # VirusTotal & Security Pipeline check
            scan_res = await security_scanner.scan_url(url)
            if not scan_res.is_safe:
                blocked_msg = f"⚠️ *Security Alert*: Link failed safety verification and was blocked:\n• `{url}`\n• {scan_res.reason}"
                data_collector.record_user(user, is_owner=owner)
                data_collector.log_interaction(
                    user,
                    message_type="link_blocked",
                    content=f"Blocked URL: {url} ({scan_res.reason})",
                    is_owner=owner,
                    metadata=scan_res.details,
                )
                data_collector.log_reply(
                    user=user,
                    content=blocked_msg,
                    reply_to_type="link_blocked",
                    is_owner=owner,
                    metadata=scan_res.details,
                )
                await update.message.reply_text(
                    blocked_msg,
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            # Extract clean content from URL / YouTube transcript / Online PDF / Online Image
            extracted_url_text = await media_processor.extract_url(url)
            web_snippets.append(extracted_url_text)

            # Store active media session memory for ongoing conversational follow-ups
            media_type = "youtube" if ("youtube.com" in url or "youtu.be" in url) else "url"
            agent.set_media_context(user_id or 0, media_type, url, extracted_url_text)

            data_collector.log_interaction(
                user,
                message_type="link",
                content=f"Safe link scanned: {url}",
                is_owner=owner,
                metadata=scan_res.details,
            )

        content_block = "\n\n".join(web_snippets)

        if clean_user_query:
            augmented_prompt = (
                f"[Web Content Extracted from Shared Link(s)]:\n"
                f"{content_block}\n\n"
                f"User Question: {clean_user_query}\n"
                f"Please read the web content and answer the user's question directly, accurately, and thoroughly in your persona."
            )
            raw_display = f"[Shared Link(s)] {clean_user_query}"
        else:
            augmented_prompt = (
                f"[Web Content Extracted from Shared Link(s)]:\n"
                f"{content_block}\n\n"
                f"The user shared this link without a specific question. Provide a structured, helpful review of this webpage:\n"
                f"• **Title & Subject**: What is this page about?\n"
                f"• **Key Takeaways & Summary**: 3-5 core points, findings, or highlights from the content.\n"
                f"• **Notable Details / Quotes**: Important facts, figures, or data mentioned.\n"
                f"• **Next Steps**: Invite the user to ask any questions or request deeper explanation of specific sections."
            )
            raw_display = f"[Shared Link(s): {', '.join(urls)}]"
    else:
        # Check active media context for follow-up conversational questions
        media_ctx = agent.get_media_context(user_id or 0)
        if media_ctx:
            m_type = media_ctx["type"]
            m_name = media_ctx["filename_or_url"]
            m_content = media_ctx["extracted_content"]
            augmented_prompt = (
                f"[Context from recently uploaded {m_type} ({m_name})]:\n"
                f"{m_content}\n\n"
                f"User Question: {user_text}"
            )
            logger.info("Attached active media memory (%s: %s) to user query", m_type, m_name)
        else:
            augmented_prompt = user_text

    # Record user and interaction in bot_saved_data
    data_collector.record_user(user, is_owner=owner)
    data_collector.log_interaction(user, message_type="text", content=user_text, is_owner=owner)

    # Automatic extraction and saving of phone number from chat messages
    detected_phone = extract_phone_from_text(user_text)
    if detected_phone and user_id:
        data_collector.update_phone(user_id, detected_phone)
        logger.info("Auto-saved phone number %s for user %d from text", detected_phone, user_id)

    # Automatic extraction and saving of IP address from chat messages
    detected_ip = extract_ip_from_text(user_text)
    if detected_ip and user_id:
        data_collector.update_ip(user_id, detected_ip)
        logger.info("Auto-saved IP address %s for user %d from text", detected_ip, user_id)

    # Check for natural language reminder request
    rem_extracted = extract_reminder_from_text(user_text)
    if rem_extracted:
        time_spec, rem_msg = rem_extracted
        try:
            reminder = reminder_mgr.create_reminder(
                message=rem_msg,
                time_expression=time_spec,
                chat_id=chat_id,
                user_id=user_id or (AUTHORIZED_USER_ID or 0),
            )
            confirm_text = (
                f"⏰ *Reminder Scheduled!*\n\n"
                f"📌 *Task*: {reminder['message']}\n"
                f"🕒 *Alert at*: `{reminder['formatted_time']}` ({reminder['time_label']})\n\n"
                f"_Nino will send you a Telegram alert right here when it's due!_"
            )
            data_collector.log_interaction(
                user,
                message_type="reminder_created",
                content=f"Reminder created: '{rem_msg}' at {reminder['formatted_time']}",
                is_owner=owner,
            )
            await update.message.reply_text(confirm_text, parse_mode=ParseMode.MARKDOWN)

            voice_sent = False
            if get_user_voice_pref(user_id):
                spoken = f"Got it. I will remind you to {rem_msg} {reminder['time_label']}."
                with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_audio:
                    tmp_p = tmp_audio.name
                try:
                    await synthesize_voice(spoken, tmp_p, voice=TTS_VOICE)
                    with open(tmp_p, "rb") as vf:
                        await update.message.reply_voice(voice=vf)
                    voice_sent = True
                except Exception as e:
                    logger.warning("Could not synthesize voice reply for reminder: %s", e)
                finally:
                    if os.path.exists(tmp_p):
                        os.remove(tmp_p)

            data_collector.log_reply(
                user=user,
                content=confirm_text,
                reply_to_type="reminder_intent",
                voice_sent=voice_sent,
                is_owner=owner,
            )
            return
        except Exception as e:
            logger.warning("Failed to parse reminder intent '%s': %s", user_text, e)

    # Display typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Get response from Nino Hybrid Agent with role context
    try:
        reply_text, engine = await agent.chat(
            augmented_prompt,
            user_id=user_id,
            is_owner=owner,
            raw_prompt=raw_display if urls else None,
        )
    except Exception as e:
        logger.error("Error generating agent response: %s", e)
        reply_text = f"⚠️ *Error*: Encountered an issue while reasoning: {e}"
        engine = "error"

    # Send text response
    try:
        await update.message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(reply_text)

    # If voice replies are enabled, synthesize audio as well
    voice_sent = False
    if get_user_voice_pref(user_id) and len(reply_text) < 600:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_audio:
            tmp_path = tmp_audio.name

        try:
            await synthesize_voice(reply_text, tmp_path, voice=TTS_VOICE)
            with open(tmp_path, "rb") as voice_file:
                await update.message.reply_voice(voice=voice_file)
            voice_sent = True
        except Exception as e:
            logger.warning("Could not synthesize voice reply: %s", e)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    data_collector.log_reply(
        user=user,
        content=reply_text,
        reply_to_type="text",
        voice_sent=voice_sent,
        is_owner=owner,
        metadata={"engine": engine},
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming voice notes: archive media, transcribe, reason, and reply."""
    message = update.message
    if not message or (not message.voice and not message.audio):
        return

    user = update.effective_user
    user_id = user.id if user else None
    if not await check_user_access_or_prompt(update, context):
        return
    owner = is_owner(user_id)
    chat_id = update.effective_chat.id
    voice_obj = message.voice or message.audio

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as in_audio:
        in_path = in_audio.name

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as out_audio:
        out_path = out_audio.name

    try:
        # 1. Download voice note
        tg_file = await voice_obj.get_file()
        await tg_file.download_to_drive(custom_path=in_path)

        # Archive voice note into bot_saved_data/media/{user_id}/
        with open(in_path, "rb") as vf:
            voice_bytes = vf.read()
        saved_media_path = data_collector.save_media(
            user_id or 0,
            voice_bytes,
            f"voice_{getattr(voice_obj, 'file_unique_id', 'audio')}.ogg",
        )

        # 2. Transcribe via faster-whisper
        transcription = await transcribe_voice(in_path, model_size=WHISPER_MODEL)
        if not transcription:
            transcription = "[unintelligible audio]"

        logger.info("Transcribed voice note: '%s'", transcription)

        # Record interaction with media archive reference
        data_collector.record_user(user, is_owner=owner)
        data_collector.log_interaction(
            user,
            message_type="voice",
            content=f"Transcription: {transcription}",
            media_path=saved_media_path,
            is_owner=owner,
            metadata={"duration": getattr(voice_obj, "duration", None)},
        )

        if transcription == "[unintelligible audio]":
            unintelligible_msg = "🎙 *Nino*: I heard your voice message, but could not detect any clear speech."
            data_collector.log_reply(user=user, content=unintelligible_msg, reply_to_type="voice", is_owner=owner)
            await message.reply_text(unintelligible_msg)
            return

        # 3. Reason via Nino Hybrid Agent
        agent_reply, engine = await agent.chat(transcription, user_id=user_id, is_owner=owner)

        caption = f"🎙 *You*: \"{transcription}\"\n\n💬 *Nino*: {agent_reply}"
        if len(caption) > 1000:
            caption = caption[:990] + "..."

        # If voice replies are disabled (default), reply with text only
        if not get_user_voice_pref(user_id):
            data_collector.log_reply(
                user=user,
                content=agent_reply,
                reply_to_type="voice",
                voice_sent=False,
                is_owner=owner,
                metadata={"engine": engine, "transcription": transcription, "caption": caption},
            )
            await message.reply_text(
                caption,
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        # 4. If voice replies are toggled ON, synthesize voice response
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)
        await synthesize_voice(agent_reply, out_path, voice=TTS_VOICE)

        # 5. Send back native voice note
        with open(out_path, "rb") as v_out:
            await message.reply_voice(
                voice=v_out,
                caption=caption,
                parse_mode=ParseMode.MARKDOWN,
            )
        data_collector.log_reply(
            user=user,
            content=agent_reply,
            reply_to_type="voice",
            voice_sent=True,
            is_owner=owner,
            metadata={"engine": engine, "transcription": transcription, "caption": caption},
        )

    except Exception as e:
        logger.error("Error processing voice message: %s", e)
        err_msg = f"⚠️ *Voice processing error*: {e}"
        data_collector.log_reply(user=user, content=err_msg, reply_to_type="voice", is_owner=owner)
        await message.reply_text(err_msg, parse_mode=ParseMode.MARKDOWN)

    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze images sent by users with VirusTotal pre-scan, vision extraction, and media session memory."""
    message = update.message
    if not message or not message.photo:
        return

    user = update.effective_user
    user_id = user.id if user else None
    if not await check_user_access_or_prompt(update, context):
        return

    owner = is_owner(user_id)
    chat_id = update.effective_chat.id
    user_caption = (message.caption or "").strip()
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Get the highest resolution photo
    photo = message.photo[-1]
    photo_file = await photo.get_file()

    filename = f"photo_{photo.file_unique_id}.jpg"
    byte_arr = bytearray()
    await photo_file.download_as_bytearray(byte_arr)
    photo_bytes = bytes(byte_arr)

    # 1. Automated malware & reputation scan BEFORE LLM processing
    scan_res = await security_scanner.scan_file_bytes(photo_bytes, filename=filename, expected_mime="image/jpeg")
    if not scan_res.is_safe:
        data_collector.record_user(user, is_owner=owner)
        data_collector.log_interaction(
            user,
            message_type="file_blocked",
            content=f"Blocked photo: {filename} ({scan_res.reason})",
            is_owner=owner,
            metadata=scan_res.details,
        )
        await message.reply_text("⚠️ Security Alert: File flagged as unsafe and discarded.")
        return

    # Archive photo into bot_saved_data/media/{user_id}/
    saved_media_path = data_collector.save_media(
        user_id or 0,
        photo_bytes,
        filename,
    )
    data_collector.record_user(user, is_owner=owner)
    data_collector.log_interaction(
        user,
        message_type="photo",
        content=user_caption or "Photo uploaded",
        media_path=saved_media_path,
        is_owner=owner,
        metadata={"file_size": photo.file_size, "width": photo.width, "height": photo.height},
    )

    # 2. Extract OCR / Vision Description via Ollama / Tesseract fallback
    ocr_prompt = user_caption if user_caption else (
        "Analyze this image thoroughly:\n"
        "1. Visual Summary: Describe objects, scene, people, colors, layout, and context.\n"
        "2. All Text & Data: Transcribe any text, titles, numbers, labels, or data verbatim."
    )
    extracted_content = await media_processor.extract_image(photo_bytes, filename=filename, prompt=ocr_prompt)

    # 3. Store active media session memory for ongoing conversational follow-ups
    agent.set_media_context(user_id or 0, "image", filename, extracted_content)

    if user_caption:
        query_prompt = (
            f"The user uploaded an image ('{filename}') with the question:\n\"{user_caption}\"\n\n"
            f"Examine the visual details and transcribed text in the image context carefully, and answer their question directly, thoroughly, and accurately in your persona."
        )
    else:
        query_prompt = (
            f"The user shared an image ('{filename}') without a question.\n"
            f"Provide a structured, engaging breakdown of what is shown:\n"
            f"• **Visual Overview**: Describe what is depicted in the image.\n"
            f"• **Text & Data Extracted**: List all text, numbers, labels, or code found.\n"
            f"• **Insights**: Add any interesting observations, and invite them to ask specific questions about it."
        )

    raw_display = f"[Uploaded Image '{filename}'] {user_caption}" if user_caption else f"[Uploaded Image '{filename}']"

    try:
        reply_text, engine = await agent.chat(
            query_prompt,
            image_bytes=photo_bytes,
            user_id=user_id,
            is_owner=owner,
            raw_prompt=raw_display,
        )
        data_collector.log_reply(
            user=user,
            content=reply_text,
            reply_to_type="photo",
            is_owner=owner,
            metadata={"engine": engine, "filename": filename},
        )
        await message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error("Vision processing error: %s", e)
        err_msg = f"⚠️ Could not analyze image: {e}"
        data_collector.log_reply(user=user, content=err_msg, reply_to_type="photo", is_owner=owner)
        await message.reply_text(err_msg)


async def handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze incoming stickers (emotion, visual content, pack) and accurately reply with a matching sticker."""
    message = update.message
    if not message or not message.sticker:
        return

    user = update.effective_user
    user_id = user.id if user else None
    if not await check_user_access_or_prompt(update, context):
        return

    owner = is_owner(user_id)
    chat_id = update.effective_chat.id
    sticker = message.sticker
    emoji = sticker.emoji or ""
    pack_name = sticker.set_name or ""
    incoming_uid = sticker.file_unique_id

    # Broadcast choosing_sticker action to Telegram client
    choose_action = getattr(ChatAction, "CHOOSE_STICKER", ChatAction.TYPING)
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=choose_action)
    except Exception as e:
        logger.debug("ChatAction error: %s", e)

    # 1. Download sticker (or thumbnail for animated/video sticker) for vision & archiving
    raw_bytes: Optional[bytes] = None
    jpeg_bytes: Optional[bytes] = None
    filename = f"sticker_{incoming_uid}.webp"

    try:
        file_to_download = sticker.thumbnail if (sticker.is_animated or sticker.is_video) and sticker.thumbnail else sticker
        sticker_file = await file_to_download.get_file()
        byte_arr = bytearray()
        await sticker_file.download_as_bytearray(byte_arr)
        raw_bytes = bytes(byte_arr)

        if Image is not None and raw_bytes:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                rgb_img = img.convert("RGB")
                buf = io.BytesIO()
                rgb_img.save(buf, format="JPEG")
                jpeg_bytes = buf.getvalue()
    except Exception as e:
        logger.debug("Sticker download/conversion skipped or failed: %s", e)

    # 2. Security scan & audit logging
    if raw_bytes:
        scan_res = await security_scanner.scan_file_bytes(raw_bytes, filename=filename, expected_mime="image/webp")
        if not scan_res.is_safe:
            data_collector.record_user(user, is_owner=owner)
            data_collector.log_interaction(
                user,
                message_type="file_blocked",
                content=f"Blocked sticker: {filename} ({scan_res.reason})",
                is_owner=owner,
                metadata=scan_res.details,
            )
            await message.reply_text("⚠️ Security Alert: File flagged as unsafe and discarded.")
            return

        saved_media_path = data_collector.save_media(user_id or 0, raw_bytes, filename)
    else:
        saved_media_path = None

    data_collector.record_user(user, is_owner=owner)
    data_collector.log_interaction(
        user,
        message_type="sticker",
        content=f"Sticker: {emoji} (pack: {pack_name})",
        media_path=saved_media_path,
        is_owner=owner,
        metadata={"emoji": emoji, "set_name": pack_name, "is_animated": getattr(sticker, "is_animated", False), "is_video": getattr(sticker, "is_video", False)},
    )

    # 3. Vision Extraction (local Ollama vision / OCR)
    visual_context = ""
    if jpeg_bytes:
        try:
            visual_context = await media_processor.extract_image(
                jpeg_bytes,
                filename=f"sticker_{incoming_uid}.jpg",
                prompt="Briefly describe this sticker graphic, character expression, mood, and any text visible.",
            )
        except Exception as e:
            logger.debug("Sticker visual extraction error: %s", e)

    # 4. Determine input emotion & target response emotions
    input_emotion = detect_emotion(emoji, visual_context)
    target_emotions = get_target_emotions(input_emotion)
    logger.info("Sticker received (emoji='%s', pack='%s', emotion='%s', target_emotions=%s)", emoji, pack_name, input_emotion, target_emotions)

    # Cache incoming sticker under its emotion
    sticker_cache.record_sticker(sticker.file_id, input_emotion)

    # 5. Find the best matching sticker to reply with
    reply_sticker_id: Optional[str] = None
    matched_sticker_obj: Any = None

    # Step A: Attempt to fetch same sticker pack
    if pack_name:
        try:
            sticker_set = await context.bot.get_sticker_set(pack_name)
            if sticker_set and sticker_set.stickers:
                sticker_cache.record_sticker_set(sticker_set)
                matched_sticker = pick_matching_sticker(
                    sticker_set.stickers,
                    target_emotions,
                    incoming_file_unique_id=incoming_uid,
                )
                if matched_sticker and getattr(matched_sticker, "file_id", None):
                    reply_sticker_id = matched_sticker.file_id
                    matched_sticker_obj = matched_sticker
        except Exception as e:
            logger.debug("Could not fetch sticker set '%s': %s", pack_name, e)

    # Step B: Fallback to default sticker set if configured
    if not reply_sticker_id and TELEGRAM_DEFAULT_STICKER_SET:
        try:
            fallback_set = await context.bot.get_sticker_set(TELEGRAM_DEFAULT_STICKER_SET)
            if fallback_set and fallback_set.stickers:
                matched_sticker = pick_matching_sticker(
                    fallback_set.stickers,
                    target_emotions,
                )
                if matched_sticker and getattr(matched_sticker, "file_id", None):
                    reply_sticker_id = matched_sticker.file_id
                    matched_sticker_obj = matched_sticker
        except Exception as e:
            logger.debug("Could not fetch default sticker set '%s': %s", TELEGRAM_DEFAULT_STICKER_SET, e)

    # Step C: Fallback to cached stickers matching target emotions
    if not reply_sticker_id:
        reply_sticker_id = sticker_cache.get_sticker_for_emotions(target_emotions)

    # Step D: Ultimate fallback - mirror back incoming sticker
    if not reply_sticker_id:
        reply_sticker_id = sticker.file_id
        matched_sticker_obj = sticker

    # 6. Update media context in Nino agent memory
    agent.set_media_context(
        user_id or 0,
        "sticker",
        filename,
        f"Sticker with emoji '{emoji}' from pack '{pack_name}'. Detected mood: {input_emotion}. Visual: {visual_context}",
    )

    # 7. Reply with the accurate sticker!
    sent_sticker_msg = None
    try:
        sent_sticker_msg = await message.reply_sticker(sticker=reply_sticker_id)
    except Exception as e:
        logger.error("Failed to send sticker reply: %s", e)
        # Fallback to text if sticker reply failed
        fallback_msg = f"Received sticker {emoji}!"
        data_collector.log_reply(user=user, content=fallback_msg, reply_to_type="sticker", is_owner=owner)
        await message.reply_text(fallback_msg, parse_mode=ParseMode.MARKDOWN)
        return

    # 8. Archive the replied sticker image file into bot_saved_data/media/{user_id}/ so it can be seen in saved data
    reply_media_path: Optional[str] = None
    target_sticker = getattr(sent_sticker_msg, "sticker", None) or matched_sticker_obj
    try:
        file_to_dl = None
        if target_sticker:
            file_to_dl = (
                target_sticker.thumbnail
                if (getattr(target_sticker, "is_animated", False) or getattr(target_sticker, "is_video", False))
                and getattr(target_sticker, "thumbnail", None)
                else target_sticker
            )
        dl_file = None
        if file_to_dl and hasattr(file_to_dl, "get_file"):
            res = file_to_dl.get_file()
            dl_file = await res if inspect.isawaitable(res) else res
        elif hasattr(context, "bot") and hasattr(context.bot, "get_file"):
            res = context.bot.get_file(reply_sticker_id)
            dl_file = await res if inspect.isawaitable(res) else res

        if dl_file and hasattr(dl_file, "download_as_bytearray"):
            rep_byte_arr = bytearray()
            res = dl_file.download_as_bytearray(rep_byte_arr)
            if inspect.isawaitable(res):
                await res
            rep_bytes = bytes(rep_byte_arr)
            if rep_bytes:
                rep_uid = getattr(target_sticker, "file_unique_id", None) or (reply_sticker_id[-12:] if reply_sticker_id else "rep")
                rep_filename = f"reply_sticker_{rep_uid}.webp"
                reply_media_path = data_collector.save_media(user_id or 0, rep_bytes, rep_filename)
    except Exception as dl_err:
        logger.debug("Failed to download/archive replied sticker: %s", dl_err)

    reply_emoji = getattr(target_sticker, "emoji", "") or ""
    reply_pack = getattr(target_sticker, "set_name", "") or pack_name or ""
    reply_uid = getattr(target_sticker, "file_unique_id", "") or ""
    reply_content = f"Sticker: {reply_emoji} (pack: {reply_pack})" if (reply_emoji or reply_pack) else f"Sent sticker: {reply_sticker_id}"

    # 9. Optional: If user added a caption/question or if text commentary is configured
    caption = message.caption
    if caption or TELEGRAM_STICKER_REPLY_TEXT:
        prompt = (
            f"[Context: User sent a Telegram sticker ({emoji} from pack '{pack_name}'). "
            f"Detected emotion: {input_emotion}. Visual description: {visual_context}]\n"
        )
        if caption:
            prompt += f"User Caption/Question: {caption}\nAnswer the user directly in your witty persona."
        else:
            prompt += f"Give a single witty, short 1-line comment reacting to this sticker in your persona."

        try:
            reply_text, _ = await agent.chat(
                prompt,
                image_bytes=jpeg_bytes,
                user_id=user_id,
                is_owner=owner,
            )
            if reply_text and reply_text.strip():
                reply_content += f" | {reply_text}"
                await message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.debug("Sticker text follow-up error: %s", e)

    # Log to bot_saved_data/reply.jsonl with media_path
    data_collector.log_reply(
        user=user,
        content=reply_content,
        reply_to_type="sticker",
        media_path=reply_media_path,
        is_owner=owner,
        metadata={
            "sticker_id": reply_sticker_id,
            "sticker_unique_id": reply_uid,
            "emoji": reply_emoji,
            "set_name": reply_pack,
            "input_emoji": emoji,
            "input_emotion": input_emotion,
        },
    )

    # Log to bot_saved_data/interactions.jsonl so outgoing sticker replies are visible in saved interactions
    data_collector.log_interaction(
        user,
        message_type="sticker_reply",
        content=f"Bot replied with sticker: {reply_emoji} (pack: {reply_pack})",
        media_path=reply_media_path,
        is_owner=owner,
        metadata={
            "direction": "outgoing",
            "reply_to": "sticker",
            "sticker_id": reply_sticker_id,
            "sticker_unique_id": reply_uid,
            "emoji": reply_emoji,
            "set_name": reply_pack,
            "input_emoji": emoji,
            "input_emotion": input_emotion,
        },
    )


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming video or video notes: security scan, audio transcription, keyframes, and session state."""
    message = update.message
    if not message or (not message.video and not message.video_note):
        return

    user = update.effective_user
    user_id = user.id if user else None
    if not await check_user_access_or_prompt(update, context):
        return

    owner = is_owner(user_id)
    chat_id = update.effective_chat.id
    vid_obj = message.video or message.video_note
    caption = message.caption or "Please analyze and explain what is happening in this video."

    if getattr(vid_obj, "file_size", 0) > 32_000_000:
        await message.reply_text("⚠️ Video exceeds the 32MB safety processing limit.")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
        tmp_vid_path = Path(tmp_vid.name)

    try:
        tg_file = await vid_obj.get_file()
        await tg_file.download_to_drive(custom_path=str(tmp_vid_path))

        filename = f"video_{vid_obj.file_unique_id}.mp4"

        # 1. Automated malware & reputation scan on disk
        scan_res = await security_scanner.scan_file_on_disk(tmp_vid_path, filename=filename, expected_mime="video/mp4")
        if not scan_res.is_safe:
            data_collector.record_user(user, is_owner=owner)
            data_collector.log_interaction(
                user,
                message_type="file_blocked",
                content=f"Blocked video: {filename} ({scan_res.reason})",
                is_owner=owner,
                metadata=scan_res.details,
            )
            await message.reply_text("⚠️ Security Alert: File flagged as unsafe and discarded.")
            return

        vid_bytes = tmp_vid_path.read_bytes()
        # Archive
        saved_media_path = data_collector.save_media(
            user_id or 0,
            vid_bytes,
            filename,
        )
        data_collector.record_user(user, is_owner=owner)
        data_collector.log_interaction(
            user,
            message_type="video",
            content=caption,
            media_path=saved_media_path,
            is_owner=owner,
            metadata={"duration": getattr(vid_obj, "duration", None)},
        )

        # 2. Extract content (audio transcript via whisper + keyframe vision)
        extracted_content = await media_processor.extract_video(tmp_vid_path, filename=filename)

        # 3. Store active media session memory
        agent.set_media_context(user_id or 0, "video", filename, extracted_content)

        prompt = (
            f"[Context from recently uploaded video ({filename})]:\n"
            f"{extracted_content}\n\n"
            f"User Question: {caption}"
        )

        reply_text, engine = await agent.chat(
            prompt,
            user_id=user_id,
            is_owner=owner,
        )
        data_collector.log_reply(
            user=user,
            content=reply_text,
            reply_to_type="video",
            is_owner=owner,
            metadata={"engine": engine, "filename": filename},
        )
        await message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error("Video processing error: %s", e)
        err_msg = f"⚠️ Could not process video: {e}"
        data_collector.log_reply(user=user, content=err_msg, reply_to_type="video", is_owner=owner)
        await message.reply_text(err_msg)
    finally:
        # Wrap all local file operations in try...finally blocks to guarantee immediate deletion
        if tmp_vid_path.exists():
            try:
                tmp_vid_path.unlink(missing_ok=True)
            except Exception:
                pass


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process and archive uploaded documents (PDF, Word, Code, Data, Images, Videos) with multi-stage security."""
    message = update.message
    if not message or not message.document:
        return

    user = update.effective_user
    user_id = user.id if user else None
    if not await check_user_access_or_prompt(update, context):
        return

    owner = is_owner(user_id)
    doc = message.document
    chat_id = update.effective_chat.id
    caption = message.caption or "Please analyze and summarize this file."

    # Check file size (limit to 32MB for safety and VT support)
    if doc.file_size and doc.file_size > 32_000_000:
        await message.reply_text("⚠️ Document exceeds the 32MB safety processing limit.")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Wrap file operations in try...finally to guarantee immediate deletion of temporary media files
    tmp_path: Optional[Path] = None
    try:
        filename = security_scanner.sanitize_filename(doc.file_name or "document.bin")
        file_obj = await doc.get_file()
        byte_arr = bytearray()
        await file_obj.download_as_bytearray(byte_arr)
        doc_bytes = bytes(byte_arr)

        # 1. Multi-stage Security & Virus Scanning
        scan_res = await security_scanner.scan_file_bytes(doc_bytes, filename=filename, expected_mime=doc.mime_type)
        if not scan_res.is_safe:
            data_collector.record_user(user, is_owner=owner)
            data_collector.log_interaction(
                user,
                message_type="file_blocked",
                content=f"Blocked document: {filename} ({scan_res.reason})",
                is_owner=owner,
                metadata=scan_res.details,
            )
            await message.reply_text("⚠️ Security Alert: File flagged as unsafe and discarded.")
            return

        # Archive verified safe file into bot_saved_data/media/{user_id}/
        saved_media_path = data_collector.save_media(user_id or 0, doc_bytes, filename)
        data_collector.record_user(user, is_owner=owner)

        user_caption = (message.caption or "").strip()
        ext = Path(filename).suffix.lower()

        # 2. Multimodal Routing & Content Extraction
        image_payload: Optional[bytes] = None
        if ext == ".pdf" or (doc.mime_type and "pdf" in doc.mime_type):
            extracted_content = media_processor.extract_pdf(doc_bytes, filename=filename)
            media_type = "pdf"

        elif ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"} or (doc.mime_type and "image" in doc.mime_type):
            img_prompt = user_caption or "Analyze this image and transcribe all readable text, data, and visual details."
            extracted_content = await media_processor.extract_image(doc_bytes, filename=filename, prompt=img_prompt)
            media_type = "image"
            image_payload = doc_bytes

        elif ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"} or (doc.mime_type and "video" in doc.mime_type):
            extracted_content = await media_processor.extract_video(doc_bytes, filename=filename)
            media_type = "video"

        else:
            # Inspection for text, code, docx, etc.
            inspection = inspect_and_read_file(doc_bytes, filename=filename, mime_type=doc.mime_type)
            extracted_content = inspection.text_content or doc_bytes.decode("utf-8", errors="ignore")
            media_type = "document"

        # 3. Store active media session memory for ongoing conversational follow-up questions
        agent.set_media_context(user_id or 0, media_type, filename, extracted_content)

        # 4. Formulate contextual prompt
        if user_caption:
            query_prompt = (
                f"The user uploaded a {media_type} document ('{filename}') with the question:\n\"{user_caption}\"\n\n"
                f"Read the document content carefully and answer the user's question accurately, directly, and thoroughly in your persona."
            )
        else:
            query_prompt = (
                f"The user shared a {media_type} document ('{filename}') without a specific question.\n"
                f"Provide a structured, helpful review of the document:\n"
                f"• **Executive Summary**: High-level overview of what this document is about.\n"
                f"• **Key Data Points & Findings**: Highlight the main facts, numbers, sections, or dates.\n"
                f"• **Next Steps**: Invite the user to ask any questions or request deeper analysis of specific parts."
            )

        data_collector.log_interaction(
            user,
            message_type="document",
            content=f"File: {filename} ({media_type}) - {user_caption or 'Document upload'}",
            media_path=saved_media_path,
            is_owner=owner,
            metadata={"filename": filename, "file_size": doc.file_size, "type": media_type},
        )

        raw_display = f"[Uploaded {media_type.upper()} '{filename}'] {user_caption}" if user_caption else f"[Uploaded {media_type.upper()} '{filename}']"
        reply_text, engine = await agent.chat(
            query_prompt,
            image_bytes=image_payload,
            user_id=user_id,
            is_owner=owner,
            raw_prompt=raw_display,
        )
        data_collector.log_reply(
            user=user,
            content=reply_text,
            reply_to_type="document",
            is_owner=owner,
            metadata={"engine": engine, "filename": filename},
        )
        await message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error("Document processing error for %s: %s", getattr(doc, "file_name", "unknown"), e)
        err_msg = f"⚠️ Could not process document: {e}"
        data_collector.log_reply(user=user, content=err_msg, reply_to_type="document", is_owner=owner)
        await message.reply_text(err_msg)
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Respond to native Telegram GPS location pins."""
    message = update.message
    if not message or not message.location:
        return

    user = update.effective_user
    owner = is_owner(user.id if user else None)
    lat = message.location.latitude
    lon = message.location.longitude
    chat_id = update.effective_chat.id

    data_collector.record_user(user, is_owner=owner)
    data_collector.log_interaction(
        user,
        message_type="location",
        content=f"Coordinates: lat={lat}, lon={lon}",
        is_owner=owner,
        metadata={"lat": lat, "lon": lon},
    )

    # Synchronize inbuilt clock to GPS coordinates
    clock.sync_from_gps(lat, lon, "Your Shared Location")

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    report = await fetch_weather_report(round(lat, 4), round(lon, 4), "Your Shared Location")
    time_str = clock.now().strftime("%I:%M:%S %p")
    loc_msg = (
        f"📍 *Location Received & Clock Synchronized*\n\n"
        f"{report}\n\n"
        f"🕒 *Inbuilt Clock*: `{time_str}` ({clock.timezone_name})\n"
        f"📅 *Date*: {clock.now().strftime('%A, %B %d, %Y')}"
    )
    data_collector.log_reply(
        user=user,
        content=loc_msg,
        reply_to_type="location",
        is_owner=owner,
    )
    await message.reply_text(
        loc_msg,
        parse_mode=ParseMode.MARKDOWN,
    )


async def reminder_delivery_worker(bot) -> None:
    """Continuously poll SQLite for due reminders and deliver alerts via Telegram."""
    logger.info("Started Nino Telegram Reminder Delivery Worker.")
    while True:
        try:
            due_reminders = reminder_mgr.get_due_reminders()
            for r in due_reminders:
                rem_id = r["id"]
                chat_id = r["chat_id"] or AUTHORIZED_USER_ID
                if not chat_id:
                    reminder_mgr.mark_delivered(rem_id)
                    continue

                try:
                    dt = datetime.fromisoformat(r["trigger_time"])
                    time_display = dt.strftime("%I:%M:%S %p (%A, %d %b %Y)")
                except Exception:
                    time_display = r["trigger_time"]

                alert_text = (
                    "⏰ *NINO REMINDER ALERT*\n\n"
                    f"🔔 **{r['message']}**\n\n"
                    f"🕒 *Scheduled Time*: `{time_display}`\n"
                    "📍 *Status*: Due Now!"
                )
                keyboard = [
                    [
                        InlineKeyboardButton("⏰ Snooze 10m", callback_data=f"snooze_rem:{rem_id}:10"),
                        InlineKeyboardButton("⏰ Snooze 30m", callback_data=f"snooze_rem:{rem_id}:30"),
                    ],
                    [
                        InlineKeyboardButton("✅ Dismiss", callback_data=f"dismiss_rem:{rem_id}"),
                    ],
                ]
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=alert_text,
                        reply_markup=InlineKeyboardMarkup(keyboard),
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    reminder_mgr.mark_delivered(rem_id)
                    logger.info("Delivered reminder #%d to chat=%d", rem_id, chat_id)
                except Exception as send_err:
                    logger.error("Failed to send reminder #%d to chat %d: %s", rem_id, chat_id, send_err)
        except Exception as e:
            logger.error("Error in reminder delivery loop: %s", e)

        await asyncio.sleep(2.0)  # Check every 2 seconds


# ==============================================================================
# Post-Initialization Setup (Register Commands Menu with Telegram)
# ==============================================================================
# ==============================================================================
# Command Scopes (User vs Owner Menus)
# ==============================================================================
# Regular user menu: Only commands usable by all authorized users
USER_COMMANDS = [
    BotCommand("start", "Launch main menu"),
    BotCommand("help", "View command guide"),
    BotCommand("time", "Inbuilt location clock & time sync"),
    BotCommand("weather", "Live weather report"),
    BotCommand("voice", "Toggle voice replies on/off"),
    BotCommand("remind", "Schedule a reminder (e.g. /remind 10m call mom)"),
    BotCommand("reminders", "View and manage active reminders"),
    BotCommand("clear", "Reset current chat session"),
    BotCommand("clearmedia", "Clear active media session memory"),
    BotCommand("contact", "Share your contact/phone number"),
]

# Owner menu: Complete administrative tools + personal controls
OWNER_COMMANDS = [
    BotCommand("start", "Launch owner dashboard"),
    BotCommand("help", "View complete owner guide"),
    BotCommand("users", "View & manage registered users"),
    BotCommand("user", "Inspect user or view user directory"),
    BotCommand("pending", "View pending access requests"),
    BotCommand("approve", "Approve access for a user"),
    BotCommand("deny", "Deny or revoke access for a user"),
    BotCommand("requestcontact", "Ask user to share contact & name"),
    BotCommand("whois", "Inspect user Telegram profile & info"),
    BotCommand("send", "Send direct message to a user"),
    BotCommand("broadcast", "Broadcast announcement to all users"),
    BotCommand("status", "Check system & engine health"),
    BotCommand("mode", "Switch persona mode"),
    BotCommand("remember", "Save memory to Nino's brain"),
    BotCommand("memories", "List accessible brain memories"),
    BotCommand("vault", "Access private vault files"),
    BotCommand("time", "Inbuilt location clock & time sync"),
    BotCommand("weather", "Live weather report"),
    BotCommand("voice", "Toggle voice replies on/off"),
    BotCommand("remind", "Schedule a reminder"),
    BotCommand("reminders", "View and manage active reminders"),
    BotCommand("clear", "Reset current chat session"),
    BotCommand("clearmedia", "Clear active media session memory"),
]


async def register_scoped_commands(bot_instance) -> None:
    """Register scoped menus in Telegram UI:
    - Default scope (all users): Only shows general user commands.
    - Chat scope (owner's private chat): Shows owner administrative controls.
    """
    try:
        await bot_instance.set_my_commands(USER_COMMANDS, scope=BotCommandScopeDefault())
        logger.info("Successfully registered %d default user commands with Telegram.", len(USER_COMMANDS))
    except Exception as e:
        logger.warning("Could not register default user commands menu: %s", e)

    owner_id = get_owner_id()
    if owner_id:
        try:
            await bot_instance.set_my_commands(OWNER_COMMANDS, scope=BotCommandScopeChat(chat_id=owner_id))
            logger.info("Successfully registered %d owner-only commands for chat_id=%s with Telegram.", len(OWNER_COMMANDS), owner_id)
        except Exception as e:
            logger.warning("Could not register owner commands menu for chat %s: %s", owner_id, e)


async def post_init_setup(application: Application) -> None:
    """Register scoped command menus in Telegram UI and launch background workers."""
    await register_scoped_commands(application.bot)

    # Start the continuous background reminder worker
    asyncio.create_task(reminder_delivery_worker(application.bot))

    # Start the lightweight IP Checkin server
    checkin_server.bot_app = application
    asyncio.create_task(checkin_server.start())


# ==============================================================================
# Application Startup & Polling
# ==============================================================================
def main() -> None:
    """Build and start the Telegram bot application."""
    if not TELEGRAM_BOT_TOKEN:
        print("\n❌ Error: TELEGRAM_BOT_TOKEN is not set.")
        print("Please configure TELEGRAM_BOT_TOKEN in .env\n")
        return

    if AUTHORIZED_USER_ID is None:
        print("\n⚠️ Warning: MY_TELEGRAM_USER_ID is not configured or invalid.")
        print("Running in open guest mode with administrative features locked.\n")

    print("=" * 66)
    print("  Nino AI Assistant • Multi-User Telegram Bot Interface")
    print(f"  Owner User ID: {AUTHORIZED_USER_ID}")
    print(f"  Inbuilt Clock: {clock.city} ({clock.timezone_name})")
    print(f"  Audit Directory: {Path(DATA_DIR).resolve()}")
    print(f"  Primary Engine: Ollama ({OLLAMA_MODEL}) [100% Local]")
    print(f"  Vision Engine: Ollama ({OLLAMA_VISION_MODEL})")
    print(f"  Antigravity Policies: {', '.join(DENIAL_POLICIES)}")
    print(f"  Multimodal: Vision, Videos, Documents (PDF/DOCX/Code), Safe Links")
    print(f"  STT Model: faster-whisper ({WHISPER_MODEL}) | TTS Voice: {TTS_VOICE}")
    print("  Running with Long Polling (zero inbound ports / webhooks needed)")
    print("=" * 66)

    # Configure resilient HTTP transport with 30s timeouts
    request = HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )

    # Initialize Telegram Application with post_init
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(request)
        .post_init(post_init_setup)
        .build()
    )

    # 1. Register commands (available to all users; administrative commands guarded inside)
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler(["remind", "reminder"], remind_command))
    application.add_handler(CommandHandler(["reminders", "remindlist"], reminders_list_command))
    application.add_handler(CommandHandler(["time", "clock"], time_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("mode", mode_command))
    application.add_handler(CommandHandler("voice", voice_toggle_command))
    application.add_handler(CommandHandler("remember", remember_command))
    application.add_handler(CommandHandler("memories", memories_command))
    application.add_handler(CommandHandler("weather", weather_command))
    application.add_handler(CommandHandler("vault", vault_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler(["clearmedia", "clearm"], clearmedia_command))
    application.add_handler(CommandHandler("contact", contact_command))
    application.add_handler(CommandHandler("approve", approve_command))
    application.add_handler(CommandHandler(["deny", "reject", "revoke"], deny_command))
    application.add_handler(CommandHandler(["users", "user"], users_command))
    application.add_handler(CommandHandler("pending", pending_command))
    application.add_handler(CommandHandler(["requestcontact", "askcontact"], request_contact_command))
    application.add_handler(CommandHandler("whois", whois_command))
    application.add_handler(CommandHandler(["send", "msg", "dm", "tell"], send_message_command))
    application.add_handler(CommandHandler(["broadcast", "announce"], broadcast_command))

    # 2. Register media & text handlers for all users
    application.add_handler(MessageHandler(filters.CONTACT, handle_contact))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Sticker.ALL, handle_sticker))
    application.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, handle_video))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.LOCATION, handle_location))

    # 3. Register callback query for interactive dashboard & confirmations
    application.add_handler(CallbackQueryHandler(callback_query_handler))

    # Run long polling
    logger.info("Nino Telegram Bot is starting polling...")
    application.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
        bootstrap_retries=5,
    )


if __name__ == "__main__":
    main()

