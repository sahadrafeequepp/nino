"""Root entry point and Telegram handlers for Nino AI Assistant.

Exports handlers for:
- Document (PDF, images, videos, text with automated malware scan and extraction)
- Photo (images with vision OCR and media session buffer)
- Video (audio transcription via faster-whisper, keyframe vision)
- Text (URL reputation checks, web scraping, and follow-up media session QA)
- Commands: /clearmedia, /status, /clear, /start, /help, etc.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.resolve()
bot_dir = root_dir / "telegram_bot"

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

from telegram_bot.bot import (
    approve_command,
    broadcast_command,
    clear_command,
    clearmedia_command,
    deny_command,
    handle_document,
    handle_photo,
    handle_text,
    handle_video,
    main,
    pending_command,
    request_contact_command,
    send_message_command,
    status_command,
    users_command,
    whois_command,
)

__all__ = [
    "main",
    "handle_document",
    "handle_photo",
    "handle_video",
    "handle_text",
    "clear_command",
    "clearmedia_command",
    "status_command",
    "approve_command",
    "deny_command",
    "users_command",
    "pending_command",
    "request_contact_command",
    "whois_command",
    "send_message_command",
    "broadcast_command",
]

if __name__ == "__main__":
    main()
