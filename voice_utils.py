"""Root entry point for Nino voice utilities (faster-whisper and edge-tts)."""

import sys
from pathlib import Path

bot_dir = Path(__file__).parent / "telegram_bot"
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

from telegram_bot.voice_utils import (
    sanitize_text_for_speech,
    synthesize_voice,
    transcribe_voice,
)


__all__ = ["transcribe_voice", "synthesize_voice", "sanitize_text_for_speech"]
