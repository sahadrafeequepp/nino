"""Root entry point for Nino multi-modal ingestion and content extraction pipeline."""

import sys
from pathlib import Path

bot_dir = Path(__file__).parent / "telegram_bot"
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

from telegram_bot.media_processor import (
    DEFAULT_OLLAMA_VISION_MODELS,
    MediaProcessor,
    extract_text_from_photo,
    media_processor,
)

__all__ = [
    "DEFAULT_OLLAMA_VISION_MODELS",
    "MediaProcessor",
    "media_processor",
    "extract_text_from_photo",
]
