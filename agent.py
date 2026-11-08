"""Root entry point for Nino Antigravity Agent with Local Ollama LLM and Media Session Memory."""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.resolve()
bot_dir = root_dir / "telegram_bot"

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

from telegram_bot.agent import (
    DENIAL_POLICIES,
    NINO_SYSTEM_PROMPT,
    MediaSessionManager,
    NinoAgent,
    VaultTool,
    ask_nino,
    chat_history,
    clear_chat_history,
    clear_media_context,
    get_default_agent,
    history,
)

__all__ = [
    "NinoAgent",
    "MediaSessionManager",
    "VaultTool",
    "DENIAL_POLICIES",
    "NINO_SYSTEM_PROMPT",
    "ask_nino",
    "clear_media_context",
    "clear_chat_history",
    "get_default_agent",
    "history",
    "chat_history",
]
