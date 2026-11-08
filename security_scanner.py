"""Root entry point for Nino multi-stage security and virus scanning pipeline."""

import sys
from pathlib import Path

bot_dir = Path(__file__).parent / "telegram_bot"
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

from telegram_bot.security_scanner import (
    DANGEROUS_EXTENSIONS,
    EXECUTABLE_MAGIC_BYTES,
    PDF_EXPLOIT_TOKENS,
    SecurityScanResult,
    SecurityScanner,
    security_scanner,
)

__all__ = [
    "DANGEROUS_EXTENSIONS",
    "EXECUTABLE_MAGIC_BYTES",
    "PDF_EXPLOIT_TOKENS",
    "SecurityScanResult",
    "SecurityScanner",
    "security_scanner",
]
