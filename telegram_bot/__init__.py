"""Nino Telegram Bot package."""

import sys
import clock

# Forward telegram_bot.clock directly to root clock package
sys.modules["telegram_bot.clock"] = clock
