"""Privacy, PII, and secret filtering for Nino fine-tuning datasets."""

import re
from typing import Any

from nino.security.privacy_gateway import SECRET_PATTERNS, PII_RULES
from nino.memory.facts import SENSITIVE_CATEGORIES

SENSITIVE_KEYWORDS = [
    re.compile(r"\b(password|passcode|pin|secret|api[-_]?key|auth[-_]?token)\b", re.IGNORECASE),
    re.compile(r"\b(diagnos\w*|prescri\w*|medic\w*|doctor\w*|hospital\w*|symptom\w*|disease\w*|illness\w*)\b", re.IGNORECASE),
    re.compile(r"\b(bank\s*account|routing\s*number|credit\s*score|salary|net\s*worth|tax\s*return|income)\b", re.IGNORECASE),
    re.compile(r"\b(passport|driver'?s?\s*license|social\s*security)\b", re.IGNORECASE),
]


class SecretFilter:
    """Detects credentials, keys, or secrets that must never enter model weights."""

    @staticmethod
    def contains_secret(text: str) -> bool:
        """Return True if text contains any known secret or confidential key pattern."""
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                return True
        return False


class PIIScrubber:
    """Scrubs personally identifiable information (PII) using consistent placeholders."""

    @staticmethod
    def scrub(text: str) -> str:
        """Redact credit cards, SSNs, emails, phones, and IP addresses."""
        sanitized = text
        for pattern, placeholder in PII_RULES:
            sanitized = pattern.sub(placeholder, sanitized)
        return sanitized


class SensitiveMemoryFilter:
    """Drops data related to sensitive memory categories (health, finances, identity)."""

    @staticmethod
    def is_sensitive(text: str, category: str | None = None) -> bool:
        """Check if message content or category touches sensitive personal topics."""
        if category and category.lower().strip() in SENSITIVE_CATEGORIES:
            return True

        for pattern in SENSITIVE_KEYWORDS:
            if pattern.search(text):
                return True

        return False


class DataCleaner:
    """Coordinates filtering, sanitizing, and validation of candidate training examples."""

    def __init__(self, drop_sensitive: bool = True):
        self.drop_sensitive = drop_sensitive
        self.secret_filter = SecretFilter()
        self.pii_scrubber = PIIScrubber()
        self.sensitive_filter = SensitiveMemoryFilter()

    def process_turn(
        self,
        messages: list[dict[str, str]],
        category: str | None = None,
    ) -> tuple[bool, list[dict[str, str]], str]:
        """Validate and clean a conversational turn.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}, ...]
            category: Optional session or fact category

        Returns:
            (is_valid, cleaned_messages, drop_reason)
        """
        if not messages:
            return False, [], "Empty message list."

        cleaned: list[dict[str, str]] = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            # 1. Check for hard secrets
            if self.secret_filter.contains_secret(content):
                return False, [], "Contains confidential credential or secret key."

            # 2. Check for sensitive categories (health, finance, identity)
            if self.drop_sensitive and self.sensitive_filter.is_sensitive(content, category=category):
                return False, [], "Touches sensitive personal memory category (health, finances, identity)."

            # 3. Scrub PII
            scrubbed_content = self.pii_scrubber.scrub(content)
            cleaned.append({"role": role, "content": scrubbed_content})

        return True, cleaned, ""
