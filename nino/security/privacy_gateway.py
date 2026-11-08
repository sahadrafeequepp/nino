"""Privacy Gateway: Gatekeeper for external AI communication, secret detection, and PII redaction."""

import datetime
import logging
import re
from typing import Any

from nino.security.network_firewall import network_firewall

logger = logging.getLogger("nino.security.privacy_gateway")


class SecurityException(Exception):
    """Base exception for security and privacy violations."""
    pass


class BlockedSecretException(SecurityException):
    """Raised when text contains confidential credentials, keys, or secrets."""
    pass


class MediaQuarantineException(SecurityException):
    """Raised when audio, video, or raw media is attempted to be routed externally."""
    pass


# Pattern detecting audio data URIs and embedded base64 audio streams
AUDIO_DATA_URI_PATTERN = re.compile(
    r"data:audio/(?:wav|pcm|mp3|ogg|flac|webm|aac|m4a);base64,[A-Za-z0-9+/=]+",
    re.IGNORECASE,
)

# Pattern detecting image data URIs and embedded base64 image streams
IMAGE_DATA_URI_PATTERN = re.compile(
    r"data:image/(?:png|jpeg|jpg|webp|gif);base64,[A-Za-z0-9+/=]+",
    re.IGNORECASE,
)


# High-confidence secret patterns that cause immediate request blocking
SECRET_PATTERNS = [
    # AWS Access Key IDs
    re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    # Private Key PEM blocks
    re.compile(r"-----BEGIN\s+(?:[A-Z0-9_-]+\s+)?PRIVATE\s+KEY-----", re.IGNORECASE),
    # OpenAI API Keys
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    # Anthropic API Keys
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    # Google Gemini API Keys
    re.compile(r"\bAIzaSy[A-Za-z0-9_-]{33}\b"),
    # GitHub Personal Access Tokens
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b"),
    # Slack Tokens
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    # JWT Tokens
    re.compile(r"\beyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*\b"),
]

# PII redaction patterns replaced with placeholders
PII_RULES = [
    # Credit card numbers (13-16 digits with optional spaces or dashes)
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED_CARD]"),
    # US Social Security Numbers (SSN)
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    # Phone numbers (US and international formats)
    (re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
    # IPv4 addresses
    (re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"), "[REDACTED_IP]"),
    # Local user directory paths (e.g. C:\Users\Username or /home/username)
    (re.compile(r"(?:[a-zA-Z]:[\\/](?:Users|home)[\\/][a-zA-Z0-9_\-\.]+)", re.IGNORECASE), "[REDACTED_USER_PATH]"),
    # GPS coordinates (latitude and longitude pairs to prevent external location breach)
    (re.compile(r"\b(?:lat(?:itude)?[:\s=]+)?-?\d{1,3}\.\d{3,}(?:[,\s]+(?:lon(?:gitude)?[:\s=]+)?-?\d{1,3}\.\d{3,})\b", re.IGNORECASE), "[REDACTED_COORDINATES]"),
    (re.compile(r"\b\d{1,3}\.\d+°?\s*[NSEW],\s*\d{1,3}\.\d+°?\s*[NSEW]\b", re.IGNORECASE), "[REDACTED_COORDINATES]"),
]


# Conversational query prefix filler to prune when distilling search queries
CONVERSATIONAL_PREFIXES = [
    re.compile(r"^(?:can you\s+)?(?:please\s+)?(?:tell me\s+)?(?:search for\s+|look up\s+|find\s+)?", re.IGNORECASE),
]


class PrivacyGateway:
    """Privacy Gateway enforcing outbound data controls, DLP, and network perimeter protection."""

    def __init__(self, allow_external: bool = False, allow_external_images: bool = False):
        self.allow_external = allow_external
        self.allow_external_images = allow_external_images
        self.outbound_log: list[dict[str, Any]] = []

    def check_media_quarantine(self, content: Any) -> None:
        """Scan content or objects for audio/media to enforce Local-Only Media Quarantine."""
        if isinstance(content, (bytes, bytearray, memoryview)):
            raise MediaQuarantineException(
                "Local media quarantine: Binary audio/media data transmission to external providers is strictly prohibited."
            )
        if isinstance(content, str):
            if AUDIO_DATA_URI_PATTERN.search(content):
                raise MediaQuarantineException(
                    "Local media quarantine: Audio data URI detected. Transmission of audio to external providers is strictly prohibited."
                )
            if not self.allow_external_images and IMAGE_DATA_URI_PATTERN.search(content):
                raise MediaQuarantineException(
                    "Local media quarantine: Image data URI detected. Forwarding images to external providers is refused by policy."
                )

    def check_for_secrets(self, text: str) -> None:
        """Scan text for high-risk secrets and raise BlockedSecretException if found."""
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise BlockedSecretException(
                    "Outbound request blocked: sensitive credential or secret key detected in prompt."
                )

    def redact_pii(self, text: str) -> str:
        """Redact known PII entities (emails, phones, IPs, cards, SSN, user paths)."""
        sanitized = text
        for pattern, placeholder in PII_RULES:
            sanitized = pattern.sub(placeholder, sanitized)
        return sanitized

    def sanitize(self, text: str) -> str:
        """Sanitize text: check media quarantine, block if secrets are detected, otherwise redact PII."""
        self.check_media_quarantine(text)
        self.check_for_secrets(text)
        return self.redact_pii(text)

    @classmethod
    def sanitize_outbound(cls, text: str) -> str:
        """Class method for rapid sanitization of text intended for external egress (e.g. search queries)."""
        instance = cls(allow_external=True)
        return instance.sanitize(text)

    @classmethod
    def anonymize_search_query(cls, query: str) -> str:
        """Anonymize and distill a web search query by stripping personal pronouns, local paths, and PII."""
        # 1. First run full outbound sanitization (secrets check + PII redaction)
        sanitized = cls.sanitize_outbound(query.strip())
        # 2. Strip conversational search prefixes
        for pat in CONVERSATIONAL_PREFIXES:
            sanitized = pat.sub("", sanitized).strip()
        return sanitized

    def log_outbound_metadata(
        self,
        provider: str,
        message_count: int,
        char_count: int,
    ) -> dict[str, Any]:
        """Log outbound transmission metadata strictly without message content."""
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "target_provider": provider,
            "message_count": message_count,
            "char_count": char_count,
            "content_included": False,
        }
        self.outbound_log.append(entry)
        logger.info("External AI request logged: %s", entry)
        return entry

    async def dispatch_external(
        self,
        provider: str,
        messages: list[dict[str, Any]],
        target_endpoint_url: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Gate all external outbound requests.

        1. Ensures external providers are enabled by policy.
        2. Validates target endpoint against SSRF Network Firewall if URL is supplied.
        3. Enforces Local-Only Media Quarantine (refuses any audio/media/image).
        4. Sanitizes all message text (blocks secrets, redacts PII).
        5. Logs outbound transmission metadata (never content).
        6. Returns sanitized messages ready for provider transport.
        """
        # 1. Check endpoint URL against firewall if provided
        if target_endpoint_url:
            network_firewall.validate_url(target_endpoint_url)

        # Block if any audio or binary keyword kwargs are passed
        for k, v in kwargs.items():
            if "audio" in k.lower() or "media" in k.lower() or isinstance(v, (bytes, bytearray)):
                raise MediaQuarantineException(
                    f"Local media quarantine: Refusing external dispatch with parameter '{k}'."
                )
            if not self.allow_external_images and ("image" in k.lower() or "images" in k.lower()):
                raise MediaQuarantineException(
                    f"Local media quarantine: Refusing external dispatch with image parameter '{k}'."
                )

        if not self.allow_external:
            raise SecurityException(
                "External AI providers are disabled by policy. To enable, set ALLOW_EXTERNAL_PROVIDERS=true in environment."
            )

        sanitized_messages: list[dict[str, str]] = []
        total_chars = 0

        for msg in messages:
            # Check for any audio or binary fields in message dict
            for key, val in msg.items():
                if "audio" in key.lower() or "media" in key.lower():
                    raise MediaQuarantineException(
                        f"Local media quarantine: Message field '{key}' cannot be forwarded externally."
                    )
                if not self.allow_external_images and ("image" in key.lower() or "images" in key.lower()):
                    raise MediaQuarantineException(
                        f"Local media quarantine: Message field '{key}' containing image data cannot be forwarded externally."
                    )
                self.check_media_quarantine(val)

            role = msg.get("role", "user")
            content = msg.get("content", "")
            sanitized_content = self.sanitize(content)
            total_chars += len(sanitized_content)
            sanitized_messages.append({"role": role, "content": sanitized_content})

        self.log_outbound_metadata(
            provider=provider,
            message_count=len(messages),
            char_count=total_chars,
        )

        return sanitized_messages
