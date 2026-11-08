"""Async helpers for voice processing: Speech-To-Text (faster-whisper) and Text-To-Speech (edge-tts)."""

import asyncio
import logging
import os
from pathlib import Path
import re
from typing import Any, Optional

logger = logging.getLogger("nino.telegram.voice")

# Global cache for faster-whisper models to prevent redundant reload latency
_whisper_cache: dict[str, Any] = {}


def get_whisper_model(model_size: str = "base.en"):
    """Retrieve or initialize a cached faster-whisper model."""
    if model_size not in _whisper_cache:
        try:
            from faster_whisper import WhisperModel

            logger.info("Loading faster-whisper model '%s' (compute_type=int8)...", model_size)
            # Use int8 quantization on CPU/auto for fast lightweight inference
            model = WhisperModel(model_size, device="auto", compute_type="int8")
            _whisper_cache[model_size] = model
        except Exception as e:
            logger.error("Failed to load faster-whisper model: %s", e)
            raise
    return _whisper_cache[model_size]


def _transcribe_sync(audio_path: str, model_size: str = "base.en") -> str:
    """Synchronous worker function to run faster-whisper transcription."""
    model = get_whisper_model(model_size)
    segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    text_chunks = [segment.text.strip() for segment in segments if segment.text]
    transcribed_text = " ".join(text_chunks).strip()
    logger.debug("Transcribed %s (lang=%s, prob=%.2f): %s", audio_path, info.language, info.language_probability, transcribed_text)
    return transcribed_text


async def transcribe_voice(audio_path: str | Path, model_size: str = "base.en") -> str:
    """Asynchronously transcribe an audio file (e.g. Telegram .ogg or .wav) via faster-whisper."""
    path_str = str(audio_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Audio file not found: {path_str}")

    return await asyncio.to_thread(_transcribe_sync, path_str, model_size)


# Comprehensive Unicode emoji pattern covering all modern emoji planes, pictographs, dingbats, and joiners
# NOTE: General punctuation (\u2000-\u206f) is explicitly excluded so contractions (curly apostrophes),
# dashes, and ellipses are preserved and normalized rather than stripped.
EMOJI_PATTERN = re.compile(
    "["
    "\U00010000-\U0010ffff"  # All Supplementary Planes (emoticons, pictographs, symbols, flags)
    "\u200d"                 # Zero-width joiner
    "\ufe0e\ufe0f"           # Variation selectors
    "\u20d0-\u20ff"          # Combining marks for symbols
    "\u2100-\u214f"          # Letterlike symbols
    "\u2190-\u21ff"          # Arrows
    "\u2200-\u22ff"          # Mathematical operators
    "\u2300-\u23ff"          # Misc technical symbols
    "\u2460-\u24ff"          # Enclosed alphanumerics
    "\u25a0-\u25ff"          # Geometric shapes
    "\u2600-\u27bf"          # Misc symbols & dingbats
    "\u2900-\u297f"          # Supplemental arrows
    "\u2b00-\u2bff"          # Misc symbols and arrows
    "\u3030\u303d"           # Wavy dash / part alternation
    "\u3297\u3299"           # Circled ideographs
    "]+",
    flags=re.UNICODE,
)

# Text emoticons and Discord/Slack shortcode emojis
SHORTCODE_EMOJI_PATTERN = re.compile(r":[a-zA-Z0-9_+-]+:")
TEXT_EMOTICON_PATTERN = re.compile(
    r"(?<!\w)(?::[-~]?[)D(P/\\|*oOpP3sS]|;[-~]?[)D(P/\\|*oOpP]|<3|</3|XD|xd|xD|-_-|\^_\^|T_T|o_O|O_o)(?!\w)"
)


def sanitize_text_for_speech(text: str) -> str:
    """Strip markdown code blocks, links, emojis, emoticons, stage directions, and formatting symbols for natural vocalization."""
    if not text:
        return ""

    # Remove markdown code fences ```code```
    cleaned = re.sub(r"```[\s\S]*?```", " [code snippet omitted] ", text)
    # Remove inline code `code`
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

    # Normalize unicode apostrophes, quotes, dashes, and ellipses
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("…", "...")
    cleaned = re.sub(r"\s*[—–]+\s*", ", ", cleaned)
    # Replace markdown links [label](url) with just label
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    # Strip markdown bold/strike markers while preserving the inner text
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"~~([^~]+)~~", r"\1", cleaned)

    ACTION_KEYWORDS = (
        "chuckle", "laugh", "smile", "grin", "sigh", "shrug", "nod",
        "wink", "smirk", "giggle", "snicker", "frown", "gasp", "eye",
        "eyes", "glint", "finger", "avatar", "click", "snap", "twirl",
        "look", "glance", "whisper", "lean", "head", "throat", "breath",
        "face", "hand", "walk", "clears", "pause", "hesitate",
    )

    # Remove stage action directions in asterisks (*chuckles*, *eyes widen*, etc.)
    # Emphasized words (e.g. *really*, *bold text*) are preserved without asterisks
    def _clean_asterisks(m):
        inner = m.group(1).strip()
        if any(p in inner.lower() for p in ACTION_KEYWORDS):
            return " "
        return f" {inner} "

    cleaned = re.sub(r"\*([^*]+)\*", _clean_asterisks, cleaned)

    # Remove emotion tags like [emotion:happy]
    cleaned = re.sub(r"\[emotion:[a-zA-Z_-]+\]", " ", cleaned)
    # Remove markdown headers and bullets
    cleaned = re.sub(r"^[#*>\-\s]+", "", cleaned, flags=re.MULTILINE)
    # Clean up any leftover asterisks/underscores
    cleaned = re.sub(r"[*_~]+", "", cleaned)

    # Remove raw URLs
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    # Remove all Unicode emojis and pictographs
    cleaned = EMOJI_PATTERN.sub("", cleaned)
    # Remove shortcode emojis like :rocket: or :fire:
    cleaned = SHORTCODE_EMOJI_PATTERN.sub("", cleaned)
    # Remove text emoticons like :) or :D
    cleaned = TEXT_EMOTICON_PATTERN.sub("", cleaned)
    return " ".join(cleaned.split()).strip()



async def synthesize_voice(
    text: str,
    output_path: str | Path,
    voice: str = "en-US-JennyNeural",
) -> str:
    """Synthesize speech audio from text using Microsoft edge-tts."""
    import edge_tts

    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    speech_text = sanitize_text_for_speech(text)
    if not speech_text:
        speech_text = "I received your message."

    communicate = edge_tts.Communicate(speech_text, voice=voice)
    await communicate.save(str(path_obj))
    logger.debug("Synthesized voice response saved to %s", path_obj)
    return str(path_obj)
