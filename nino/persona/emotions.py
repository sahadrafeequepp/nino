"""Emotion tag parsing and event dispatching for animated avatar reactions."""

from dataclasses import dataclass
import re
from typing import Optional

ALLOWED_EMOTIONS = {
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "relaxed",
    "thinking",
    "concerned",
}

# Regex to find complete emotion tags in text, handling optional inner and trailing whitespace
EMOTION_TAG_REGEX = re.compile(r"\[emotion:\s*([a-zA-Z0-9_-]+)\s*\]\s*", re.IGNORECASE)


@dataclass
class ParseResult:
    clean_text: str
    emotion: Optional[str]


class EmotionParser:
    """Parses streaming LLM token chunks, extracts emotion signals, and cleans text."""

    def __init__(self, default_emotion: str = "neutral"):
        self.default_emotion = default_emotion
        self.current_emotion = default_emotion
        self._partial_buffer = ""

    def reset(self) -> None:
        """Reset parser state for a new conversational turn."""
        self.current_emotion = self.default_emotion
        self._partial_buffer = ""

    def process_chunk(self, chunk: str) -> tuple[str, list[str]]:
        """Process incoming token chunk.

        Returns:
            (cleaned_text, detected_emotions)
        """
        combined = self._partial_buffer + chunk
        self._partial_buffer = ""

        # Check if there is an unclosed bracket at the end of the text
        last_open_bracket = combined.rfind("[")
        last_close_bracket = combined.rfind("]")

        if last_open_bracket != -1 and (last_close_bracket == -1 or last_open_bracket > last_close_bracket):
            potential_tag = combined[last_open_bracket:]
            tag_lower = potential_tag.lower()
            if potential_tag.startswith("[") and (
                "[emotion:".startswith(tag_lower) or tag_lower.startswith("[emotion:")
            ):
                self._partial_buffer = potential_tag
                combined = combined[:last_open_bracket]

        # Extract all complete emotion tags from the processed portion
        detected_emotions: list[str] = []

        def replace_tag(match: re.Match) -> str:
            tag_name = match.group(1).strip().lower()
            if tag_name in ALLOWED_EMOTIONS:
                self.current_emotion = tag_name
                detected_emotions.append(tag_name)
            return ""

        cleaned_text = EMOTION_TAG_REGEX.sub(replace_tag, combined)
        return cleaned_text, detected_emotions

    def flush(self) -> tuple[str, list[str]]:
        """Flush any remaining partial buffer when streaming ends."""
        if not self._partial_buffer:
            return "", []

        remaining = self._partial_buffer
        self._partial_buffer = ""
        detected_emotions: list[str] = []

        def replace_tag(match: re.Match) -> str:
            tag_name = match.group(1).strip().lower()
            if tag_name in ALLOWED_EMOTIONS:
                self.current_emotion = tag_name
                detected_emotions.append(tag_name)
            return ""

        cleaned_text = EMOTION_TAG_REGEX.sub(replace_tag, remaining)
        return cleaned_text, detected_emotions

    def parse_text(self, text: str) -> ParseResult:
        """Helper to parse a full static string in one call."""
        self.reset()
        clean, emotions = self.process_chunk(text)
        leftover, final_emotions = self.flush()
        all_clean = (clean + leftover).strip()
        all_emotions = emotions + final_emotions
        chosen_emotion = all_emotions[-1] if all_emotions else None
        return ParseResult(clean_text=all_clean, emotion=chosen_emotion)

    def feed_chunk(self, chunk: str) -> ParseResult:
        """Helper to feed a streaming chunk and return ParseResult."""
        clean, emotions = self.process_chunk(chunk)
        chosen_emotion = emotions[-1] if emotions else None
        return ParseResult(clean_text=clean, emotion=chosen_emotion)


def strip_emotion_tags(text: str) -> str:
    """Helper to remove all [emotion:...] tags from static text."""
    return EMOTION_TAG_REGEX.sub("", text).strip()
