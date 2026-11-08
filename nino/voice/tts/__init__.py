"""TTS adapters package."""

from nino.voice.tts.piper_adapter import (
    PiperTTSAdapter,
    VOICE_CATALOG,
    resolve_voice_name,
    get_voice_info,
    is_voice_downloaded,
    insert_breather_pauses,
    prepare_headings_with_pauses,
    HEADLINE_PAUSE_TOKEN,
    CONJUNCTIONS_AND_CONNECTORS,
    NON_ENDING_WORDS,
)
from nino.voice.tts.mock_tts import MockTTSAdapter

__all__ = [
    "PiperTTSAdapter",
    "MockTTSAdapter",
    "VOICE_CATALOG",
    "resolve_voice_name",
    "get_voice_info",
    "is_voice_downloaded",
    "insert_breather_pauses",
    "prepare_headings_with_pauses",
    "HEADLINE_PAUSE_TOKEN",
    "CONJUNCTIONS_AND_CONNECTORS",
    "NON_ENDING_WORDS",
]
