"""STT adapters package."""

from nino.voice.stt.whisper_adapter import FasterWhisperAdapter
from nino.voice.stt.mock_stt import MockSTTAdapter

__all__ = ["FasterWhisperAdapter", "MockSTTAdapter"]
