"""Mock STT adapter for testing without model downloads."""

from typing import AsyncIterator
from nino.voice.interfaces import STTAdapter


class MockSTTAdapter(STTAdapter):
    """Mock STT adapter returning scripted transcripts or echo."""

    def __init__(self, fixed_transcript: str = "Hello Nino"):
        self.fixed_transcript = fixed_transcript
        self.transcribe_call_count = 0
        self.last_audio_bytes = b""

    async def transcribe(self, audio_pcm: bytes, sample_rate: int = 16000) -> str:
        self.transcribe_call_count += 1
        self.last_audio_bytes = audio_pcm
        return self.fixed_transcript
