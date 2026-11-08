"""Mock TTS adapter for fast testing without audio model downloads."""

import math
from typing import AsyncIterator
from nino.voice.interfaces import TTSAdapter


class MockTTSAdapter(TTSAdapter):
    """Mock TTS adapter yielding synthetic PCM audio."""

    def __init__(self, sample_rate: int = 16000, duration_seconds: float = 0.2):
        self._sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.synthesize_call_count = 0
        self.last_synthesized_text = ""

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _generate_beep(self) -> bytes:
        """Generate a short 440Hz sine wave as 16-bit PCM."""
        num_samples = int(self._sample_rate * self.duration_seconds)
        audio_bytes = bytearray()
        for i in range(num_samples):
            val = int(10000 * math.sin(2 * math.pi * 440 * i / self._sample_rate))
            audio_bytes.extend(val.to_bytes(2, byteorder="little", signed=True))
        return bytes(audio_bytes)

    async def synthesize(self, text: str) -> bytes:
        self.synthesize_call_count += 1
        self.last_synthesized_text = text
        if not text.strip():
            return b""
        return self._generate_beep()

    async def synthesize_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        async for sentence in text_stream:
            audio = await self.synthesize(sentence)
            if audio:
                yield audio
