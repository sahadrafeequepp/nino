"""Interface definitions for Nino Voice Subsystem (STT, TTS, Wake Word, VAD)."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class STTAdapter(ABC):
    """Abstract interface for Speech-to-Text inference."""

    @abstractmethod
    async def transcribe(self, audio_pcm: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw 16-bit linear PCM audio into text.

        Args:
            audio_pcm: Raw PCM byte buffer (16-bit signed, mono).
            sample_rate: Audio sample rate in Hz (default: 16000).

        Returns:
            The transcribed text string.
        """
        pass


class TTSAdapter(ABC):
    """Abstract interface for Text-to-Speech synthesis."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text into raw 16-bit linear PCM audio bytes.

        Args:
            text: Text to speak.

        Returns:
            Raw PCM audio bytes (16-bit signed, mono, typically 16kHz or 22.05kHz).
        """
        pass

    @abstractmethod
    async def synthesize_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Stream sentence-level audio synthesis as text arrives.

        Args:
            text_stream: Asynchronous stream of text sentences/tokens.

        Yields:
            PCM audio chunk bytes.
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the output sample rate in Hz."""
        pass


class WakeWordDetector(ABC):
    """Abstract interface for streaming wake word detection."""

    @abstractmethod
    def process_frame(self, frame_pcm: bytes) -> tuple[bool, float, str | None]:
        """Process a single PCM frame and detect wake word activation.

        Args:
            frame_pcm: Raw 16-bit linear PCM audio frame (e.g. 1280 samples = 80ms at 16kHz).

        Returns:
            Tuple of (detected: bool, score: float, wake_word_name: str | None).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal detector state."""
        pass


class VADDetector(ABC):
    """Abstract interface for Voice Activity Detection."""

    @abstractmethod
    def is_speech(self, frame_pcm: bytes) -> bool:
        """Determine if the provided frame contains human speech.

        Args:
            frame_pcm: Raw 16-bit linear PCM audio frame (e.g. 512 samples = 32ms at 16kHz).

        Returns:
            True if speech is detected, False otherwise.
        """
        pass

    @abstractmethod
    def speech_probability(self, frame_pcm: bytes) -> float:
        """Return confidence score [0.0, 1.0] that the frame contains human speech."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal recurrent states."""
        pass
