"""faster-whisper STT adapter implementation for on-device transcription."""

import asyncio
import io
import logging
from typing import Literal
import numpy as np

from nino.voice.interfaces import STTAdapter

logger = logging.getLogger("nino.voice.stt.whisper")


class FasterWhisperAdapter(STTAdapter):
    """Speech-to-text adapter powered by faster-whisper (CTranslate2)."""

    def __init__(
        self,
        model_size_or_path: str = "base.en",
        compute_type: str = "int8",
        device: str = "auto",
        language: str = "en",
    ):
        self.model_size_or_path = model_size_or_path
        self.compute_type = compute_type
        self.device = device
        self.language = language
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)",
                self.model_size_or_path,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size_or_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def _transcribe_sync(self, audio_pcm: bytes, sample_rate: int) -> str:
        if not audio_pcm:
            return ""

        # Convert raw 16-bit signed PCM to normalized float32 in [-1.0, 1.0]
        audio_array = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_array) == 0:
            return ""

        model = self._ensure_model()
        segments, info = model.transcribe(
            audio_array,
            language=self.language,
            beam_size=1,
            vad_filter=False,  # VAD already handled upstream by Silero
        )

        texts = [segment.text.strip() for segment in segments if segment.text.strip()]
        transcript = " ".join(texts).strip()
        logger.debug("Transcribed audio (%d samples) -> '%s'", len(audio_array), transcript)
        return transcript

    async def transcribe(self, audio_pcm: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw PCM in-memory without creating temporary files on disk."""
        if not audio_pcm:
            return ""
        return await asyncio.to_thread(self._transcribe_sync, audio_pcm, sample_rate)
