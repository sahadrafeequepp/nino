"""Latency measurement and telemetry for on-device voice interactions."""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("nino.voice.telemetry")


class VoiceLatencyTracker:
    """Tracks transition timestamps and latency metrics across the voice pipeline."""

    def __init__(self):
        self.t_wake: Optional[float] = None
        self.t_speech_start: Optional[float] = None
        self.t_speech_end: Optional[float] = None
        self.t_stt_done: Optional[float] = None
        self.t_llm_first_token: Optional[float] = None
        self.t_first_sentence: Optional[float] = None
        self.t_first_audio: Optional[float] = None
        self.t_playback_done: Optional[float] = None

    def mark_wake(self) -> float:
        self.t_wake = time.perf_counter()
        return self.t_wake

    def mark_speech_start(self) -> float:
        self.t_speech_start = time.perf_counter()
        return self.t_speech_start

    def mark_speech_end(self) -> float:
        self.t_speech_end = time.perf_counter()
        return self.t_speech_end

    def mark_stt_done(self) -> float:
        self.t_stt_done = time.perf_counter()
        return self.t_stt_done

    def mark_llm_first_token(self) -> float:
        self.t_llm_first_token = time.perf_counter()
        return self.t_llm_first_token

    def mark_first_sentence(self) -> float:
        self.t_first_sentence = time.perf_counter()
        return self.t_first_sentence

    def mark_first_audio(self) -> float:
        self.t_first_audio = time.perf_counter()
        return self.t_first_audio

    def mark_playback_done(self) -> float:
        self.t_playback_done = time.perf_counter()
        return self.t_playback_done

    def compute_metrics(self) -> dict[str, float]:
        """Compute duration metrics in milliseconds."""
        metrics: dict[str, float] = {}

        # 1. Wake to transcript
        if self.t_wake and self.t_stt_done:
            metrics["wake_to_transcript_ms"] = round((self.t_stt_done - self.t_wake) * 1000, 1)

        # 2. End of speech to transcript (STT processing duration)
        if self.t_speech_end and self.t_stt_done:
            metrics["speech_end_to_transcript_ms"] = round((self.t_stt_done - self.t_speech_end) * 1000, 1)

        # 3. Transcript to first token (LLM time-to-first-token TTFT)
        if self.t_stt_done and self.t_llm_first_token:
            metrics["transcript_to_first_token_ms"] = round((self.t_llm_first_token - self.t_stt_done) * 1000, 1)

        # 4. First token to first sentence
        if self.t_llm_first_token and self.t_first_sentence:
            metrics["first_token_to_sentence_ms"] = round((self.t_first_sentence - self.t_llm_first_token) * 1000, 1)

        # 5. First token to first audio chunk
        if self.t_llm_first_token and self.t_first_audio:
            metrics["first_token_to_first_audio_ms"] = round((self.t_first_audio - self.t_llm_first_token) * 1000, 1)

        # 6. Total end-to-end response latency (End of speech to first audio heard)
        ref_start = self.t_speech_end or self.t_wake
        if ref_start and self.t_first_audio:
            metrics["total_speech_to_first_audio_ms"] = round((self.t_first_audio - ref_start) * 1000, 1)

        return metrics

    def log_summary(self) -> None:
        """Log structured latency report."""
        m = self.compute_metrics()
        total = m.get("total_speech_to_first_audio_ms", 0.0)
        target = 2000.0  # 2.0s target
        status = "PASSED (< 2.0s)" if total <= target else "EXCEEDED TARGET (> 2.0s)"

        logger.info(
            "Voice Turn Latency: total=%.1fms [%s] | stt=%.1fms | ttft=%.1fms | tts_chunk=%.1fms",
            total,
            status,
            m.get("speech_end_to_transcript_ms", 0.0),
            m.get("transcript_to_first_token_ms", 0.0),
            m.get("first_token_to_first_audio_ms", 0.0),
        )

    def summary_string(self) -> str:
        """Human-readable string for CLI display."""
        m = self.compute_metrics()
        total_s = m.get("total_speech_to_first_audio_ms", 0.0) / 1000.0
        stt_s = m.get("speech_end_to_transcript_ms", 0.0) / 1000.0
        ttft_s = m.get("transcript_to_first_token_ms", 0.0) / 1000.0
        tts_s = m.get("first_token_to_first_audio_ms", 0.0) / 1000.0

        return (
            f"[Latency] Speech-to-Audio: {total_s:.2f}s "
            f"(STT: {stt_s:.2f}s | LLM TTFT: {ttft_s:.2f}s | First Audio: {tts_s:.2f}s)"
        )
