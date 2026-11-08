"""Tests for voice latency tracking and instrumentation."""

import time
import pytest
from nino.voice.telemetry import VoiceLatencyTracker


def test_latency_tracker_checkpoints_and_math():
    tracker = VoiceLatencyTracker()

    # Simulate realistic timeline
    t0 = time.perf_counter()
    tracker.t_wake = t0
    tracker.t_speech_start = t0 + 0.1
    tracker.t_speech_end = t0 + 1.2
    tracker.t_stt_done = t0 + 1.6  # 400ms STT
    tracker.t_llm_first_token = t0 + 1.9  # 300ms TTFT
    tracker.t_first_sentence = t0 + 2.2  # 300ms sentence
    tracker.t_first_audio = t0 + 2.4  # 200ms TTS chunk
    tracker.t_playback_done = t0 + 4.0

    metrics = tracker.compute_metrics()

    assert "speech_end_to_transcript_ms" in metrics
    assert 390.0 <= metrics["speech_end_to_transcript_ms"] <= 410.0

    assert "transcript_to_first_token_ms" in metrics
    assert 290.0 <= metrics["transcript_to_first_token_ms"] <= 310.0

    assert "first_token_to_first_audio_ms" in metrics
    assert 490.0 <= metrics["first_token_to_first_audio_ms"] <= 510.0

    assert "total_speech_to_first_audio_ms" in metrics
    # From speech_end (1.2) to first_audio (2.4) = 1.2s = 1200ms
    assert 1190.0 <= metrics["total_speech_to_first_audio_ms"] <= 1210.0

    summary = tracker.summary_string()
    assert "[Latency]" in summary
    assert "Speech-to-Audio" in summary
