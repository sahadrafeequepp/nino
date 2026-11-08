"""Tests for audio privacy perimeter and zero-disk-storage verification."""

import os
from pathlib import Path
import pytest

from nino.config import get_settings
from nino.memory.conversation import ConversationStore
from nino.memory.db import MemoryDB
from nino.models.base import LLMAdapter
from nino.security.privacy_gateway import MediaQuarantineException, PrivacyGateway, SecurityException
from nino.voice.pipeline import VoicePipeline
from nino.voice.stt.mock_stt import MockSTTAdapter
from nino.voice.tts.mock_tts import MockTTSAdapter
from nino.voice.vad.silero_vad import SileroVAD
from nino.voice.wakeword.openwakeword_detector import OpenWakeWordDetector


class MockAdapter(LLMAdapter):
    @property
    def model_name(self) -> str:
        return "mock"

    async def chat(self, messages, model=None, **kwargs):
        return "Mock response"

    async def stream_chat(self, messages, model=None, **kwargs):
        yield "Mock response sentence."


@pytest.mark.asyncio
async def test_privacy_gateway_refuses_raw_audio_bytes():
    """Privacy Gateway must refuse to forward raw audio bytes."""
    gw = PrivacyGateway(allow_external=True)

    fake_audio_pcm = b"\x00\x01\x00\x02" * 100
    with pytest.raises(MediaQuarantineException) as exc_info:
        await gw.dispatch_external(
            provider="openai",
            messages=[{"role": "user", "content": "hello"}],
            audio=fake_audio_pcm,
        )
    assert "audio" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_privacy_gateway_refuses_audio_data_uri():
    """Privacy Gateway must refuse to forward base64 audio data URIs."""
    gw = PrivacyGateway(allow_external=True)

    audio_uri = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    with pytest.raises(MediaQuarantineException) as exc_info:
        await gw.dispatch_external(
            provider="openai",
            messages=[{"role": "user", "content": f"Listen to this: {audio_uri}"}],
        )
    assert "audio data uri" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_privacy_gateway_refuses_audio_message_fields():
    """Privacy Gateway refuses message dicts containing audio fields."""
    gw = PrivacyGateway(allow_external=True)

    with pytest.raises(MediaQuarantineException) as exc_info:
        await gw.dispatch_external(
            provider="openai",
            messages=[{"role": "user", "content": "hello", "audio_data": b"123"}],
        )
    assert "audio" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_store_audio_false_writes_no_files_to_disk(tmp_path):
    """When STORE_AUDIO=false, zero audio files are created on disk."""
    settings = get_settings()
    assert settings.store_audio is False

    db = MemoryDB(str(tmp_path / "test.db"))
    conv_store = ConversationStore(db)

    pipeline = VoicePipeline(
        stt=MockSTTAdapter(fixed_transcript="What is the weather today?"),
        tts=MockTTSAdapter(),
        wake_word=OpenWakeWordDetector(),
        vad=SileroVAD(),
        conv_store=conv_store,
        adapter=MockAdapter(),
        settings=settings,
    )

    # Count initial files in tmp_path and workspace
    initial_workspace_files = list(Path(".").glob("**/*.wav")) + list(Path(".").glob("**/*.pcm"))

    # Simulate speech turn with in-memory PCM audio
    fake_speech_pcm = b"\x00\x10\x00\x20" * 4000
    pipeline.speech_buffer.extend(fake_speech_pcm)

    audio_chunks = []
    async for chunk in pipeline._process_speech_turn():
        audio_chunks.append(chunk)

    assert len(audio_chunks) > 0

    # Ensure no wav or pcm files were written to disk
    new_workspace_files = list(Path(".").glob("**/*.wav")) + list(Path(".").glob("**/*.pcm"))
    assert len(new_workspace_files) == len(initial_workspace_files)
    assert len(pipeline.speech_buffer) == 0  # Buffer discarded from memory
