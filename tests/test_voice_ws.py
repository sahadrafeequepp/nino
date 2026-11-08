"""Tests for /ws/voice WebSocket endpoint authentication and messaging."""

import json
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from nino.config import get_settings
from nino.main import app
from nino.models.base import LLMAdapter
from nino.voice.pipeline import VoicePipeline
from nino.voice.stt.mock_stt import MockSTTAdapter
from nino.voice.tts.mock_tts import MockTTSAdapter
from nino.voice.vad.silero_vad import SileroVAD
from nino.voice.wakeword.openwakeword_detector import OpenWakeWordDetector


class MockVoiceLLMAdapter(LLMAdapter):
    @property
    def model_name(self) -> str:
        return "mock"

    async def chat(self, messages, model=None, **kwargs):
        return "It is currently 3 PM."

    async def stream_chat(self, messages, model=None, **kwargs):
        yield "It is currently 3 PM."


@pytest.fixture
def voice_client():
    """Configure app with mock voice pipeline for fast deterministic testing."""
    settings = get_settings()
    with TestClient(app) as client:
        app.state.voice_pipeline = VoicePipeline(
            stt=MockSTTAdapter(fixed_transcript="What time is it?"),
            tts=MockTTSAdapter(),
            wake_word=OpenWakeWordDetector(),
            vad=SileroVAD(),
            conv_store=app.state.conv_store,
            adapter=MockVoiceLLMAdapter(),
            settings=settings,
            orchestrator=None,
            confirmation_manager=app.state.confirmation_manager,
        )
        yield client


def test_voice_ws_unauthorized_rejected(voice_client):
    """Connecting without valid Bearer token must close with policy violation."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with voice_client.websocket_connect("/ws/voice?token=wrong-token") as ws:
            pass

    assert exc_info.value.code == 1008  # WS_1008_POLICY_VIOLATION


def test_voice_ws_authorized_session_and_ptt_flow(voice_client):
    """Verify authorized connection, status updates, and push-to-talk speech turn."""
    settings = get_settings()

    with voice_client.websocket_connect(f"/ws/voice?token={settings.auth_token}") as ws:
        # Initial status frame should be idle
        init_msg = ws.receive_json()
        assert init_msg["type"] == "status"
        assert init_msg["state"] == "idle"

        # Send PTT start
        ws.send_text(json.dumps({"type": "ptt_start"}))
        status_listening = ws.receive_json()
        assert status_listening["type"] == "status"
        assert status_listening["state"] == "listening"

        # Send simulated 16kHz audio PCM frame
        fake_frame = b"\x00\x05\x00\x05" * 256
        ws.send_bytes(fake_frame)

        # Send PTT stop
        ws.send_text(json.dumps({"type": "ptt_stop"}))

        # Expect transcript, text deltas, or audio frames
        messages_received = []
        audio_received = 0

        # Receive responses from turn
        for _ in range(15):
            msg = ws.receive()
            if "text" in msg:
                data = json.loads(msg["text"])
                messages_received.append(data.get("type"))
                if data.get("type") == "status" and data.get("state") == "idle":
                    break
            elif "bytes" in msg:
                audio_received += len(msg["bytes"])

        assert "transcript" in messages_received or audio_received > 0
