"""Tests for /ws/avatar WebSocket endpoint, REST endpoints, and avatar animation streaming."""

import json
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from nino.config import get_settings
from nino.main import app
from nino.models.base import LLMAdapter
from nino.persona.manager import PersonaManager


class MockAvatarLLMAdapter(LLMAdapter):
    @property
    def model_name(self) -> str:
        return "mock"

    async def chat(self, messages, model=None, **kwargs):
        return "[emotion:happy] Hello! How can I help you today?"

    async def stream_chat(self, messages, model=None, **kwargs):
        yield "[emotion:happy] Hello!"
        yield " How can I help you today?"


@pytest.fixture
def avatar_client():
    """Configure test client with mock LLM and fresh persona manager."""
    settings = get_settings()
    with TestClient(app) as client:
        app.state.adapter = MockAvatarLLMAdapter()
        app.state.persona_manager = PersonaManager()
        yield client


def test_avatar_rest_persona_endpoints(avatar_client):
    """Test GET /persona and POST /persona/mode."""
    res = avatar_client.get("/persona")
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "Nino"
    assert data["current_mode"] in ["companion", "work", "focus"]
    assert "work" in data["available_modes"]

    post_res = avatar_client.post("/persona/mode", json={"mode": "work"})
    assert post_res.status_code == 200
    post_data = post_res.json()
    assert post_data["status"] == "ok"
    assert post_data["mode"] == "work"


def test_avatar_ws_unauthorized_rejected(avatar_client):
    """Connecting without valid token must close with WS_1008_POLICY_VIOLATION."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with avatar_client.websocket_connect("/ws/avatar?token=bad-token") as ws:
            pass
    assert exc_info.value.code == 1008


def test_avatar_ws_authorized_session_and_set_mode(avatar_client):
    """Verify handshake and set_mode event."""
    settings = get_settings()

    with avatar_client.websocket_connect(f"/ws/avatar?token={settings.auth_token}") as ws:
        # Initial handshake frames
        msg1 = ws.receive_json()
        assert msg1["type"] == "state"
        assert msg1["state"] == "idle"

        msg2 = ws.receive_json()
        assert msg2["type"] == "mode"

        msg3 = ws.receive_json()
        assert msg3["type"] == "emotion"
        assert msg3["emotion"] == "neutral"

        # Send set_mode
        ws.send_text(json.dumps({"type": "set_mode", "mode": "focus"}))
        res1 = ws.receive_json()
        assert res1["type"] == "mode"
        assert res1["mode"] == "focus"

        res2 = ws.receive_json()
        assert res2["type"] == "mode_changed"
        assert res2["mode"] == "focus"


def test_avatar_ws_chat_streaming_and_emotion(avatar_client):
    """Verify chat turn returns state changes, emotion events, text deltas, and completion."""
    settings = get_settings()

    with avatar_client.websocket_connect(f"/ws/avatar?token={settings.auth_token}") as ws:
        # Consume 3 initial frames
        for _ in range(3):
            ws.receive_json()

        # Send chat message
        ws.send_text(json.dumps({"type": "user_text", "text": "Hello Nino"}))

        # Collect response frames
        received_msgs = []
        received_types = []
        received_deltas = []
        emotions_seen = []

        while True:
            msg = ws.receive_json()
            received_msgs.append(msg)
            mtype = msg.get("type")
            received_types.append(mtype)

            if mtype == "emotion":
                emotions_seen.append(msg.get("emotion"))
            elif mtype == "delta":
                received_deltas.append(msg.get("text"))
            elif mtype == "state" and msg.get("state") == "idle":
                break

        states_seen = [m.get("state") for m in received_msgs if m.get("type") == "state"]
        assert "thinking" in states_seen
        assert "speaking" in states_seen
        assert "happy" in emotions_seen
        full_text = "".join(received_deltas)
        assert "Hello!" in full_text
        assert "text_complete" in received_types


def test_avatar_ws_crisis_handling(avatar_client):
    """Verify crisis input immediately triggers supportive response without LLM generation."""
    settings = get_settings()

    with avatar_client.websocket_connect(f"/ws/avatar?token={settings.auth_token}") as ws:
        for _ in range(3):
            ws.receive_json()

        ws.send_text(json.dumps({"type": "user_text", "text": "I feel like killing myself"}))

        received = []
        while True:
            msg = ws.receive_json()
            received.append(msg)
            if msg.get("type") == "state" and msg.get("state") == "idle":
                break

        types = [m.get("type") for m in received]
        assert "emotion" in types
        assert any(m.get("emotion") == "concerned" for m in received if m.get("type") == "emotion")
        
        # Verify crisis hotline is communicated
        deltas = [m.get("text") for m in received if m.get("type") == "delta"]
        text = " ".join([d for d in deltas if d])
        assert "988" in text or "helpline" in text.lower()
