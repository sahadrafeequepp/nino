"""Integration tests for the FastAPI Gateway endpoints."""

import json
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from nino.config import Settings, get_settings
from nino.main import app, get_adapter
from nino.models.base import LLMAdapter


class MockLLMAdapter(LLMAdapter):
    """Test double for LLMAdapter."""

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        return f"Echo: {messages[-1]['content']}"

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs):
        tokens = ["Echo:", " ", messages[-1]["content"]]
        for tok in tokens:
            yield tok


from nino.api.memory import get_embedder
from nino.memory.db import MemoryDB
from nino.memory.conversation import ConversationStore
from tests.test_facts import DeterministicFakeEmbedder


@pytest.fixture
def mock_env(tmp_path: Path):
    audit_file = tmp_path / "audit_test.jsonl"
    db_file = tmp_path / "test_api.db"
    settings = Settings(
        auth_token="secure-test-token",
        model_adapter="ollama",
        model_name="gemma2:2b",
        db_path=str(db_file),
        audit_log_path=str(audit_file),
        log_content=False,
    )
    test_db = MemoryDB(db_path=str(db_file))
    test_conv_store = ConversationStore(test_db)
    orig_db = getattr(app.state, "memory_db", None)
    orig_conv = getattr(app.state, "conv_store", None)
    app.state.memory_db = test_db
    app.state.conv_store = test_conv_store

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_adapter] = lambda: MockLLMAdapter()
    app.dependency_overrides[get_embedder] = lambda: DeterministicFakeEmbedder()
    yield settings, audit_file
    app.dependency_overrides.clear()
    app.state.memory_db = orig_db
    app.state.conv_store = orig_conv


@pytest.mark.asyncio
async def test_health_check(mock_env):
    settings, _ = mock_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["adapter"] == "ollama"
        assert data["model"] == "gemma2:2b"


@pytest.mark.asyncio
async def test_post_chat_json(mock_env):
    settings, audit_file = mock_env
    headers = {"Authorization": f"Bearer {settings.auth_token}"}
    payload = {"messages": [{"role": "user", "content": "What is 2+2?"}]}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/chat", headers=headers, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["reply"] == "Echo: What is 2+2?"
        assert data["model"] == "gemma2:2b"

    # Verify audit log was recorded by middleware
    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    audit_entry = json.loads(lines[-1])
    assert audit_entry["endpoint"] == "/chat"
    assert audit_entry["status_code"] == 200
    assert audit_entry["content"] == "[CONTENT_OMITTED]"


@pytest.mark.asyncio
async def test_post_chat_stream_sse(mock_env):
    settings, audit_file = mock_env
    headers = {"Authorization": f"Bearer {settings.auth_token}"}
    payload = {"messages": [{"role": "user", "content": "Streaming test"}]}

    collected_chunks = []
    received_done = False

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with client.stream("POST", "/chat/stream", headers=headers, json=payload) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                raw_data = line[6:].strip()
                if raw_data == "[DONE]":
                    received_done = True
                    break
                chunk_json = json.loads(raw_data)
                collected_chunks.append(chunk_json["content"])

    assert received_done is True
    assert "".join(collected_chunks) == "Echo: Streaming test"


@pytest.mark.asyncio
async def test_memory_api_endpoints(mock_env):
    settings, _ = mock_env
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Export memory
        res_exp = await client.get("/memory/export", headers=headers)
        assert res_exp.status_code == 200
        assert "facts" in res_exp.json()

        # 2. List facts
        res_facts = await client.get("/memory/facts", headers=headers)
        assert res_facts.status_code == 200
        assert "facts" in res_facts.json()

        # 3. Forget memory preview
        res_forget = await client.post("/memory/forget", headers=headers, json={"query": "test", "confirm": False})
        assert res_forget.status_code == 200
        assert res_forget.json()["confirmed"] is False

        # 4. Delete all memory without confirm -> 400
        res_del_fail = await client.delete("/memory/all", headers=headers)
        assert res_del_fail.status_code == 400

        # 5. Delete all memory with confirm -> 200
        res_del_ok = await client.delete("/memory/all?confirm=true", headers=headers)
        assert res_del_ok.status_code == 200
        assert res_del_ok.json()["status"] == "all_memory_purged"


@pytest.mark.asyncio
async def test_chat_ui_endpoints(mock_env):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Test root /
        res_root = await client.get("/")
        assert res_root.status_code == 200
        assert "text/html" in res_root.headers.get("content-type", "")
        assert "Nino" in res_root.text

        # 2. Test /chat-ui
        res_ui = await client.get("/chat-ui")
        assert res_ui.status_code == 200
        assert "text/html" in res_ui.headers.get("content-type", "")

        # 3. Test /api/client-config
        res_cfg = await client.get("/api/client-config")
        assert res_cfg.status_code == 200
        cfg_data = res_cfg.json()
        assert cfg_data["assistant"] == "Nino"
        assert "auth_token" in cfg_data


@pytest.mark.asyncio
async def test_api_tts_endpoints(mock_env, monkeypatch):
    class DummyTTS:
        sample_rate = 22050
        async def synthesize(self, text: str) -> bytes:
            return b"\x00\x00" * 500

    import nino.main
    monkeypatch.setattr(nino.main, "get_tts_adapter", lambda: DummyTTS())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Test POST /api/tts
        res_post = await client.post("/api/tts", json={"text": "Hello Nino"})
        assert res_post.status_code == 200
        assert res_post.headers["content-type"] == "audio/wav"
        assert res_post.content[:4] == b"RIFF"

        # 2. Test GET /api/tts
        res_get = await client.get("/api/tts?text=Hello+Nino")
        assert res_get.status_code == 200
        assert res_get.headers["content-type"] == "audio/wav"
        assert res_get.content[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_parse_document_endpoint(mock_env):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Test text file upload
        file_content = b"def hello_world():\n    return 'Hello from Nino!'"
        files = {"file": ("script.py", file_content, "text/x-python")}
        res = await client.post("/api/parse-document", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["filename"] == "script.py"
        assert "Hello from Nino!" in data["text"]


@pytest.mark.asyncio
async def test_chat_handles_empty_history_gracefully(mock_env):
    settings, _ = mock_env
    headers = {"Authorization": f"Bearer {settings.auth_token}"}
    payload = {
        "messages": [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": ""},  # Previously triggered 422
            {"role": "user", "content": "Second message"},
        ]
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/chat", headers=headers, json=payload)
        assert res.status_code == 200
        assert "Echo: Second message" in res.json()["reply"]


@pytest.mark.asyncio
async def test_session_management_endpoints(mock_env):
    settings, _ = mock_env
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Create a new session
        res_create = await client.post("/memory/sessions", headers=headers, json={"title": "Test Chat"})
        assert res_create.status_code == 200
        data_create = res_create.json()
        assert data_create["status"] == "created"
        sess_id = data_create["session_id"]
        assert sess_id is not None

        # 2. Post a chat in that session
        chat_payload = {
            "messages": [{"role": "user", "content": "How are you doing today?"}],
            "session_id": sess_id,
        }
        res_chat = await client.post("/chat", headers=headers, json=chat_payload)
        assert res_chat.status_code == 200

        # 3. List sessions
        res_list = await client.get("/memory/sessions", headers=headers)
        assert res_list.status_code == 200
        sessions = res_list.json().get("sessions", [])
        assert any(s["id"] == sess_id for s in sessions)

        # 4. Get session messages
        res_sess = await client.get(f"/memory/sessions/{sess_id}", headers=headers)
        assert res_sess.status_code == 200
        sess_data = res_sess.json()
        assert sess_data["session"]["id"] == sess_id
        assert len(sess_data["messages"]) >= 2
        assert sess_data["messages"][0]["role"] == "user"
        assert sess_data["messages"][0]["content"] == "How are you doing today?"

        # 5. Delete the session
        res_del = await client.delete(f"/memory/sessions/{sess_id}", headers=headers)
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "deleted"

        # 6. Verify 404 after deletion
        res_get_del = await client.get(f"/memory/sessions/{sess_id}", headers=headers)
        assert res_get_del.status_code == 404


@pytest.mark.asyncio
async def test_client_config_lan_and_trusted_networks(mock_env):
    settings, _ = mock_env

    # 1. Localhost client
    async with AsyncClient(
        transport=ASGITransport(app=app, client=("127.0.0.1", 54321)),
        base_url="http://test",
    ) as client:
        res = await client.get("/api/client-config")
        assert res.status_code == 200
        data = res.json()
        assert data["is_local"] is True
        assert data["auth_token"] == settings.auth_token

    # 2. LAN Wi-Fi client (192.168.1.100)
    async with AsyncClient(
        transport=ASGITransport(app=app, client=("192.168.1.100", 54321)),
        base_url="http://test",
    ) as client:
        res = await client.get("/api/client-config")
        assert res.status_code == 200
        data = res.json()
        assert data["is_local"] is True
        assert data["auth_token"] == settings.auth_token

    # 3. Tailscale client (100.89.60.45)
    async with AsyncClient(
        transport=ASGITransport(app=app, client=("100.89.60.45", 54321)),
        base_url="http://test",
    ) as client:
        res = await client.get("/api/client-config")
        assert res.status_code == 200
        data = res.json()
        assert data["is_local"] is True
        assert data["auth_token"] == settings.auth_token

    # 4. Untrusted public internet client (8.8.8.8)
    async with AsyncClient(
        transport=ASGITransport(app=app, client=("8.8.8.8", 54321)),
        base_url="http://test",
    ) as client:
        res = await client.get("/api/client-config")
        assert res.status_code == 200
        data = res.json()
        assert data["is_local"] is False
        assert data["auth_token"] == ""


@pytest.mark.asyncio
async def test_auth_registration_login_and_cookie(mock_env):
    settings, _ = mock_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Initially unauthenticated
        res_me1 = await client.get("/api/auth/me")
        assert res_me1.status_code == 200
        assert res_me1.json()["authenticated"] is False

        # 2. Register user
        reg_payload = {"username": "testuser", "password": "password123"}
        res_reg = await client.post("/api/auth/register", json=reg_payload)
        assert res_reg.status_code == 200
        data_reg = res_reg.json()
        assert data_reg["status"] == "ok"
        assert data_reg["user"]["username"] == "testuser"
        assert "nino_session" in client.cookies

        # 3. Check /me with cookie
        res_me2 = await client.get("/api/auth/me")
        assert res_me2.status_code == 200
        data_me2 = res_me2.json()
        assert data_me2["authenticated"] is True
        assert data_me2["user"]["username"] == "testuser"

        # 4. Sign out
        res_logout = await client.post("/api/auth/logout")
        assert res_logout.status_code == 200

        # 5. Check /me after logout
        res_me3 = await client.get("/api/auth/me")
        assert res_me3.status_code == 200
        assert res_me3.json()["authenticated"] is False

        # 6. Wrong password login
        res_fail = await client.post("/api/auth/login", json={"username": "testuser", "password": "wrong"})
        assert res_fail.status_code == 401

        # 7. Correct password login
        res_login = await client.post("/api/auth/login", json={"username": "testuser", "password": "password123"})
        assert res_login.status_code == 200
        assert "nino_session" in client.cookies


@pytest.mark.asyncio
async def test_user_session_chat_history_isolation(mock_env):
    settings, _ = mock_env

    # 1. Register Alice and create Alice's chat session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as alice_client:
        await alice_client.post("/api/auth/register", json={"username": "alice", "password": "alicepass123"})
        res_sess_a = await alice_client.post("/memory/sessions", json={"title": "Alice Private Project"})
        assert res_sess_a.status_code == 200
        alice_session_id = res_sess_a.json()["session_id"]

        # Alice sends a chat message
        res_chat_a = await alice_client.post(
            "/chat",
            json={"session_id": alice_session_id, "messages": [{"role": "user", "content": "My secret is pineapple"}]},
        )
        assert res_chat_a.status_code == 200

        # Alice lists her sessions
        res_list_a = await alice_client.get("/memory/sessions")
        assert res_list_a.status_code == 200
        alice_sessions = res_list_a.json()["sessions"]
        assert any(s["id"] == alice_session_id for s in alice_sessions)

    # 2. Register Bob in a separate client/session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as bob_client:
        await bob_client.post("/api/auth/register", json={"username": "bob", "password": "bobpassword456"})

        # Bob lists his sessions -> Alice's session must NOT appear!
        res_list_b = await bob_client.get("/memory/sessions")
        assert res_list_b.status_code == 200
        bob_sessions = res_list_b.json()["sessions"]
        assert not any(s["id"] == alice_session_id for s in bob_sessions)

        # Bob attempts direct access to Alice's session -> 404 Not Found
        res_get_a = await bob_client.get(f"/memory/sessions/{alice_session_id}")
        assert res_get_a.status_code == 404

        # Bob attempts to delete Alice's session -> 404 Not Found
        res_del_a = await bob_client.delete(f"/memory/sessions/{alice_session_id}")
        assert res_del_a.status_code == 404

        # Bob creates his own session
        res_sess_b = await bob_client.post("/memory/sessions", json={"title": "Bob Meeting Notes"})
        assert res_sess_b.status_code == 200
        bob_session_id = res_sess_b.json()["session_id"]
        assert bob_session_id != alice_session_id

        # Bob's session list now contains Bob's session, but still NOT Alice's
        res_list_b2 = await bob_client.get("/memory/sessions")
        assert res_list_b2.status_code == 200
        b_sessions = res_list_b2.json()["sessions"]
        assert any(s["id"] == bob_session_id for s in b_sessions)
        assert not any(s["id"] == alice_session_id for s in b_sessions)

        # 3. Admin logs in with master auth token
        admin_headers = {"Authorization": f"Bearer {settings.auth_token}"}
        res_admin_list = await bob_client.get("/memory/sessions", headers=admin_headers)
        assert res_admin_list.status_code == 200
        admin_sessions = res_admin_list.json()["sessions"]

        # Admin must see BOTH Alice's and Bob's sessions, labeled with their respective usernames!
        session_ids = [s["id"] for s in admin_sessions]
        assert alice_session_id in session_ids
        assert bob_session_id in session_ids

        alice_entry = next(s for s in admin_sessions if s["id"] == alice_session_id)
        bob_entry = next(s for s in admin_sessions if s["id"] == bob_session_id)
        assert alice_entry["username"] == "alice"
        assert bob_entry["username"] == "bob"

        # Admin can view Alice's session details and messages
        res_admin_alice_get = await bob_client.get(f"/memory/sessions/{alice_session_id}", headers=admin_headers)
        assert res_admin_alice_get.status_code == 200
        assert res_admin_alice_get.json()["session"]["username"] == "alice"
        assert len(res_admin_alice_get.json()["messages"]) > 0


@pytest.mark.asyncio
async def test_auth_registration_validation_and_special_characters(mock_env):
    """Verify registration handles spaces, @, duplicate usernames, and short passwords."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Register with username containing space and @
        res1 = await client.post("/api/auth/register", json={"username": "Jane Doe", "password": "securepassword123"})
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["status"] == "ok"
        assert data1["user"]["username"] == "Jane Doe"

        # 2. Duplicate registration attempt fails with 400
        res_dup = await client.post("/api/auth/register", json={"username": "jane doe", "password": "anotherpassword"})
        assert res_dup.status_code == 400
        assert "already taken" in res_dup.json()["detail"].lower()

        # 3. Register with email-style username
        res_email = await client.post("/api/auth/register", json={"username": "user@domain.com", "password": "mypassword"})
        assert res_email.status_code == 200
        assert res_email.json()["user"]["username"] == "user@domain.com"

        # 4. Too short password rejected
        res_short_pw = await client.post("/api/auth/register", json={"username": "validname", "password": "12"})
        assert res_short_pw.status_code == 422 or res_short_pw.status_code == 400






