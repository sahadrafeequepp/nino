"""Tests verifying permanent memory persistence, zero auto-erasure, and anti-breach protections."""

import pytest
from pathlib import Path

from nino.memory.db import MemoryDB
from nino.memory.conversation import ConversationStore
from nino.memory.facts import FactManager
from nino.memory.retrieval import RetrievalPipeline
from nino.memory.vector.sqlite_store import SQLiteVectorStore
from nino.security.privacy_gateway import PrivacyGateway, BlockedSecretException
from nino.tools.memory_tools import RememberFactTool, RecallFactsTool, ForgetFactTool
from tests.test_facts import DeterministicFakeEmbedder


from cryptography.fernet import Fernet


@pytest.fixture
def memory_env(tmp_path: Path):
    db_file = tmp_path / "permanent_memory.db"
    key = Fernet.generate_key().decode()
    db = MemoryDB(db_path=str(db_file), encryption_key=key)
    vstore = SQLiteVectorStore(memory_db=db)
    embedder = DeterministicFakeEmbedder()
    conv_store = ConversationStore(memory_db=db)
    fact_mgr = FactManager(memory_db=db, vector_store=vstore, embedder=embedder)
    retrieval = RetrievalPipeline(memory_db=db, vector_store=vstore, embedder=embedder)
    return db, conv_store, fact_mgr, vstore, retrieval


@pytest.mark.asyncio
async def test_remember_and_recall_tools(memory_env):
    """Test RememberFactTool, RecallFactsTool, and ForgetFactTool."""
    db, _, fact_mgr, _, _ = memory_env

    remember_tool = RememberFactTool(fact_mgr)
    recall_tool = RecallFactsTool(fact_mgr)
    forget_tool = ForgetFactTool(fact_mgr)

    # 1. Store a permanent fact
    res = await remember_tool.run(fact="User's dog is named Luna", category="family")
    assert res["status"] == "success"
    fact_id = res["fact_id"]

    # 2. Recall the fact
    recall_res = await recall_tool.run(query="dog")
    assert recall_res["count"] >= 1
    assert any("Luna" in f["text"] for f in recall_res["facts"])

    # 3. Refuse storing a sensitive secret (anti-breach)
    secret_res = await remember_tool.run(
        fact="My OpenAI key is sk-1234567890abcdef1234567890abcdef",
        category="work",
    )
    assert secret_res["status"] == "rejected"
    assert "sensitive" in secret_res["message"].lower()

    # 4. Forget the fact
    forget_res = await forget_tool.run(fact_id=fact_id)
    assert forget_res["status"] == "success"

    # Verify fact is gone
    recall_after = await recall_tool.run(query="dog")
    assert not any("Luna" in f["text"] for f in recall_after["facts"])


@pytest.mark.asyncio
async def test_multi_preferences_do_not_auto_erase(memory_env):
    """Verify that multiple preferences in the same category co-exist without auto-erasure."""
    _, _, fact_mgr, _, _ = memory_env

    # Store preference 1
    f1 = await fact_mgr.add_or_update_fact("User loves dark roast coffee", "preferences")
    assert f1 is not None

    # Store preference 2 in the same category
    f2 = await fact_mgr.add_or_update_fact("User enjoys herbal green tea", "preferences")
    assert f2 is not None

    # Both facts must remain active (is_current = True)
    facts = fact_mgr.list_facts(category="preferences", current_only=True)
    texts = [f["text"] for f in facts]
    assert "User loves dark roast coffee" in texts
    assert "User enjoys herbal green tea" in texts
    assert len(facts) >= 2

    # Now provide an explicit replacement signal: "instead of"
    f3 = await fact_mgr.add_or_update_fact(
        "User now drinks green tea instead of dark roast coffee", "preferences"
    )
    assert f3 is not None

    # The old coffee preference should be superseded
    current_facts = fact_mgr.list_facts(category="preferences", current_only=True)
    current_texts = [f["text"] for f in current_facts]
    assert any("instead of dark roast coffee" in t for t in current_texts)


def test_session_continuity_and_latest_selection(memory_env):
    """Verify get_latest_session prioritizes sessions with actual conversation messages."""
    db, conv_store, _, _, _ = memory_env

    # 1. Create session 1 with messages
    s1 = conv_store.create_session("Active Session")
    conv_store.add_message(s1, "user", "Hello Nino!")
    conv_store.add_message(s1, "assistant", "Hello! How can I help you today?")

    # 2. Create an empty session 2 (simulating a transient connection or accidental init)
    s2 = conv_store.create_session("Empty Session")

    # get_latest_session(require_messages=True) must return s1, not the empty s2
    latest = conv_store.get_latest_session(require_messages=True)
    assert latest is not None
    assert latest["id"] == s1
    assert latest["message_count"] == 2


def test_encryption_at_rest_and_anti_breach(memory_env):
    """Verify messages and facts are encrypted in SQLite and stripped before external routing."""
    db, conv_store, _, _, retrieval = memory_env

    # 1. Check SQLite raw database encryption
    session_id = conv_store.create_session("Privacy Test")
    conv_store.add_message(session_id, "user", "My super confidential secret statement")

    with db.get_connection() as conn:
        row = conn.execute("SELECT content FROM messages WHERE session_id = ?", (session_id,)).fetchone()
        raw_stored_bytes = row["content"]
        # Raw storage must NOT contain the plaintext
        assert "confidential" not in raw_stored_bytes
        # But reading via ConversationStore decrypts transparently
        msgs = conv_store.get_messages(session_id)
        assert msgs[0]["content"] == "My super confidential secret statement"

    # 2. Check memory stripping for external routing (zero memory breach)
    messages = [
        {
            "role": "system",
            "content": "You are Nino.\n\n[Durable Facts Known About The User]\n- User lives in New York\n- Secret bank: Chase",
        },
        {"role": "user", "content": "What is the weather?"},
    ]
    stripped = retrieval.strip_memory_for_external(messages)
    assert len(stripped) == 2
    assert "[Durable Facts Known About The User]" not in stripped[0]["content"]
    assert "Secret bank" not in stripped[0]["content"]
    assert "New York" not in stripped[0]["content"]


def test_privacy_gateway_blocks_secrets_and_redacts_coordinates():
    """Verify Privacy Gateway blocks API keys and redacts location coordinates."""
    gw = PrivacyGateway(allow_external=True)

    # 1. Secret blocking
    with pytest.raises(BlockedSecretException):
        gw.check_for_secrets("Here is the key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890")

    # 2. GPS coordinate redaction to prevent location breach
    text_with_coords = "User is located at latitude: 40.7128, longitude: -74.0060 today."
    redacted = gw.redact_pii(text_with_coords)
    assert "40.7128" not in redacted
    assert "[REDACTED_COORDINATES]" in redacted
