"""Unit tests for persistent memory store, load, session restoration, and durable facts."""

import pytest
from nino.memory.conversation import ConversationStore
from nino.memory.db import MemoryDB


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_memory.db"
    return MemoryDB(db_path=str(db_file))


def test_session_lifecycle_and_restart_restore(temp_db):
    """Verify that messages and sessions can be stored, listed, and restored across restarts."""
    store1 = ConversationStore(temp_db)

    # 1. Create a session and store messages
    session_id = store1.create_session(title="Research Project")
    store1.add_message(session_id, "user", "Remember that my favourite programming language is Rust.")
    store1.add_message(session_id, "assistant", "I will remember that you love Rust!")

    # Verify session details
    sess_meta = store1.get_session(session_id)
    assert sess_meta is not None
    assert sess_meta["title"] == "Research Project"
    assert sess_meta["message_count"] == 2

    # Update title
    store1.update_session_title(session_id, "Updated Rust Project")
    updated_meta = store1.get_session(session_id)
    assert updated_meta["title"] == "Updated Rust Project"

    # 2. Simulate restarting the process/session with a new ConversationStore instance
    store2 = ConversationStore(temp_db)

    # List sessions
    sessions = store2.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id
    assert sessions[0]["title"] == "Updated Rust Project"
    assert sessions[0]["message_count"] == 2

    # Get latest session
    latest = store2.get_latest_session()
    assert latest is not None
    assert latest["id"] == session_id

    # Restore messages
    restored_msgs = store2.get_messages(session_id)
    assert len(restored_msgs) == 2
    assert restored_msgs[0]["role"] == "user"
    assert "Rust" in restored_msgs[0]["content"]
    assert restored_msgs[1]["role"] == "assistant"
    assert "love Rust" in restored_msgs[1]["content"]


@pytest.mark.asyncio
async def test_budgeted_messages_restoration(temp_db):
    """Verify that get_budgeted_messages restores context seamlessly."""
    store = ConversationStore(temp_db)
    session_id = store.create_session("Budget Test")
    store.add_message(session_id, "user", "Message 1")
    store.add_message(session_id, "assistant", "Reply 1")

    budgeted = await store.get_budgeted_messages(session_id, token_budget=500)
    assert len(budgeted) == 2
    assert budgeted[0]["content"] == "Message 1"
    assert budgeted[1]["content"] == "Reply 1"


def test_multi_user_history_isolation_and_admin_view(temp_db):
    """Verify users only see their own history, and admin sees all sessions labelled with usernames."""
    from nino.auth import create_user

    store = ConversationStore(temp_db)

    # 1. Create two separate user accounts
    alice = create_user(temp_db, "alice", "alicepass123")
    bob = create_user(temp_db, "bob", "bobpass456")

    # 2. Create sessions for Alice, Bob, and Admin
    sess_alice = store.create_session("Alice Private Work", user_id=alice["id"])
    store.add_message(sess_alice, "user", "Alice confidential data")

    sess_bob = store.create_session("Bob Meeting Notes", user_id=bob["id"])
    store.add_message(sess_bob, "user", "Bob confidential plan")

    sess_admin = store.create_session("Admin Server Setup", user_id="admin")
    store.add_message(sess_admin, "user", "Admin maintenance tasks")

    # 3. Alice lists her sessions: ONLY Alice's session appears
    alice_sessions = store.list_sessions(user_id=alice["id"])
    alice_ids = [s["id"] for s in alice_sessions]
    assert sess_alice in alice_ids
    assert sess_bob not in alice_ids
    assert sess_admin not in alice_ids

    # Alice cannot read Bob's session or messages
    assert store.get_session(sess_bob, user_id=alice["id"]) is None
    assert store.get_messages(sess_bob, user_id=alice["id"]) == []
    # Alice cannot delete Bob's session
    assert store.delete_session(sess_bob, user_id=alice["id"]) is False

    # 4. Bob lists his sessions: ONLY Bob's session appears
    bob_sessions = store.list_sessions(user_id=bob["id"])
    bob_ids = [s["id"] for s in bob_sessions]
    assert sess_bob in bob_ids
    assert sess_alice not in bob_ids
    assert sess_admin not in bob_ids

    # Bob cannot read Alice's session or messages
    assert store.get_session(sess_alice, user_id=bob["id"]) is None
    assert store.get_messages(sess_alice, user_id=bob["id"]) == []

    # 5. Admin lists sessions: Admin sees ALL sessions labelled with their respective usernames!
    admin_sessions = store.list_sessions(user_id="admin")
    admin_ids = [s["id"] for s in admin_sessions]
    assert sess_alice in admin_ids
    assert sess_bob in admin_ids
    assert sess_admin in admin_ids

    # Verify each session is labelled with the correct author username
    alice_item = next(s for s in admin_sessions if s["id"] == sess_alice)
    bob_item = next(s for s in admin_sessions if s["id"] == sess_bob)
    admin_item = next(s for s in admin_sessions if s["id"] == sess_admin)

    assert alice_item["username"] == "alice"
    assert bob_item["username"] == "bob"
    assert admin_item["username"] == "admin"

    # Admin can access Alice's and Bob's session details & messages
    admin_alice_sess = store.get_session(sess_alice, user_id="admin")
    assert admin_alice_sess is not None
    assert admin_alice_sess["username"] == "alice"
    alice_msgs = store.get_messages(sess_alice, user_id="admin")
    assert len(alice_msgs) == 1
    assert "Alice confidential data" in alice_msgs[0]["content"]

    admin_bob_sess = store.get_session(sess_bob, user_id="admin")
    assert admin_bob_sess is not None
    assert admin_bob_sess["username"] == "bob"
    bob_msgs = store.get_messages(sess_bob, user_id="admin")
    assert len(bob_msgs) == 1
    assert "Bob confidential plan" in bob_msgs[0]["content"]

    # Admin can delete any session
    assert store.delete_session(sess_bob, user_id="admin") is True
    assert store.get_session(sess_bob, user_id="admin") is None


def test_admin_login_auth_verification(temp_db):
    """Verify admin login with anjana@021 password and session token creation."""
    import secrets
    from nino.config import Settings
    from nino.auth import create_user_session, get_user_by_session_token

    settings = Settings(auth_token="anjana@021")
    password = "anjana@021"

    # Verify admin password matches auth_token
    assert secrets.compare_digest(password.encode("utf-8"), settings.auth_token.encode("utf-8"))

    # Verify creating session for admin
    token = create_user_session(temp_db, user_id="admin", username="admin")
    assert token

    # Verify session lookup returns admin with is_admin=True
    user = get_user_by_session_token(temp_db, token)
    assert user is not None
    assert user["id"] == "admin"
    assert user["username"] == "admin"
    assert user["is_admin"] is True
