"""Unit tests for Telegram Bot security gatekeeper, Antigravity Agent, and voice helpers."""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Mock telegram if not installed in local test environment
class MockBotCommand:
    def __init__(self, command, description=""):
        self.command = command
        self.description = description

try:
    import telegram
    from telegram.ext import filters
except ImportError:
    mock_telegram = MagicMock()
    mock_filters = MagicMock()
    class DummyBaseFilter:
        pass
    mock_filters.BaseFilter = DummyBaseFilter
    mock_telegram.BotCommand = MockBotCommand
    sys.modules["telegram"] = mock_telegram
    sys.modules["telegram.ext"] = MagicMock()
    sys.modules["telegram.ext"].filters = mock_filters
    sys.modules["telegram.constants"] = MagicMock()
    sys.modules["telegram.request"] = MagicMock()

if "telegram" in sys.modules and not isinstance(sys.modules["telegram"].BotCommand, type):
    sys.modules["telegram"].BotCommand = MockBotCommand


from telegram_bot.agent import NinoAgent, VaultTool, DENIAL_POLICIES
from telegram_bot.voice_utils import sanitize_text_for_speech
from telegram_bot.bot import OwnerFilter



def test_owner_filter_authorization():
    """Verify that OwnerFilter allows ONLY the authorized user ID."""
    filt = OwnerFilter()

    # Mock authorized message
    mock_auth_msg = MagicMock()
    mock_auth_msg.from_user.id = 1234567890

    # Test with matching ID
    from telegram_bot import bot
    bot.AUTHORIZED_USER_ID = 1234567890

    assert filt.filter(mock_auth_msg) is True

    # Mock unauthorized intruder
    mock_intruder_msg = MagicMock()
    mock_intruder_msg.from_user.id = 9999999999

    assert filt.filter(mock_intruder_msg) is False

    # Empty user
    mock_none_user = MagicMock()
    mock_none_user.from_user = None
    assert filt.filter(mock_none_user) is False


def test_vault_tool_security(tmp_path: Path):
    """Verify VaultTool reads notes and strictly denies path traversal."""
    vault = VaultTool(vault_dir=tmp_path)

    # 1. Create a safe note
    safe_note = tmp_path / "notes.txt"
    safe_note.write_text("Secret project plan", encoding="utf-8")

    files = vault.list_files()
    assert "notes.txt" in files

    content = vault.read_file("notes.txt")
    assert content["content"] == "Secret project plan"
    assert content["size_bytes"] == len("Secret project plan")

    # 2. Test Path Traversal rejection
    traversal_attempt = vault.read_file("../../../etc/passwd")
    assert "error" in traversal_attempt
    assert "Access Denied" in traversal_attempt["error"] or "not found" in traversal_attempt["error"]

    traversal_attempt2 = vault.read_file("..\\..\\windows\\system32\\config")
    assert "error" in traversal_attempt2


def test_speech_sanitization():
    """Verify that sanitize_text_for_speech cleans markdown, emojis, and actions for natural vocalization."""
    raw = (
        "*chuckles playfully* Here is the code:\n"
        "```python\ndef foo():\n    return 42\n```\n"
        "Check [this link](https://example.com) and *bold text* with `inline code` 🚀 😊 :fire: <3 :D."
    )
    cleaned = sanitize_text_for_speech(raw)

    assert "```" not in cleaned
    assert "def foo():" not in cleaned  # Code block omitted
    assert "[code snippet omitted]" in cleaned
    assert "https://example.com" not in cleaned
    assert "bold text" in cleaned
    assert "inline code" in cleaned
    # Verify emojis and actions are stripped
    assert "chuckles" not in cleaned
    assert "🚀" not in cleaned
    assert "😊" not in cleaned
    assert ":fire:" not in cleaned
    assert "<3" not in cleaned
    assert ":D" not in cleaned



def test_antigravity_agent_policies():
    """Verify that NinoAgent configures Antigravity denial policies."""
    agent = NinoAgent(api_key="test-key", mode="companion")
    assert "deny('command(*)')" in DENIAL_POLICIES
    assert "deny('read_file(*)')" in DENIAL_POLICIES
    assert "deny('execute_url(*)')" in DENIAL_POLICIES

    prompt = agent._build_full_system_prompt()
    assert "You are Nino" in prompt
    assert "[COMPANION MODE]" in prompt

    agent.reset_conversation()
    assert len(agent.history) == 0


@pytest.mark.asyncio
async def test_nino_agent_ollama_chat(monkeypatch):
    """Verify that NinoAgent chats with Ollama API endpoint when online."""
    import httpx

    agent = NinoAgent(base_url="http://127.0.0.1:11434", model_name="gemma2:2b")

    # Mock health check as online
    async def mock_is_online():
        return True

    monkeypatch.setattr(agent, "is_ollama_online", mock_is_online)

    async def mock_post(self, url, json=None, **kwargs):
        assert url == "http://127.0.0.1:11434/api/chat"
        assert json["model"] == "gemma2:2b"
        assert "Hello Nino" in json["messages"][-1]["content"]
        assert "[Current Timestamp:" in json["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Hey there! How's it going?"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    reply, engine = await agent.chat("Hello Nino")
    assert reply == "Hey there! How's it going?"
    assert engine == "ollama"
    assert len(agent.history) == 2
    assert agent.history[0]["content"] == "Hello Nino"


@pytest.mark.asyncio
async def test_nino_agent_offline_ollama_reporting(monkeypatch):
    """Verify that NinoAgent reports Ollama offline status when local system is unreachable."""
    agent = NinoAgent(
        base_url="http://127.0.0.1:11434",
        model_name="gemma2:2b",
    )

    # Mock Ollama as offline (e.g. PC turned off)
    async def mock_offline():
        return False

    monkeypatch.setattr(agent, "is_ollama_online", mock_offline)

    reply, engine = await agent.chat("Are you there?")
    assert "Ollama is Offline" in reply
    assert engine == "offline"


def test_asymmetric_brain_privacy(tmp_path: Path):
    """Verify that Telegram bot sees shared/telegram facts, but private PC memories are shielded."""
    import sqlite3
    db_file = tmp_path / "nino_test.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE facts (
            id TEXT PRIMARY KEY,
            text TEXT,
            category TEXT,
            source_message_id TEXT,
            is_current INTEGER DEFAULT 1,
            origin TEXT DEFAULT 'pc',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # 1. Private PC conversation memory (sensitive/desktop work)
    conn.execute(
        "INSERT INTO facts (id, text, category, origin) VALUES ('1', 'User had private counseling session about confidential work', 'confidential', 'pc')"
    )
    # 2. Public profile fact (shared)
    conn.execute(
        "INSERT INTO facts (id, text, category, origin) VALUES ('2', 'User lives in Vadakara, Calicut', 'location', 'pc')"
    )
    # 3. Telegram-created memory
    conn.execute(
        "INSERT INTO facts (id, text, category, origin) VALUES ('3', 'User ordered favorite Ethiopian coffee from phone', 'preference', 'telegram')"
    )
    conn.commit()
    conn.close()

    from telegram_bot.agent import NinoBrainBridge
    bridge = NinoBrainBridge(str(db_file))

    # Test Telegram Context
    context = bridge.get_shared_context()
    assert "Vadakara, Calicut" in context  # Profile/shared fact is present
    assert "Ethiopian coffee" in context   # Telegram fact is present
    assert "counseling session" not in context  # Private PC memory is SHIELDED!

    # Test saving new Telegram fact
    assert bridge.save_telegram_fact("User is traveling to Kochi tomorrow") is True

    # Check that new fact is marked with origin='telegram'
    conn = sqlite3.connect(str(db_file))
    row = conn.execute("SELECT origin, text FROM facts WHERE text LIKE '%Kochi%'").fetchone()
    conn.close()
    assert row[0] == "telegram"


def test_data_collector_and_media_archiving(tmp_path: Path):
    """Verify DataCollector creates directory structure, tracks users, logs interactions, and archives media."""
    import json
    from telegram_bot.data_collector import DataCollector, IP_NOTE

    collector = DataCollector(base_dir=tmp_path / "bot_saved_data")

    # Mock user object
    class DummyUser:
        id = 987654321
        username = "guest_tester"
        first_name = "Jane"
        last_name = "Doe"

    dummy_user = DummyUser()

    # 1. Record user
    rec = collector.record_user(dummy_user, is_owner=False)
    assert rec["user_id"] == 987654321
    assert rec["username"] == "guest_tester"
    assert rec["role"] == "guest"
    assert rec["ip_address"] == IP_NOTE
    assert rec["phone_number"] is None

    # Verify users.json on disk
    users_data = json.loads((tmp_path / "bot_saved_data" / "users.json").read_text(encoding="utf-8"))
    assert "987654321" in users_data
    assert users_data["987654321"]["first_name"] == "Jane"

    # 2. Update phone number (e.g. from contact card)
    assert collector.update_phone(987654321, "+919876543210") is True
    users_data = json.loads((tmp_path / "bot_saved_data" / "users.json").read_text(encoding="utf-8"))
    assert users_data["987654321"]["phone_number"] == "+919876543210"

    # 3. Archive media
    photo_payload = b"\xff\xd8\xff\xe0mock_jpeg_stream_data"
    media_rel_path = collector.save_media(987654321, photo_payload, "test_pic.jpg")
    assert "987654321" in media_rel_path
    assert media_rel_path.endswith(".jpg")

    # Verify media file exists on disk
    user_media_folder = tmp_path / "bot_saved_data" / "media" / "987654321"
    assert user_media_folder.exists()
    saved_files = list(user_media_folder.glob("*.jpg"))
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == photo_payload

    # 4. Log interaction
    log_entry = collector.log_interaction(
        dummy_user,
        message_type="photo",
        content="Check this photo",
        media_path=media_rel_path,
        is_owner=False,
    )
    assert log_entry["type"] == "photo"
    assert log_entry["role"] == "guest"

    # Verify interaction line in interactions.jsonl
    lines = (tmp_path / "bot_saved_data" / "interactions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    last_log = json.loads(lines[-1])
    assert last_log["user_id"] == 987654321
    assert last_log["content"] == "Check this photo"
    assert last_log["media_path"] == media_rel_path


def test_agent_role_privacy_separation():
    """Verify NinoAgent protects owner memories from guest prompts and isolates per-user history."""
    agent = NinoAgent(api_key="test-key", mode="companion")

    # Mock shared context on brain
    agent.brain.get_shared_context = MagicMock(return_value="[Secret Facts]: Owner loves dark chocolate")

    # 1. Owner System Prompt
    owner_prompt = agent._build_full_system_prompt(active_engine="ollama", is_owner=True)
    assert "[Owner Access]" in owner_prompt
    assert "Owner loves dark chocolate" in owner_prompt

    # 2. Guest System Prompt
    guest_prompt = agent._build_full_system_prompt(active_engine="ollama", is_owner=False)
    assert "[Guest Access]" in guest_prompt
    assert "preserve your owner's privacy" in guest_prompt
    assert "Owner loves dark chocolate" not in guest_prompt  # STRICTLY HIDDEN!

    # 3. Multi-user history isolation
    hist_owner = agent._get_user_history(111)
    hist_guest = agent._get_user_history(222)

    hist_owner.append({"role": "user", "content": "Owner private chat"})
    hist_guest.append({"role": "user", "content": "Guest public chat"})

    assert len(agent._get_user_history(111)) == 1
    assert agent._get_user_history(111)[0]["content"] == "Owner private chat"

    assert len(agent._get_user_history(222)) == 1
    assert agent._get_user_history(222)[0]["content"] == "Guest public chat"

    # Reset guest history should not affect owner
    agent.reset_conversation(user_id=222)
    assert len(agent._get_user_history(222)) == 0
    assert len(agent._get_user_history(111)) == 1


@pytest.mark.asyncio
async def test_link_inspector_security():
    """Verify LinkInspector catches SSRF, data-breaching parameters, and cleans safe HTML."""
    from telegram_bot.link_inspector import LinkInspector

    inspector = LinkInspector()

    # 1. URL extraction
    text = "Hey Nino, check out https://example.com/blog and https://test.org/news?id=42!"
    urls = inspector.extract_urls(text)
    assert "https://example.com/blog" in urls
    assert "https://test.org/news?id=42" in urls

    # 2. SSRF block on loopback
    safe, err, meta = await inspector.fetch_and_scan_url("http://127.0.0.1:8080/secret")
    assert safe is False
    assert "Firewall Blocked" in err
    assert meta["stage"] == "firewall"

    # 3. SSRF block on cloud metadata IP
    safe, err, meta = await inspector.fetch_and_scan_url("http://169.254.169.254/latest/meta-data")
    assert safe is False
    assert "Firewall Blocked" in err

    # 4. Data breach parameter heuristic detection
    safe, err, meta = await inspector.fetch_and_scan_url("https://malicious.org/exfiltrate?token=eyJhbGciOi")
    assert safe is False
    assert "Security Alert" in err or "Data-breach" in err

    # 5. Clean HTML text extraction
    raw_html = b"<html><head><title>Test Article</title><script>alert(1)</script></head><body><h1>Welcome</h1><p>This is safe content.</p></body></html>"
    cleaned = inspector._extract_clean_text(raw_html, "text/html")
    assert "Test Article" in cleaned
    assert "This is safe content" in cleaned
    assert "alert(1)" not in cleaned  # Scripts removed!


def test_file_reader_antivirus_and_formats(tmp_path: Path):
    """Verify inspect_and_read_file scans antivirus and parses PDF, Word, text, and media."""
    import io
    import zipfile
    from telegram_bot.file_reader import inspect_and_read_file

    # 1. Antivirus blocks dangerous extensions
    res_exe = inspect_and_read_file(b"echo Malicious", "payload.bat")
    assert res_exe.is_safe is False
    assert res_exe.file_type == "blocked"
    assert "Antivirus Alert" in res_exe.error_message

    # 2. Antivirus blocks masqueraded executable magic bytes (MZ header named readme.txt)
    res_masq = inspect_and_read_file(b"MZ\x90\x00\x03\x00\x00\x00fake_pe", "readme.txt")
    assert res_masq.is_safe is False
    assert res_masq.file_type == "blocked"

    # 3. Plain text / code reading
    res_py = inspect_and_read_file(b"def add(a, b):\n    return a + b\n", "solution.py")
    assert res_py.is_safe is True
    assert res_py.file_type == "text"
    assert "def add(a, b):" in res_py.text_content

    # 4. Image routing
    res_img = inspect_and_read_file(b"\xff\xd8\xff\xe0jpeg_data", "photo.jpg")
    assert res_img.is_safe is True
    assert res_img.file_type == "image"
    assert res_img.image_bytes is not None

    # 5. Video routing
    res_vid = inspect_and_read_file(b"\x00\x00\x00\x18ftypmp42video_data", "clip.mp4")
    assert res_vid.is_safe is True
    assert res_vid.file_type == "video"
    assert res_vid.video_bytes is not None

    # 6. Word Document (.docx) parsing via zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as docx_zip:
        docx_zip.writestr(
            "word/document.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                <w:body>
                    <w:p><w:t>Project Milestone 1 Summary</w:t></w:p>
                    <w:p><w:t>All tests are green and passing.</w:t></w:p>
                </w:body>
            </w:document>""",
        )
    docx_bytes = buf.getvalue()
    res_docx = inspect_and_read_file(docx_bytes, "report.docx")
    assert res_docx.is_safe is True
    assert res_docx.file_type == "docx"
    assert "Project Milestone 1 Summary" in res_docx.text_content
    assert "All tests are green" in res_docx.text_content


def test_video_storyboard_generation(tmp_path: Path):
    """Verify video inspector can extract metadata and storyboard from a video file."""
    import cv2
    import numpy as np
    from telegram_bot.video_inspector import generate_video_storyboard, inspect_video_file

    # Create a tiny 0.5s synthetic MP4 video file
    video_path = tmp_path / "test_synth.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (160, 120))

    for frame_idx in range(6):
        # Create distinct color frames
        color = (frame_idx * 40, 100, 200)
        img = np.full((120, 160, 3), color, dtype=np.uint8)
        writer.write(img)
    writer.release()

    meta = inspect_video_file(video_path)
    assert meta is not None
    assert meta.width == 160
    assert meta.height == 120
    assert meta.total_frames >= 5

    storyboard = generate_video_storyboard(video_path, num_frames=4)
    assert storyboard is not None
    assert len(storyboard) > 100  # Valid encoded JPEG bytes


def test_inbuilt_location_clock():
    """Verify LocationClock tracks local time, formats reports, syncs GPS, and injects into LLM prompt."""
    from telegram_bot.clock import LocationClock
    from telegram_bot.agent import NinoAgent

    clk = LocationClock(
        default_city="Vadakara",
        default_region="Kerala",
        default_country="India",
        default_lat=11.5978,
        default_lon=75.5814,
        default_timezone="Asia/Kolkata",
    )

    # 1. Exact local time & timezone
    now = clk.now()
    assert now.tzinfo is not None
    assert "+05:30" in now.strftime("%z") or "IST" in now.strftime("%Z")

    # 2. Clock report formatting
    report = clk.get_clock_report()
    assert "Nino Inbuilt Location Clock" in report
    assert "Vadakara" in report
    assert "Timezone" in report
    assert "Synchronized & Active" in report

    # 3. GPS synchronization
    res = clk.sync_from_gps(51.5074, -0.1278, location_name="London, UK", tz_name="Europe/London")
    assert clk.latitude == 51.5074
    assert clk.longitude == -0.1278
    assert clk.city == "London, UK"
    assert clk.timezone_name == "Europe/London"
    assert "local_time" in res

    # 4. LLM Prompt temporal context injection
    prompt_ctx = clk.get_prompt_context()
    assert "[Inbuilt Clock & Location Synchronization]" in prompt_ctx
    assert "London, UK" in prompt_ctx
    assert "Europe/London" in prompt_ctx

    # 5. NinoAgent prompt contains clock context and real-time context
    agent = NinoAgent(api_key="test-key", mode="companion")
    full_prompt = agent._build_full_system_prompt()
    assert "[Inbuilt Clock & Location Synchronization]" in full_prompt
    assert "CURRENT REAL-TIME CONTEXT: Today is" in full_prompt


@pytest.mark.asyncio
async def test_live_host_timestamp_and_ask_nino(monkeypatch):
    """Verify get_current_time_str formats host timestamp and ask_nino handles both time queries and dynamic prompt injection."""
    from telegram_bot.clock import get_current_time_str, is_time_query, get_time_conversational_reply
    from telegram_bot.agent import ask_nino, history, NinoAgent

    # 1. Test get_current_time_str
    ts_str = get_current_time_str()
    assert any(day in ts_str for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    assert ":" in ts_str

    # 2. Test exact time query resolution (including typos like 'rell me the time')
    assert is_time_query("nino rell me the time") is True
    assert is_time_query("nino tell me the time") is True
    assert is_time_query("what time is it") is True
    assert is_time_query("time ethrayaayi machan") is True
    assert is_time_query("tell me a joke") is False

    time_reply = await ask_nino("nino rell me the time")
    assert "It is currently" in time_reply or "Vadakara" in time_reply
    assert ":" in time_reply

    # 3. Test ask_nino dynamically updating prompt context for general queries
    async def fake_chat(self, user_prompt, **kwargs):
        return f"Echoing back at you: {user_prompt}", "ollama"

    monkeypatch.setattr(NinoAgent, "chat", fake_chat)

    general_reply = await ask_nino("Explain quantum computing")
    assert "Echoing back at you: Explain quantum computing" in general_reply
    assert len(history) >= 2
    assert history[0]["role"] == "system"
    assert "CURRENT REAL-TIME CONTEXT: Today is" in history[0]["content"]
    assert "always rely on the [Current Timestamp]" in history[0]["content"]
    assert history[-2]["role"] == "user"
    assert "[Current Timestamp:" in history[-2]["content"]
    assert "Explain quantum computing" in history[-2]["content"]
    assert history[-1]["role"] == "assistant"


def test_telegram_reminder_system(tmp_path):
    """Verify ReminderManager creation, time parsing, SQLite persistence, and due delivery."""
    from datetime import datetime, timedelta
    from telegram_bot.reminders import ReminderManager, extract_reminder_from_text, parse_time_expression

    # 1. Time expression parsing
    now = datetime(2026, 10, 5, 20, 0, 0)
    dt_sec, lbl_sec = parse_time_expression("30s", base_dt=now)
    assert dt_sec == now + timedelta(seconds=30)
    assert "30 second" in lbl_sec

    dt_min, lbl_min = parse_time_expression("15m", base_dt=now)
    assert dt_min == now + timedelta(minutes=15)
    assert "15 minute" in lbl_min

    dt_hr, lbl_hr = parse_time_expression("2h", base_dt=now)
    assert dt_hr == now + timedelta(hours=2)

    dt_tod, lbl_tod = parse_time_expression("9:30pm", base_dt=now)
    assert dt_tod.hour == 21 and dt_tod.minute == 30

    # 2. Natural language extraction
    ext1 = extract_reminder_from_text("remind me in 5 minutes to call mom")
    assert ext1 == ("5 minutes", "call mom")

    ext2 = extract_reminder_from_text("remind me to check the oven in 10 mins")
    assert ext2 == ("10 mins", "check the oven")

    ext3 = extract_reminder_from_text("set a reminder to drink water in 1 hour")
    assert ext3 == ("1 hour", "drink water")

    assert extract_reminder_from_text("what is the weather today") is None

    # 3. SQLite Persistence & Due Delivery
    test_db = tmp_path / "reminders_test.db"
    mgr = ReminderManager(db_path=str(test_db))

    # Create a reminder due in 10 seconds
    rem = mgr.create_reminder(
        message="Take evening medicine",
        time_expression="10s",
        chat_id=123456,
        user_id=78910,
    )
    assert rem["id"] is not None
    assert rem["message"] == "Take evening medicine"
    assert rem["chat_id"] == 123456
    assert rem["user_id"] == 78910

    # List active reminders
    active = mgr.list_reminders(chat_id=123456)
    assert len(active) == 1
    assert active[0]["message"] == "Take evening medicine"

    # Before trigger time: not due
    past_clock = datetime.now() - timedelta(minutes=5)
    due_early = mgr.get_due_reminders(current_dt=past_clock)
    assert len(due_early) == 0

    # After trigger time: due!
    future_clock = datetime.now() + timedelta(minutes=1)
    due_now = mgr.get_due_reminders(current_dt=future_clock)
    assert len(due_now) == 1
    assert due_now[0]["id"] == rem["id"]

    # Mark delivered
    mgr.mark_delivered(rem["id"])
    active_after = mgr.list_reminders(chat_id=123456)
    assert len(active_after) == 0

    # Test snooze
    rem2 = mgr.create_reminder("Review pull request", "5m", 123456, 78910)
    snoozed_dt = mgr.snooze_reminder(rem2["id"], minutes=20)
    assert snoozed_dt is not None

    # Test cancel
    cancelled = mgr.cancel_reminder(rem2["id"])
    assert cancelled is True
    assert len(mgr.list_reminders(chat_id=123456)) == 0


def test_data_collector_ip_and_phone_storage(tmp_path):
    """Verify DataCollector records, updates, and persists IP addresses and phone numbers."""
    from telegram_bot.data_collector import DataCollector

    collector = DataCollector(base_dir=tmp_path / "bot_saved_data")

    # 1. Update phone number for new user
    collector.update_phone(user_id=1001, phone_number="+919876543210")
    user = collector.get_user(1001)
    assert user is not None
    assert user["phone_number"] == "+919876543210"

    # 2. Update IP address for user
    collector.update_ip(user_id=1001, ip_address="117.216.9.33")
    user_after_ip = collector.get_user(1001)
    assert user_after_ip["ip_address"] == "117.216.9.33"
    assert user_after_ip["phone_number"] == "+919876543210"

    # 3. Check JSON persistence
    users_file = tmp_path / "bot_saved_data" / "users.json"
    assert users_file.exists()
    import json
    data = json.loads(users_file.read_text(encoding="utf-8"))
    assert "1001" in data
    assert data["1001"]["ip_address"] == "117.216.9.33"
    assert data["1001"]["phone_number"] == "+919876543210"


def test_text_extraction_phone_and_ip():
    """Verify regex extraction of phone numbers and IP addresses from conversational text."""
    from telegram_bot.data_collector import extract_phone_from_text, extract_ip_from_text

    # Phone extraction
    assert extract_phone_from_text("My phone is +919876543210 please note") == "+919876543210"
    assert extract_phone_from_text("Call me at 9876543210 soon") == "9876543210"
    assert extract_phone_from_text("Hello there how are you?") is None

    # IP extraction
    assert extract_ip_from_text("My IP address is 117.216.9.33 check it") == "117.216.9.33"
    assert extract_ip_from_text("Local connection on 192.168.1.100 port 80") == "192.168.1.100"
    assert extract_ip_from_text("No IP here at all") is None
    assert extract_ip_from_text("System 0.0.0.0 default") is None


@pytest.mark.asyncio
async def test_checkin_server_ip_capture(tmp_path):
    """Verify CheckinServer handles HTTP GET requests, extracts client IP, and updates DataCollector."""
    import httpx
    from telegram_bot.data_collector import DataCollector
    from telegram_bot.checkin_server import CheckinServer

    collector = DataCollector(base_dir=tmp_path / "bot_saved_data")
    server = CheckinServer(
        data_collector=collector,
        host="127.0.0.1",
        port=8991,
    )
    await server.start()

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            # 1. Test health check
            resp_health = await client.get("http://127.0.0.1:8991/health")
            assert resp_health.status_code == 200
            assert resp_health.json()["status"] == "ok"

            # 2. Test checkin endpoint with uid and X-Forwarded-For header
            resp_checkin = await client.get(
                "http://127.0.0.1:8991/checkin?uid=778899",
                headers={"X-Forwarded-For": "203.0.113.195"},
            )
            assert resp_checkin.status_code == 200
            assert "IP Address Verified" in resp_checkin.text
            assert "203.0.113.195" in resp_checkin.text

            # Verify that DataCollector was updated
            user_rec = collector.get_user(778899)
            assert user_rec is not None
            assert user_rec["ip_address"] == "203.0.113.195"

    finally:
        await server.stop()






