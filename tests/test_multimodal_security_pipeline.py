"""Comprehensive test suite for Nino's Multi-Modal Ingestion, Security Pipeline, and Media Session Memory."""

import asyncio
from datetime import datetime
import io
from pathlib import Path
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

class MockBotCommand:
    def __init__(self, command, description=""):
        self.command = command
        self.description = description

# Mock telegram if not installed in local test environment
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

from security_scanner import SecurityScanner, SecurityScanResult
from media_processor import MediaProcessor
from telegram_bot.agent import NinoAgent, MediaSessionManager
from telegram_bot.bot import is_authorized


# =============================================================================
# 1. SecurityScanner Tests
# =============================================================================
@pytest.mark.asyncio
async def test_security_scanner_filename_sanitization():
    """Verify filename sanitization eliminates path traversal artifacts."""
    scanner = SecurityScanner()

    assert scanner.sanitize_filename("../../etc/passwd") == "passwd"
    assert scanner.sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert scanner.sanitize_filename("normal_report.pdf") == "normal_report.pdf"
    assert scanner.sanitize_filename("") == "unnamed_file.bin"


@pytest.mark.asyncio
async def test_security_scanner_blocks_dangerous_extensions():
    """Verify executable extensions are rejected immediately."""
    scanner = SecurityScanner()

    for dangerous in ["malware.exe", "script.bat", "dropper.scr", "payload.sh", "virus.vbs", "shell.ps1"]:
        res = await scanner.scan_file_bytes(b"some payload", filename=dangerous)
        assert res.is_safe is False
        assert "Blocked dangerous" in res.reason


@pytest.mark.asyncio
async def test_security_scanner_blocks_executable_magic_bytes():
    """Verify masqueraded executables (e.g. PE MZ inside .jpg) are detected and blocked."""
    scanner = SecurityScanner()

    # Windows PE executable disguised as JPG
    fake_jpg = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 100
    res = await scanner.scan_file_bytes(fake_jpg, filename="innocent_photo.jpg")
    assert res.is_safe is False
    assert "Executable masquerade" in res.reason

    # Linux ELF executable disguised as PDF
    fake_pdf = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 100
    res2 = await scanner.scan_file_bytes(fake_pdf, filename="invoice.pdf")
    assert res2.is_safe is False
    assert "Executable masquerade" in res2.reason


@pytest.mark.asyncio
async def test_security_scanner_blocks_pdf_exploits():
    """Verify PDF files containing exploit structures (/Launch, /JavaScript) are rejected."""
    scanner = SecurityScanner()

    malicious_pdf = b"%PDF-1.4\n1 0 obj\n<< /JavaScript (app.alert('pwned')) >>\nendobj\n%%EOF"
    res = await scanner.scan_file_bytes(malicious_pdf, filename="exploit.pdf")
    assert res.is_safe is False
    assert "exploit structure detected" in res.reason


@pytest.mark.asyncio
async def test_security_scanner_sha256_caching():
    """Verify SHA-256 hashes are cached to avoid redundant network inspections."""
    scanner = SecurityScanner(api_key="mock_key")
    safe_data = b"Hello, this is a clean document text payload."
    filename = "doc.txt"

    # First scan
    res1 = await scanner.scan_file_bytes(safe_data, filename=filename)
    assert res1.is_safe is True
    assert res1.sha256 in scanner.hash_cache

    # Second scan must hit hash_cache
    res2 = await scanner.scan_file_bytes(safe_data, filename=filename)
    assert res2 is res1


@pytest.mark.asyncio
async def test_security_scanner_url_ssrf_blocking():
    """Verify private IPs and loopback URLs are blocked by SSRF filter."""
    scanner = SecurityScanner()

    # Loopback
    res1 = await scanner.scan_url("http://127.0.0.1:8000/secret")
    assert res1.is_safe is False
    assert "SSRF" in res1.reason

    # Cloud metadata
    res2 = await scanner.scan_url("http://169.254.169.254/latest/meta-data/")
    assert res2.is_safe is False

    # Disallowed scheme
    res3 = await scanner.scan_url("file:///etc/passwd")
    assert res3.is_safe is False
    assert "Disallowed URL scheme" in res3.reason


@pytest.mark.asyncio
async def test_security_scanner_virustotal_url_flagged():
    """Verify VirusTotal flagged URLs trigger immediate rejection."""
    scanner = SecurityScanner(api_key="real_looking_key_1234567890")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "attributes": {
                "last_analysis_stats": {
                    "malicious": 5,
                    "suspicious": 2,
                    "harmless": 40,
                }
            }
        }
    }

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        res = await scanner.scan_url("https://phishing-scam-test-domain.xyz")
        assert res.is_safe is False
        assert res.positives == 7
        assert "VirusTotal detected" in res.reason


# =============================================================================
# 2. MediaProcessor Tests
# =============================================================================
def test_media_processor_pdf_extraction():
    """Verify PDF extractor extracts text and handles page numbers."""
    processor = MediaProcessor()

    # Generate a lightweight in-memory PDF using pypdf
    import pypdf

    writer = pypdf.PdfWriter()
    # Add a blank page with some text if possible, or test pypdf reader
    buf = io.BytesIO()
    writer.add_blank_page(width=100, height=100)
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    result = processor.extract_pdf(pdf_bytes, filename="sample.pdf")
    assert "sample.pdf" in result


def test_media_processor_corrupted_pdf():
    """Verify malformed PDFs are handled gracefully without uncaught exceptions."""
    processor = MediaProcessor()
    corrupted_bytes = b"Not a real PDF header or structure"
    result = processor.extract_pdf(corrupted_bytes, filename="bad.pdf")
    assert "Malformed" in result or "Error" in result


@pytest.mark.asyncio
async def test_media_processor_url_scraping():
    """Verify web page body text scraper strips script, nav, and styles."""
    processor = MediaProcessor()

    html_content = """
    <html>
        <head><title>Test Article</title><style>.hidden{display:none;}</style></head>
        <body>
            <nav><a href="/">Home</a></nav>
            <script>console.log("tracking code");</script>
            <main>
                <h1>Nino System Architecture</h1>
                <p>Nino is a local-first personal companion and agent system built with high security.</p>
            </main>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html_content
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        result = await processor.extract_url("https://example.com/test-article")
        assert "Test Article" in result
        assert "Nino System Architecture" in result
        assert "console.log" not in result
        assert "tracking code" not in result


def test_media_processor_youtube_id_extraction():
    """Verify YouTube ID parser handles standard, short, and embed URLs."""
    processor = MediaProcessor()

    assert processor._extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert processor._extract_youtube_id("https://youtu.be/dQw4w9WgXcQ?t=42") == "dQw4w9WgXcQ"
    assert processor._extract_youtube_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert processor._extract_youtube_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_media_processor_tesseract_ocr():
    """Verify lightweight Tesseract OCR function is callable and gracefully handles images."""
    import tempfile
    from media_processor import extract_text_from_photo
    from PIL import Image

    # Create dummy in-memory image
    img = Image.new("RGB", (100, 30), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    processor = MediaProcessor()
    # Mock pytesseract.image_to_string
    with patch("pytesseract.image_to_string", return_value="Total: $42.50\nThank you!"):
        res = processor.extract_text_from_photo(png_bytes)
        assert "Total: $42.50" in res

        # Standalone function
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(png_bytes)
            tmp_path = tf.name
        try:
            res2 = extract_text_from_photo(tmp_path)
            assert "Total: $42.50" in res2
        finally:
            Path(tmp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_media_processor_keep_alive_auto_unload():
    """Verify Ollama vision query passes keep_alive='1m' to avoid dual-model memory contention."""
    processor = MediaProcessor(ollama_vision_model="moondream", keep_alive="1m")
    assert processor.keep_alive == "1m"

    captured_call = {}

    async def mock_generate(*args, **kwargs):
        captured_call.update(kwargs)
        return {"response": "A receipt on a wooden desk."}

    processor._ollama_client = MagicMock()
    processor._ollama_client.generate = AsyncMock(side_effect=mock_generate)

    # Mock tags to return moondream installed
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "moondream:latest"}]}

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        res = await processor._extract_image_ollama(b"fake_image_bytes", "Describe")
        assert "moondream" in res
        assert captured_call.get("keep_alive") == "1m"
        assert captured_call.get("model") == "moondream:latest"



# =============================================================================
# 3. Media Session Memory & Follow-up QA Tests
# =============================================================================
def test_media_session_manager():
    """Verify MediaSessionManager stores, retrieves, and clears active media context."""
    mgr = MediaSessionManager()
    user_id = 12345

    # Store
    mgr.set_context(
        user_id=user_id,
        media_type="pdf",
        filename_or_url="finance_report.pdf",
        extracted_content="Q3 Revenue was $4.2M, up 15%. Expenses were $3.1M.",
    )

    ctx = mgr.get_context(user_id)
    assert ctx is not None
    assert ctx["type"] == "pdf"
    assert ctx["filename_or_url"] == "finance_report.pdf"
    assert "Revenue was $4.2M" in ctx["extracted_content"]
    assert isinstance(ctx["timestamp"], datetime)

    # Clear
    assert mgr.clear_context(user_id) is True
    assert mgr.get_context(user_id) is None
    assert mgr.clear_context(user_id) is False


@pytest.mark.asyncio
async def test_agent_media_context_prompt_augmentation():
    """Verify NinoAgent automatically injects media context into follow-up queries."""
    agent = NinoAgent()
    user_id = 99999

    # Set media context
    agent.set_media_context(
        user_id=user_id,
        media_type="image",
        filename_or_url="chart.png",
        extracted_content="A bar chart showing 2026 user growth across 4 quarters.",
    )

    # Mock Ollama chat to capture the augmented prompt
    captured_prompt = None

    async def mock_chat_ollama(prompt, *args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "I see the chart showing 2026 growth."

    agent._chat_ollama = mock_chat_ollama
    agent.is_ollama_online = AsyncMock(return_value=True)

    reply, engine = await agent.chat("What was the highest quarter in that chart?", user_id=user_id)

    assert "Active Media Context (image: chart.png)" in captured_prompt or "recently uploaded image (chart.png)" in captured_prompt
    assert "A bar chart showing 2026 user growth" in captured_prompt
    assert "What was the highest quarter in that chart?" in captured_prompt
    assert engine == "ollama"

    # Now clear media context
    agent.clear_media_context(user_id)
    captured_prompt = None
    await agent.chat("Tell me a joke", user_id=user_id)
    assert "Active Media Context" not in captured_prompt
    assert "Tell me a joke" in captured_prompt


# =============================================================================
# 4. User Gating & Authorization Tests
# =============================================================================
def test_user_gating_authorization():
    """Verify is_authorized strictly permits MY_TELEGRAM_USER_ID and blocks others."""
    from telegram_bot import bot

    bot.AUTHORIZED_USER_ID = 1334848825

    # Authorized user
    assert is_authorized(1334848825) is True

    # Unauthorized intruder
    assert is_authorized(9876543210) is False
    assert is_authorized(None) is False
