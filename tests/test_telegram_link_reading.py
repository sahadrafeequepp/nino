"""Unit tests for opening, reading, and understanding web links and online content."""

import io
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Mock telegram if not installed in local test environment
class MockInlineKeyboardButton:
    def __init__(self, text, callback_data=None):
        self.text = text
        self.callback_data = callback_data

class MockInlineKeyboardMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard

try:
    import telegram
    from telegram.ext import filters
except ImportError:
    mock_telegram = MagicMock()
    mock_filters = MagicMock()
    class DummyBaseFilter:
        pass
    mock_filters.BaseFilter = DummyBaseFilter
    mock_telegram.constants.ParseMode.MARKDOWN = "Markdown"
    mock_telegram.BotCommand = lambda command, description="": MagicMock(command=command, description=description)
    mock_telegram.InlineKeyboardButton = MockInlineKeyboardButton
    mock_telegram.InlineKeyboardMarkup = MockInlineKeyboardMarkup
    sys.modules["telegram"] = mock_telegram
    sys.modules["telegram.ext"] = MagicMock()
    sys.modules["telegram.ext"].filters = mock_filters
    sys.modules["telegram.constants"] = mock_telegram.constants
    sys.modules["telegram.request"] = MagicMock()

from telegram_bot import bot
from telegram_bot.link_inspector import LinkInspector
from telegram_bot.media_processor import MediaProcessor


class InstantAwaitable:
    def __init__(self, val=None):
        self.val = val

    def __await__(self):
        if False:
            yield
        return self.val


def sync_run(coro):
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


def test_extract_urls_extended():
    """Verify URL extractor handles https, http, www prefixes, and strips surrounding punctuation."""
    inspector = LinkInspector()

    text1 = "Check this out: https://example.com/article, and also http://test.org/page."
    urls1 = inspector.extract_urls(text1)
    assert "https://example.com/article" in urls1
    assert "http://test.org/page" in urls1

    # www prefix without scheme
    text2 = "Visit www.wikipedia.org/wiki/Python for details."
    urls2 = inspector.extract_urls(text2)
    assert "https://www.wikipedia.org/wiki/Python" in urls2

    # Trailing punctuation in parentheses
    text3 = "Documentation at (https://docs.python.org/3/library/index.html)!"
    urls3 = inspector.extract_urls(text3)
    assert "https://docs.python.org/3/library/index.html" in urls3


@pytest.mark.asyncio
async def test_extract_url_html_webpage():
    """Verify HTML webpage parser extracts titles, meta descriptions, and clean body text."""
    processor = MediaProcessor()

    html = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Nino AI Assistant</title>
            <meta name="description" content="Local private AI companion running on your own PC.">
            <style>body { font-family: sans-serif; }</style>
        </head>
        <body>
            <header><nav><a href="/home">Home</a></nav></header>
            <main>
                <h1>Welcome to Nino</h1>
                <p>Nino is designed from the ground up for 100% privacy and local-first execution.</p>
                <h2>Core Features</h2>
                <ul>
                    <li>Local Ollama LLM integration</li>
                    <li>In-memory speech synthesis and voice notes</li>
                </ul>
            </main>
            <footer><p>Copyright 2026</p></footer>
            <script>console.log("analytics");</script>
        </body>
    </html>
    """

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html
    mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        content = await processor.extract_url("https://example.com/nino")
        assert "Nino AI Assistant" in content
        assert "Local private AI companion running on your own PC." in content
        assert "Welcome to Nino" in content
        assert "Local Ollama LLM integration" in content
        assert "analytics" not in content
        assert "font-family" not in content


@pytest.mark.asyncio
async def test_extract_url_online_pdf(monkeypatch):
    """Verify that a URL pointing to a PDF document routes directly through PDF extractor."""
    processor = MediaProcessor()

    # Mock extract_pdf
    monkeypatch.setattr(processor, "extract_pdf", lambda pdf_bytes, filename: f"=== Extracted PDF: {filename} ===\nSample PDF contents.")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"%PDF-1.4 sample content"
    mock_resp.headers = {"content-type": "application/pdf"}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        content = await processor.extract_url("https://example.com/downloads/whitepaper.pdf")
        assert "Extracted PDF: whitepaper.pdf" in content
        assert "Sample PDF contents." in content


@pytest.mark.asyncio
async def test_extract_url_json_api():
    """Verify that a URL returning JSON API content is cleanly formatted."""
    processor = MediaProcessor()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json = MagicMock(return_value={"status": "online", "version": "2.4.0", "active_nodes": 4})
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        content = await processor.extract_url("https://api.example.com/health")
        assert "=== JSON API Response:" in content
        assert '"version": "2.4.0"' in content
        assert '"active_nodes": 4' in content


def test_handle_text_url_with_question(monkeypatch):
    """Verify handle_text extracts URL, scans it, sets media context, and formats question prompt."""
    user_id = 999888
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", user_id)
    monkeypatch.setattr(bot, "data_collector", MagicMock())
    monkeypatch.setattr(bot, "security_scanner", MagicMock())
    monkeypatch.setattr(bot, "media_processor", MagicMock())

    # Mock security scan result
    mock_scan = MagicMock()
    mock_scan.is_safe = True
    bot.security_scanner.scan_url = MagicMock(return_value=InstantAwaitable(mock_scan))
    bot.media_processor.extract_url = MagicMock(return_value=InstantAwaitable("=== Webpage: Test News ===\nArticle text about renewable energy."))

    captured_prompts = []
    async def mock_chat(prompt, **kwargs):
        captured_prompts.append(prompt)
        return "The article discusses solar and wind power.", "ollama"

    monkeypatch.setattr(bot.agent, "chat", mock_chat)

    mock_update = MagicMock()
    mock_user = MagicMock()
    mock_user.id = user_id
    mock_update.effective_user = mock_user
    mock_update.effective_chat.id = 1234

    # Message contains URL and question
    mock_msg = MagicMock()
    mock_msg.text = "What is this article about? https://example.com/energy-news"
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.bot.send_chat_action = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.handle_text(mock_update, mock_context))

    assert len(captured_prompts) == 1
    assert "Web Content Extracted from Shared Link(s)" in captured_prompts[0]
    assert "What is this article about?" in captured_prompts[0]
    assert "Article text about renewable energy." in captured_prompts[0]
    assert mock_msg.reply_text.called

    # Verify media context was stored for follow-up questions
    ctx = bot.agent.get_media_context(user_id)
    assert ctx is not None
    assert ctx["type"] == "url"
    assert ctx["filename_or_url"] == "https://example.com/energy-news"


def test_handle_text_url_without_question(monkeypatch):
    """Verify handle_text without a question formats an intelligent structured review prompt."""
    user_id = 777666
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", user_id)
    monkeypatch.setattr(bot, "data_collector", MagicMock())
    monkeypatch.setattr(bot, "security_scanner", MagicMock())
    monkeypatch.setattr(bot, "media_processor", MagicMock())

    mock_scan = MagicMock()
    mock_scan.is_safe = True
    bot.security_scanner.scan_url = MagicMock(return_value=InstantAwaitable(mock_scan))
    bot.media_processor.extract_url = MagicMock(return_value=InstantAwaitable("=== Webpage: Python 3.13 Released ===\nNew features include free-threaded CPython."))

    captured_prompts = []
    async def mock_chat(prompt, **kwargs):
        captured_prompts.append(prompt)
        return "Here is a breakdown of Python 3.13...", "ollama"

    monkeypatch.setattr(bot.agent, "chat", mock_chat)

    mock_update = MagicMock()
    mock_user = MagicMock()
    mock_user.id = user_id
    mock_update.effective_user = mock_user
    mock_update.effective_chat.id = 5678

    mock_msg = MagicMock()
    # Message contains ONLY a URL
    mock_msg.text = "https://python.org/news/313"
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.bot.send_chat_action = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.handle_text(mock_update, mock_context))

    assert len(captured_prompts) == 1
    assert "without a specific question" in captured_prompts[0]
    assert "Key Takeaways & Summary" in captured_prompts[0]
    assert "Title & Subject" in captured_prompts[0]
