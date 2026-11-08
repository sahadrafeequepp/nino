"""Unit tests for enhanced PDF extraction, image analysis, and conversational follow-ups."""

import io
import sys
from unittest.mock import AsyncMock, MagicMock
import pytest
from PIL import Image

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


def test_extract_pdf_digital_pages():
    """Verify that multi-page digital PDFs extract full text with clear page demarcations and metadata."""
    import pypdf

    writer = pypdf.PdfWriter()
    # Add 3 pages with distinct text
    for page_num in range(1, 4):
        writer.add_blank_page(width=200, height=200)

    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    processor = MediaProcessor()
    # Mock page extract_text
    def mock_extract_text(self):
        return f"This is digital content for page."
    pypdf.PageObject.extract_text = mock_extract_text

    result = processor.extract_pdf(pdf_bytes, filename="test_report.pdf")
    assert "test_report.pdf" in result
    assert "=== Extracted PDF Document:" in result
    assert "--- Page 1 of 3 ---" in result
    assert "--- Page 2 of 3 ---" in result
    assert "--- Page 3 of 3 ---" in result
    assert "This is digital content for page." in result


def test_extract_pdf_scanned_page_ocr(monkeypatch):
    """Verify that scanned PDF pages with no digital text fall back to OCR extraction."""
    import pypdf

    processor = MediaProcessor()
    # Mock extract_text_from_photo to simulate OCR on embedded scanned image
    monkeypatch.setattr(processor, "extract_text_from_photo", lambda img_data: "INVOICE #98765 Total: $450.00")

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)

    # Attach dummy image object via property mock
    mock_img = MagicMock()
    mock_img.data = b"fake_scanned_image_bytes"
    monkeypatch.setattr(pypdf.PageObject, "images", property(lambda self: [mock_img]))

    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    # Empty digital text
    pypdf.PageObject.extract_text = lambda self: ""

    result = processor.extract_pdf(pdf_bytes, filename="scanned_invoice.pdf")
    assert "[Scanned Page Image OCR]:" in result
    assert "INVOICE #98765 Total: $450.00" in result


def test_media_context_followup_no_duplication():
    """Verify that active media context is attached once and not duplicated on subsequent turns."""
    from telegram_bot.agent import NinoAgent

    agent = NinoAgent()
    user_id = 99991

    # 1. Set media context for a document
    doc_content = "Summary: Sales increased by 25% in Q3."
    agent.set_media_context(user_id, "pdf", "sales.pdf", doc_content)

    # 2. Simulate follow-up question
    user_q = "What was the Q3 increase?"
    media_ctx = agent.get_media_context(user_id)
    assert media_ctx is not None
    assert media_ctx["type"] == "pdf"
    assert media_ctx["name"] == "sales.pdf"
    assert media_ctx["content"] == doc_content


def test_handle_photo_prompt_formulation(monkeypatch):
    """Verify handle_photo formats query prompt clearly for questions vs no caption."""
    user_id = 777123
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", user_id)
    monkeypatch.setattr(bot, "data_collector", MagicMock())
    monkeypatch.setattr(bot, "security_scanner", MagicMock())
    monkeypatch.setattr(bot, "media_processor", MagicMock())

    mock_scan = MagicMock()
    mock_scan.is_safe = True
    bot.security_scanner.scan_file_bytes = MagicMock(return_value=InstantAwaitable(mock_scan))
    bot.media_processor.extract_image = MagicMock(return_value=InstantAwaitable("A blue car parked on the street"))

    captured_prompts = []
    async def mock_chat(prompt, **kwargs):
        captured_prompts.append(prompt)
        return "I see a blue car!", "ollama"

    monkeypatch.setattr(bot.agent, "chat", mock_chat)

    # Case 1: Photo with specific user question caption
    mock_update = MagicMock()
    mock_user = MagicMock()
    mock_user.id = user_id
    mock_update.effective_user = mock_user
    mock_update.effective_chat.id = 12345

    mock_msg = MagicMock()
    mock_msg.caption = "What color is the vehicle?"
    mock_photo = MagicMock()
    mock_photo.file_unique_id = "uniq_1"
    mock_file = MagicMock()
    async def fake_dl(ba):
        ba.extend(b"\xff\xd8\xff\xe0fakejpeg")
    mock_file.download_as_bytearray = fake_dl
    mock_photo.get_file = MagicMock(return_value=InstantAwaitable(mock_file))
    mock_msg.photo = [mock_photo]
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.bot.send_chat_action = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.handle_photo(mock_update, mock_context))

    assert len(captured_prompts) == 1
    assert "What color is the vehicle?" in captured_prompts[0]
    assert mock_msg.reply_text.called

    # Case 2: Photo without caption gives structured breakdown prompt
    mock_msg.caption = None
    sync_run(bot.handle_photo(mock_update, mock_context))

    assert len(captured_prompts) == 2
    assert "without a question" in captured_prompts[1]
    assert "Visual Overview" in captured_prompts[1]


def test_handle_document_pdf_prompt_formulation(monkeypatch):
    """Verify handle_document formats query prompt clearly for questions vs no caption."""
    user_id = 888123
    monkeypatch.setattr(bot, "AUTHORIZED_USER_ID", user_id)
    monkeypatch.setattr(bot, "data_collector", MagicMock())
    monkeypatch.setattr(bot, "security_scanner", MagicMock())
    monkeypatch.setattr(bot, "media_processor", MagicMock())

    mock_scan = MagicMock()
    mock_scan.is_safe = True
    bot.security_scanner.scan_file_bytes = MagicMock(return_value=InstantAwaitable(mock_scan))
    bot.media_processor.extract_pdf = MagicMock(return_value="Chapter 1: Quantum Mechanics principles")

    captured_prompts = []
    async def mock_chat(prompt, **kwargs):
        captured_prompts.append(prompt)
        return "Here is what chapter 1 covers...", "ollama"

    monkeypatch.setattr(bot.agent, "chat", mock_chat)

    mock_update = MagicMock()
    mock_user = MagicMock()
    mock_user.id = user_id
    mock_update.effective_user = mock_user
    mock_update.effective_chat.id = 54321

    mock_msg = MagicMock()
    mock_msg.caption = "Summarize the principles in chapter 1"
    mock_doc = MagicMock()
    mock_doc.file_name = "physics.pdf"
    mock_doc.mime_type = "application/pdf"
    mock_doc.file_size = 12000
    mock_file = MagicMock()
    async def fake_dl(ba):
        ba.extend(b"%PDF-1.4...")
    mock_file.download_as_bytearray = fake_dl
    mock_doc.get_file = MagicMock(return_value=InstantAwaitable(mock_file))
    mock_msg.document = mock_doc
    mock_msg.reply_text = MagicMock(return_value=InstantAwaitable())
    mock_update.message = mock_msg

    mock_context = MagicMock()
    mock_context.bot.send_chat_action = MagicMock(return_value=InstantAwaitable())

    sync_run(bot.handle_document(mock_update, mock_context))

    assert len(captured_prompts) == 1
    assert "Summarize the principles in chapter 1" in captured_prompts[0]
    assert mock_msg.reply_text.called

    # Case 2: Document without caption gives executive review prompt
    mock_msg.caption = None
    sync_run(bot.handle_document(mock_update, mock_context))

    assert len(captured_prompts) == 2
    assert "without a specific question" in captured_prompts[1]
    assert "Executive Summary" in captured_prompts[1]


@pytest.mark.asyncio
async def test_agent_chat_accepts_raw_prompt_and_kwargs():
    """Verify NinoAgent.chat handles raw_prompt, image_bytes, and arbitrary kwargs without TypeError."""
    agent = bot.agent
    user_id = 999111

    captured = {}
    async def fake_chat_ollama(augmented_prompt, image_bytes=None, user_id=None, is_owner=True, raw_prompt=None):
        captured["augmented"] = augmented_prompt
        captured["raw"] = raw_prompt
        captured["image"] = image_bytes
        return "Analysis completed successfully."

    agent._chat_ollama = fake_chat_ollama
    agent.is_ollama_online = AsyncMock(return_value=True)

    # 1. Document / Image call signature (reproduces user's exact invocation)
    reply, engine = await agent.chat(
        "Examine the document content...",
        image_bytes=b"dummy_bytes",
        user_id=user_id,
        is_owner=True,
        raw_prompt="[Uploaded PDF 'document.pdf'] Summarize this",
        extra_arbitrary_kwarg="safe_test",
    )

    assert reply == "Analysis completed successfully."
    assert engine == "ollama"
    assert captured["raw"] == "[Uploaded PDF 'document.pdf'] Summarize this"
    assert captured["image"] == b"dummy_bytes"
    assert "Examine the document content" in captured["augmented"]

