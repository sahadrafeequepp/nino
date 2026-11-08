"""Universal File Reader & Antivirus Validator for Nino Telegram Bot.

Supports:
- Antivirus scan on every file before parsing (blocking executables, macros, and PDF exploits).
- PDFs (.pdf) extraction via pypdf.
- Word documents (.docx) extraction via standard library zipfile + XML parser.
- Plain text, markdown, code, and config files (.txt, .md, .py, .json, .csv, .log, .yaml, etc.).
- Image & Video detection and routing.
"""

import io
import json
import logging
from pathlib import Path
import re
from typing import Any, Optional
import xml.etree.ElementTree as ET
import zipfile

from nino.security.antivirus import ScanResult, antivirus_scanner

logger = logging.getLogger("nino.telegram.file_reader")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}
TEXT_CODE_EXTENSIONS = {
    ".txt", ".md", ".py", ".json", ".csv", ".tsv", ".log",
    ".yaml", ".yml", ".html", ".htm", ".css", ".js", ".ts",
    ".jsx", ".tsx", ".sh", ".xml", ".sql", ".ini", ".conf",
    ".env", ".toml", ".rst", ".c", ".cpp", ".h", ".rs", ".go",
}


class FileInspectionResult:
    """Represents the outcome of scanning and reading an uploaded file."""

    def __init__(
        self,
        is_safe: bool,
        file_type: str,  # 'text', 'pdf', 'docx', 'image', 'video', 'audio', 'blocked', 'unknown'
        text_content: str = "",
        image_bytes: Optional[bytes] = None,
        video_bytes: Optional[bytes] = None,
        error_message: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.is_safe = is_safe
        self.file_type = file_type
        self.text_content = text_content
        self.image_bytes = image_bytes
        self.video_bytes = video_bytes
        self.error_message = error_message
        self.metadata = metadata or {}


def inspect_and_read_file(
    file_bytes: bytes,
    filename: str,
    mime_type: Optional[str] = None,
) -> FileInspectionResult:
    """Pre-scan file with Antivirus and parse content based on format."""
    ext = Path(filename).suffix.lower()
    meta: dict[str, Any] = {"filename": filename, "size_bytes": len(file_bytes), "ext": ext}

    # 1. Antivirus & Exploit Scan
    scan_result: ScanResult = antivirus_scanner.scan_bytes(file_bytes, filename=filename)
    if not scan_result.is_clean:
        logger.warning("Antivirus blocked file %s: %s", filename, scan_result.threat_name)
        return FileInspectionResult(
            is_safe=False,
            file_type="blocked",
            error_message=(
                f"🛡️ *Antivirus Alert*: The file `{filename}` was blocked by security filters!\n"
                f"• Threat: `{scan_result.threat_name}`\n"
                f"• Details: {scan_result.details}"
            ),
            metadata=meta,
        )

    # 2. Image files
    if ext in IMAGE_EXTENSIONS or (mime_type and mime_type.startswith("image/")):
        return FileInspectionResult(
            is_safe=True,
            file_type="image",
            image_bytes=file_bytes,
            text_content=f"[Image file '{filename}' ({len(file_bytes)} bytes)]",
            metadata=meta,
        )

    # 3. Video files
    if ext in VIDEO_EXTENSIONS or (mime_type and mime_type.startswith("video/")):
        return FileInspectionResult(
            is_safe=True,
            file_type="video",
            video_bytes=file_bytes,
            text_content=f"[Video file '{filename}' ({len(file_bytes)} bytes)]",
            metadata=meta,
        )

    # 4. PDF documents (.pdf)
    if ext == ".pdf" or file_bytes.startswith(b"%PDF"):
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            num_pages = len(reader.pages)
            meta["pages"] = num_pages

            pages_text = []
            for i, page in enumerate(reader.pages[:20]):  # Inspect up to first 20 pages
                extracted = page.extract_text() or ""
                if extracted.strip():
                    pages_text.append(f"--- Page {i + 1} ---\n{extracted.strip()}")

            full_text = "\n\n".join(pages_text).strip()
            if not full_text:
                full_text = f"[PDF document '{filename}' contains {num_pages} pages, but contains mostly scanned images or non-selectable text.]"

            if len(full_text) > 20_000:
                full_text = full_text[:20_000] + "\n\n...[content truncated]"

            return FileInspectionResult(
                is_safe=True,
                file_type="pdf",
                text_content=full_text,
                metadata=meta,
            )
        except Exception as e:
            logger.error("Failed to parse PDF %s: %s", filename, e)
            return FileInspectionResult(
                is_safe=False,
                file_type="pdf",
                error_message=f"⚠️ Could not extract text from PDF `{filename}`: {e}",
                metadata=meta,
            )

    # 5. Microsoft Word Documents (.docx)
    if ext == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx_zip:
                xml_content = docx_zip.read("word/document.xml")
                root = ET.fromstring(xml_content)
                # WordprocessingML namespace
                namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                paragraphs = []
                for p in root.findall(".//w:p", namespaces):
                    texts = [t.text for t in p.findall(".//w:t", namespaces) if t.text]
                    if texts:
                        paragraphs.append("".join(texts))

                docx_text = "\n\n".join(paragraphs).strip()
                if not docx_text:
                    docx_text = f"[Word document '{filename}' contains no readable text body.]"

                if len(docx_text) > 20_000:
                    docx_text = docx_text[:20_000] + "\n\n...[content truncated]"

                return FileInspectionResult(
                    is_safe=True,
                    file_type="docx",
                    text_content=docx_text,
                    metadata=meta,
                )
        except Exception as e:
            logger.error("Failed to parse DOCX %s: %s", filename, e)
            return FileInspectionResult(
                is_safe=False,
                file_type="docx",
                error_message=f"⚠️ Could not parse Word document `{filename}`: {e}",
                metadata=meta,
            )

    # 6. Plain text, code, JSON, CSV, markdown, etc.
    if ext in TEXT_CODE_EXTENSIONS or (mime_type and (mime_type.startswith("text/") or "json" in mime_type)):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as e:
                return FileInspectionResult(
                    is_safe=False,
                    file_type="unknown",
                    error_message=f"⚠️ Could not decode `{filename}` as text: {e}",
                    metadata=meta,
                )

        if len(text) > 20_000:
            text = text[:20_000] + "\n\n...[content truncated]"

        return FileInspectionResult(
            is_safe=True,
            file_type="text",
            text_content=text,
            metadata=meta,
        )

    # 7. Fallback attempt: if it looks like UTF-8 plain text despite extension
    try:
        sample = file_bytes[:4096].decode("utf-8")
        # Check printable character ratio
        printable = sum(1 for c in sample if c.isprintable() or c in "\r\n\t")
        if len(sample) > 0 and (printable / len(sample)) > 0.85:
            text = file_bytes.decode("utf-8", errors="replace")
            if len(text) > 20_000:
                text = text[:20_000] + "\n\n...[content truncated]"
            return FileInspectionResult(
                is_safe=True,
                file_type="text",
                text_content=text,
                metadata=meta,
            )
    except Exception:
        pass

    return FileInspectionResult(
        is_safe=True,
        file_type="unknown",
        text_content=f"[Binary file '{filename}' ({len(file_bytes)} bytes) saved to audit archive]",
        metadata=meta,
    )
