"""Multi-modal Ingestion & Content Extraction Pipeline for Nino (100% Local).

Extractors for:
- PDF: Text extraction via pypdf (first 15 pages, max 4,000 characters)
- URLs: Web body scraping via httpx + BeautifulSoup4 (stripped of scripts, nav, headers, footers)
- YouTube URLs: Video transcript extraction via youtube-transcript-api
- Images: Local Ollama vision analysis (llama3.2-vision / llava)
- Video: Audio extraction via ffmpeg/moviepy/av + faster-whisper transcription
"""

import asyncio
import base64
import io
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import ollama
except ImportError:
    ollama = None

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("nino.media.processor")

# Default local Ollama vision models (optimized for low-resource hardware)
DEFAULT_OLLAMA_VISION_MODELS = [
    os.getenv("OLLAMA_VISION_MODEL", "moondream").strip(),
    "moondream:latest",
    "moondream",
    "llama3.2-vision:latest",
    "llama3.2-vision",
    "llava:latest",
    "llava",
    "minicpm-v",
]


class MediaProcessor:
    """Orchestrates content extraction for PDF, image, video, and web media."""

    def __init__(
        self,
        ollama_base_url: Optional[str] = None,
        ollama_vision_model: Optional[str] = None,
        whisper_model: str = "base.en",
        keep_alive: Optional[str] = None,
        **kwargs: Any,  # Absorb legacy kwargs without error
    ):
        raw_host = ollama_base_url or os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_base_url = raw_host.rstrip("/")
        self.ollama_vision_model = (
            ollama_vision_model
            or os.getenv("OLLAMA_VISION_MODEL")
            or "moondream"
        ).strip()
        self.whisper_model = (os.getenv("WHISPER_MODEL") or whisper_model).strip()
        self.keep_alive = (keep_alive or os.getenv("OLLAMA_KEEP_ALIVE") or "1m").strip()
        self._ollama_client = ollama.AsyncClient(host=self.ollama_base_url) if ollama is not None else None
        self._tesseract_configured = False

    # =========================================================================
    # 1. PDF Text Extractor (First 15 pages, max 4,000 characters)
    # =========================================================================
    def extract_pdf(self, pdf_bytes: bytes, filename: str = "document.pdf") -> str:
        """Extract readable plain text and OCR scanned pages from a PDF document using pypdf and Tesseract."""
        try:
            import pypdf
        except ImportError:
            return "[Error: pypdf library is not installed.]"

        try:
            stream = io.BytesIO(pdf_bytes)
            reader = pypdf.PdfReader(stream)

            # Handle password encryption
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    return f"🔒 PDF '{filename}' is password-protected and cannot be read without credentials."

            total_pages = len(reader.pages)
            if total_pages == 0:
                return f"📄 PDF '{filename}' is empty (0 pages)."

            # Read up to the first 50 pages with generous 24,000 character limit
            max_pages = min(50, total_pages)
            extracted_pages: list[str] = []
            total_chars = 0
            MAX_CHARS = 24000
            pages_inspected = 0

            for idx in range(max_pages):
                pages_inspected += 1
                try:
                    page = reader.pages[idx]
                    text = (page.extract_text() or "").strip()

                    # Fallback to OCR if digital text is empty (scanned PDF pages, invoices, lecture slides)
                    if not text and hasattr(page, "images") and page.images:
                        page_ocr_parts = []
                        for img_obj in page.images:
                            try:
                                ocr_res = self.extract_text_from_photo(img_obj.data)
                                if ocr_res and ocr_res.strip():
                                    page_ocr_parts.append(ocr_res.strip())
                            except Exception as img_err:
                                logger.debug("OCR extraction on PDF page image failed: %s", img_err)
                        if page_ocr_parts:
                            text = "[Scanned Page Image OCR]:\n" + "\n".join(page_ocr_parts)

                    if text:
                        extracted_pages.append(f"--- Page {idx + 1} of {total_pages} ---\n{text}")
                        total_chars += len(text)
                        if total_chars >= MAX_CHARS:
                            break
                except Exception as e:
                    logger.warning("Error reading page %d of %s: %s", idx + 1, filename, e)

            if not extracted_pages:
                return (
                    f"📄 PDF '{filename}' contains {total_pages} page(s), but no readable digital text "
                    f"or OCR images could be extracted."
                )

            # Extract document metadata if available
            meta_parts = [f"📄 File: {filename}", f"Pages Inspected: {pages_inspected}/{total_pages}"]
            if getattr(reader, "metadata", None):
                meta = reader.metadata
                if getattr(meta, "title", None):
                    meta_parts.append(f"Title: {meta.title}")
                if getattr(meta, "author", None):
                    meta_parts.append(f"Author: {meta.author}")
                if getattr(meta, "subject", None):
                    meta_parts.append(f"Subject: {meta.subject}")

            header = f"=== Extracted PDF Document: {' | '.join(meta_parts)} ===\n\n"
            full_text = header + "\n\n".join(extracted_pages)

            # Enforce max character limit
            if len(full_text) > MAX_CHARS:
                full_text = full_text[:MAX_CHARS] + f"\n\n[... content truncated to {MAX_CHARS:,} characters ...]"

            return full_text

        except Exception as e:
            logger.error("Failed to extract PDF %s: %s", filename, e)
            return f"⚠️ Malformed or unreadable PDF document '{filename}': {e}"

    # =========================================================================
    # 2. Image Vision & OCR Extractor (Pure Local Ollama + Tesseract OCR)
    # =========================================================================
    def _configure_tesseract(self) -> None:
        """Detect and set tesseract_cmd on Windows if not already in system PATH."""
        self._tesseract_configured = True
        try:
            import pytesseract
            import shutil

            if shutil.which("tesseract"):
                return

            candidates = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info("Configured Tesseract OCR binary path: %s", path)
                    break
        except Exception:
            pass

    def extract_text_from_photo(self, image_bytes_or_path: bytes | str | Path) -> str:
        """Lightweight local OCR that uses almost zero RAM (<100MB) via Tesseract."""
        try:
            import pytesseract
            from PIL import Image

            if not getattr(self, "_tesseract_configured", False):
                self._configure_tesseract()
        except ImportError:
            return ""

        try:
            if isinstance(image_bytes_or_path, (str, Path)):
                with Image.open(image_bytes_or_path) as img:
                    return pytesseract.image_to_string(img).strip()
            else:
                with Image.open(io.BytesIO(image_bytes_or_path)) as img:
                    return pytesseract.image_to_string(img).strip()
        except Exception as e:
            logger.debug("Tesseract OCR extraction bypassed: %s", e)
            return ""

    async def extract_image(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
        prompt: Optional[str] = None,
    ) -> str:
        """Run OCR and vision inference 100% locally using Ollama and lightweight Tesseract OCR."""
        # 1. Rapid CPU text extraction via Tesseract (<100MB RAM)
        ocr_text = self.extract_text_from_photo(image_bytes)

        caption = prompt or (
            "Analyze this image in detail:\n"
            "1. Rich Visual Description: Thoroughly describe the scene, objects, people, layout, colors, and context.\n"
            "2. OCR Text Transcript: Verbatim transcript of all readable text, titles, numbers, and labels found in the image."
        )

        ollama_reply = await self._extract_image_ollama(image_bytes, caption)
        if ollama_reply:
            if ocr_text and ocr_text.lower() not in ollama_reply.lower():
                return f"{ollama_reply}\n\n=== Direct OCR Transcript (Tesseract CPU) ===\n{ocr_text}"
            return ollama_reply

        # Fallback to Tesseract OCR if Ollama vision is offline/uninstalled
        if ocr_text:
            return f"=== Lightweight OCR Transcript (Tesseract) ===\n[File: {filename}]\n\n{ocr_text}"

        meta_str = ""
        try:
            from PIL import Image
            with Image.open(io.BytesIO(image_bytes)) as img:
                meta_str = f"Dimensions: {img.width}x{img.height}, Format: {img.format}"
        except Exception:
            pass

        extra = f" ({meta_str})" if meta_str else ""
        return (
            f"[Image '{filename}'{extra} received. Local Ollama vision is offline or no vision model is installed. "
            f"To enable AI image vision, run 'ollama pull {self.ollama_vision_model}'. "
            f"If the image contains text, ensure Tesseract OCR is installed to transcribe it.]"
        )

    async def _extract_image_ollama(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        """Query local Ollama instance for vision inference using AsyncClient.generate with auto-unload keep_alive."""
        try:
            # 1. Discover installed vision models
            installed: list[str] = []
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    tags_resp = await client.get(f"{self.ollama_base_url}/api/tags")
                    if tags_resp.status_code == 200:
                        installed = [m.get("name", "") for m in tags_resp.json().get("models", [])]
            except Exception:
                pass

            chosen_model: Optional[str] = None
            for candidate in [self.ollama_vision_model] + DEFAULT_OLLAMA_VISION_MODELS:
                for inst in installed:
                    if candidate.lower() in inst.lower():
                        chosen_model = inst
                        break
                if chosen_model:
                    break

            if not chosen_model:
                chosen_model = self.ollama_vision_model

            logger.info("Running local vision analysis with Ollama model '%s' (keep_alive=%s)...", chosen_model, self.keep_alive)
            b64_image = base64.b64encode(image_bytes).decode("utf-8")

            # 2. Try official ollama.AsyncClient().generate() with auto-unload keep_alive
            if self._ollama_client is not None:
                try:
                    res = await self._ollama_client.generate(
                        model=chosen_model,
                        prompt=prompt,
                        images=[b64_image],
                        keep_alive=self.keep_alive,
                    )
                    content = res.get("response", "").strip()
                    if content:
                        return f"=== Visual Analysis (via Ollama {chosen_model}) ===\n{content}"
                except Exception as e:
                    logger.debug("ollama.AsyncClient.generate failed (%s); trying direct REST call...", e)

            # 3. Direct REST fallback to Ollama /api/generate with keep_alive
            payload = {
                "model": chosen_model,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {"temperature": 0.2},
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                res = await client.post(f"{self.ollama_base_url}/api/generate", json=payload)
                if res.status_code == 200:
                    content = res.json().get("response", "").strip()
                    if content:
                        return f"=== Visual Analysis (via Ollama {chosen_model}) ===\n{content}"

        except Exception as e:
            logger.warning("Local Ollama vision call failed: %s", e)

        return None

    # =========================================================================
    # 3. Video Transcriber & Audio Extraction (try...finally cleanup)
    # =========================================================================
    async def extract_video(
        self,
        video_bytes_or_path: bytes | str | Path,
        filename: str = "video.mp4",
    ) -> str:
        """Extract video metadata, audio track transcription via faster-whisper."""
        temp_video_path: Optional[Path] = None
        temp_audio_path: Optional[Path] = None

        try:
            # 1. Ensure video resides on disk
            if isinstance(video_bytes_or_path, (str, Path)):
                video_path = Path(video_bytes_or_path).resolve()
            else:
                suffix = Path(filename).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                    tf.write(video_bytes_or_path)
                    temp_video_path = Path(tf.name)
                video_path = temp_video_path

            if not video_path.exists():
                return f"[Error: Video file not found: {filename}]"

            # 2. Extract metadata & keyframe snapshot
            meta_info, keyframe_bytes = self._extract_video_metadata_and_keyframes(video_path)

            # 3. Demux audio track to temporary WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as af:
                temp_audio_path = Path(af.name)

            has_audio = self._extract_audio_track(video_path, temp_audio_path)
            transcript_text = ""

            if has_audio and temp_audio_path.stat().st_size > 1000:
                transcript_text = await self._transcribe_audio_file(temp_audio_path)

            # 4. If keyframe exists and audio was silent, analyze keyframe with vision
            visual_summary = ""
            if keyframe_bytes and (not transcript_text or transcript_text == "[unintelligible audio]"):
                visual_summary = await self.extract_image(
                    keyframe_bytes,
                    filename="keyframe.jpg",
                    prompt="Analyze this keyframe snapshot from the uploaded video. What is being shown?",
                )

            parts = [f"=== Video Analysis: {filename} ==="]
            if meta_info:
                parts.append(f"Metadata: {meta_info}")
            if transcript_text:
                parts.append(f"\n--- Spoken Audio Transcript ---\n{transcript_text}")
            if visual_summary:
                parts.append(f"\n--- Keyframe Visual Context ---\n{visual_summary}")

            return "\n".join(parts)

        finally:
            # Safe cleanup: guarantee immediate deletion of temporary disk files
            if temp_video_path and temp_video_path.exists():
                try:
                    temp_video_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if temp_audio_path and temp_audio_path.exists():
                try:
                    temp_audio_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _extract_video_metadata_and_keyframes(self, video_path: Path) -> tuple[str, Optional[bytes]]:
        """Read video properties and capture representative keyframes using OpenCV."""
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return "Video opened failed", None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_sec = (total_frames / fps) if fps > 0 else 0.0

            meta = f"Duration: {duration_sec:.1f}s | Resolution: {width}x{height} | FPS: {fps:.1f} | Frames: {total_frames}"

            keyframe_bytes = None
            if total_frames > 0:
                target_frame = int(total_frames * 0.25)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if ret and frame is not None:
                    success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if success:
                        keyframe_bytes = buffer.tobytes()

            cap.release()
            return meta, keyframe_bytes
        except Exception as e:
            logger.warning("OpenCV video metadata extraction error: %s", e)
            return "Video metadata unavailable", None

    def _extract_audio_track(self, video_path: Path, output_audio_path: Path) -> bool:
        """Demux audio from video file using PyAV, moviepy, or ffmpeg."""
        try:
            import av

            container = av.open(str(video_path))
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if not audio_stream:
                logger.info("Video %s contains no audio stream.", video_path.name)
                return False

            out_container = av.open(str(output_audio_path), mode="w", format="wav")
            out_stream = out_container.add_stream("pcm_s16le", rate=16000)
            out_stream.channels = 1

            for frame in container.decode(audio_stream):
                frame.pts = None
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

            for packet in out_stream.encode(None):
                out_container.mux(packet)

            out_container.close()
            container.close()
            return True
        except Exception as e:
            logger.debug("PyAV audio demux failed (%s); trying ffmpeg...", e)

        # Fallback to ffmpeg
        try:
            import subprocess

            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(output_audio_path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return res.returncode == 0 and output_audio_path.exists()
        except Exception:
            pass

        return False

    async def _transcribe_audio_file(self, audio_path: Path) -> str:
        """Transcribe an audio file using faster-whisper."""
        try:
            from faster_whisper import WhisperModel

            def _sync_transcribe() -> str:
                model = WhisperModel(self.whisper_model, device="auto", compute_type="int8")
                segments, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
                chunks = [s.text.strip() for s in segments if s.text]
                return " ".join(chunks).strip()

            result = await asyncio.to_thread(_sync_transcribe)
            return result or "[unintelligible audio]"
        except Exception as e:
            logger.warning("Transcription failed on %s: %s", audio_path, e)
            return f"[Transcription error: {e}]"

    # =========================================================================
    # 4. YouTube Transcript Extractor
    # =========================================================================
    def extract_youtube_transcript(self, url: str) -> str:
        """Extract captions and transcript from a YouTube URL."""
        video_id = self._extract_youtube_id(url)
        if not video_id:
            return f"⚠️ Could not identify a valid YouTube Video ID from '{url}'."

        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            lines: list[str] = []
            for item in transcript_list:
                start_sec = int(item.get("start", 0))
                mins = start_sec // 60
                secs = start_sec % 60
                text = item.get("text", "").replace("\n", " ").strip()
                if text:
                    lines.append(f"[{mins:02d}:{secs:02d}] {text}")

            content = "\n".join(lines)
            return f"=== YouTube Transcript: {url} (ID: {video_id}) ===\n\n{content}"

        except Exception as e:
            logger.warning("Failed to retrieve YouTube transcript for %s: %s", video_id, e)
            return f"⚠️ Could not retrieve YouTube transcript for '{url}': {e}"

    @staticmethod
    def _extract_youtube_id(url: str) -> Optional[str]:
        """Extract the 11-character YouTube video ID from various URL formats."""
        parsed = urlparse(url)
        if "youtube.com" in parsed.netloc:
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [None])[0]
            elif parsed.path.startswith(("/embed/", "/v/", "/shorts/")):
                parts = parsed.path.strip("/").split("/")
                if len(parts) >= 2:
                    return parts[1]
        elif "youtu.be" in parsed.netloc:
            parts = parsed.path.strip("/").split("/")
            if parts and parts[0]:
                return parts[0]
        return None

    # =========================================================================
    # 5. Clean URL Body Text Scraper (HTML, Online PDFs, Images, JSON, YouTube)
    # =========================================================================
    async def extract_url(self, url: str, max_chars: int = 20000) -> str:
        """Scrape and parse clean content from a webpage, online PDF, image, JSON API, or YouTube video."""
        clean_url = url.strip()
        if not clean_url.startswith(("http://", "https://")):
            clean_url = f"https://{clean_url}"

        # 1. YouTube video transcript check
        yt_id = self._extract_youtube_id(clean_url)
        if yt_id:
            return self.extract_youtube_transcript(clean_url)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36 NinoAI/2.0"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf,image/*;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(clean_url, headers=headers)
                resp.raise_for_status()

            content_type = ""
            raw_headers = getattr(resp, "headers", None)
            if raw_headers is not None and hasattr(raw_headers, "get"):
                try:
                    raw_ct = raw_headers.get("content-type", "")
                    if isinstance(raw_ct, str):
                        content_type = raw_ct.lower()
                except Exception:
                    content_type = ""

            url_path = urlparse(clean_url).path.lower()

            # 2. Online PDF document check
            raw_content = getattr(resp, "content", b"")
            if isinstance(raw_content, (bytes, bytearray)):
                has_pdf_magic = raw_content.startswith(b"%PDF")
            else:
                has_pdf_magic = False

            if (content_type and "application/pdf" in content_type) or url_path.endswith(".pdf") or has_pdf_magic:
                pdf_filename = Path(url_path).name or "online_document.pdf"
                return self.extract_pdf(raw_content if isinstance(raw_content, bytes) else bytes(raw_content), filename=pdf_filename)

            # 3. Online Image check
            if (content_type and content_type.startswith("image/")) or url_path.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
                img_filename = Path(url_path).name or "online_image.jpg"
                img_bytes = raw_content if isinstance(raw_content, bytes) else bytes(raw_content)
                return await self.extract_image(
                    img_bytes,
                    filename=img_filename,
                    prompt="Describe this online image in detail and transcribe all visible text.",
                )

            # 4. JSON API response check
            if content_type and "application/json" in content_type:
                try:
                    json_data = resp.json()
                    formatted_json = json.dumps(json_data, indent=2, ensure_ascii=False)
                    if len(formatted_json) > max_chars:
                        formatted_json = formatted_json[:max_chars] + f"\n\n[... content truncated to {max_chars:,} characters ...]"
                    return f"=== JSON API Response: {clean_url} ===\n\n{formatted_json}"
                except Exception:
                    pass

            # 5. HTML Webpage extraction
            raw_text = getattr(resp, "text", "")
            html = raw_text if isinstance(raw_text, str) else str(raw_text or "")
            title = "Webpage"
            description = ""

            if BeautifulSoup is not None:
                soup = BeautifulSoup(html, "html.parser")
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()

                # Extract meta description or OpenGraph description
                meta_desc = (
                    soup.find("meta", attrs={"name": "description"})
                    or soup.find("meta", attrs={"property": "og:description"})
                    or soup.find("meta", attrs={"name": "twitter:description"})
                )
                if meta_desc and meta_desc.get("content"):
                    description = meta_desc["content"].strip()

                og_title = soup.find("meta", attrs={"property": "og:title"})
                if og_title and og_title.get("content") and title == "Webpage":
                    title = og_title["content"].strip()

                # Strip script, style, nav, header, footer, aside, noscript, etc.
                for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "svg", "iframe", "form", "button", "dialog"]):
                    tag.decompose()

                text_blocks: list[str] = []
                main_container = soup.find("main") or soup.find("article") or soup.body or soup

                for element in main_container.find_all(["h1", "h2", "h3", "h4", "p", "li", "article", "pre", "code", "blockquote"]):
                    t = element.get_text(strip=True)
                    if t and len(t) > 10:
                        if element.name in ["h1", "h2", "h3", "h4"]:
                            text_blocks.append(f"\n### {t}")
                        elif element.name == "li":
                            text_blocks.append(f"• {t}")
                        else:
                            text_blocks.append(t)

                body_text = "\n\n".join(text_blocks)
                body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()
                if not body_text:
                    body_text = soup.get_text(separator="\n", strip=True)
            else:
                title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else "Webpage"
                meta_match = re.search(r'<meta[^>]+(?:name|property)=["\'](?:description|og:description|twitter:description)["\'][^>]+content=["\'](.*?)["\']', html, re.IGNORECASE)
                if not meta_match:
                    meta_match = re.search(r'<meta[^>]+content=["\'](.*?)["\'][^>]+(?:name|property)=["\'](?:description|og:description|twitter:description)["\']', html, re.IGNORECASE)
                if meta_match:
                    description = meta_match.group(1).strip()

                clean_html = re.sub(r"<(script|style|nav|header|footer|aside|noscript|svg|iframe)[^>]*>.*?</\1>", "", html, flags=re.IGNORECASE | re.DOTALL)
                clean_text = re.sub(r"<[^>]+>", " ", clean_html)
                body_text = re.sub(r"\s+", " ", clean_text).strip()

            if len(body_text) > max_chars:
                body_text = body_text[:max_chars] + f"\n\n[... content truncated to {max_chars:,} characters ...]"

            header_lines = [f"=== Webpage Content: {title} ===", f"URL: {clean_url}"]
            if description:
                header_lines.append(f"Meta Summary: {description}")

            return "\n".join(header_lines) + f"\n\n{body_text}"

        except Exception as e:
            logger.error("Web scraping failed for %s: %s", clean_url, e)
            return f"⚠️ Could not extract web content from '{clean_url}': {e}"


# Singleton instance
media_processor = MediaProcessor()


def extract_text_from_photo(image_path: str) -> str:
    """Lightweight local OCR that uses almost zero RAM."""
    return media_processor.extract_text_from_photo(image_path)
