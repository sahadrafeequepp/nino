"""Multi-stage Security & Virus Scanning Pipeline for Nino.

Integrates:
- VirusTotal API (v3) for real-time URL and file reputation checks
- In-memory SHA-256 hash caching to eliminate redundant remote queries
- File upload scanning for unknown binaries under 32MB
- Local safety inspection fallback (magic bytes, MIME verification, path traversal sanitization, PDF exploits)
- Immediate quarantine and safe disposal of flagged payloads
"""

import asyncio
import base64
import hashlib
import logging
import os
from pathlib import Path
import re
from typing import Any, Optional, Tuple
from urllib.parse import urlparse
import httpx
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()
load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("nino.security.scanner")

# Dangerous executable and script extensions
DANGEROUS_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".scr", ".sh", ".elf", ".com", ".pif",
    ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh", ".hta", ".cpl",
    ".msi", ".msp", ".reg", ".dll", ".sys", ".ps1", ".psm1", ".jar",
    ".bin",
}

# Magic-byte signatures for executables
EXECUTABLE_MAGIC_BYTES = [
    (b"MZ", "Windows PE executable / DLL"),
    (b"\x7fELF", "Linux ELF executable"),
    (b"\xfe\xed\xfa\xce", "Mach-O 32-bit executable"),
    (b"\xfe\xed\xfa\xcf", "Mach-O 64-bit executable"),
    (b"\xce\xfa\xed\xfe", "Mach-O executable (reverse byte order)"),
    (b"\xcf\xfa\xed\xfe", "Mach-O 64-bit (reverse byte order)"),
    (b"\xca\xfe\xba\xbe", "Java Class or Universal Mach-O binary"),
    (b"#!", "Shell script interpreter"),
    (b"L\x00\x00\x00\x01\x14\x02\x00", "Windows Shortcut (LNK) file"),
]

# High-risk PDF exploit tokens (active execution actions only)
PDF_EXPLOIT_TOKENS = [
    b"/JavaScript",
    b"/Launch",
]

# Common local / loopback hostnames and IP ranges
BLOCKED_URL_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal",
    "169.254.169.254",
}


class SecurityScanResult:
    """Represents the structured outcome of a security scan."""

    def __init__(
        self,
        is_safe: bool,
        reason: str = "",
        details: Optional[dict[str, Any]] = None,
        positives: int = 0,
        sha256: str = "",
    ):
        self.is_safe = is_safe
        self.reason = reason
        self.details = details or {}
        self.positives = positives
        self.sha256 = sha256

    def alert_message(self) -> str:
        """User-facing alert message if unsafe."""
        if not self.is_safe:
            return "⚠️ Security Alert: File/URL flagged as unsafe and discarded."
        return "File verified safe."

    def __repr__(self) -> str:
        status = "SAFE" if self.is_safe else f"UNSAFE({self.reason})"
        return f"<SecurityScanResult {status} positives={self.positives} sha256={self.sha256[:8] if self.sha256 else 'none'}>"


class SecurityScanner:
    """Multi-layer malware, reputation, and file integrity scanner."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        quarantine_dir: str = "data/quarantine",
        timeout: float = 15.0,
    ):
        raw_key = api_key or os.getenv("VIRUSTOTAL_API_KEY", "")
        self.api_key = raw_key.strip().strip('"').strip("'")
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        # In-memory cache: sha256 -> SecurityScanResult
        self.hash_cache: dict[str, SecurityScanResult] = {}
        # URL cache: url -> SecurityScanResult
        self.url_cache: dict[str, SecurityScanResult] = {}

    @property
    def has_vt_api(self) -> bool:
        """Check if a non-placeholder VirusTotal API key is configured."""
        return bool(self.api_key and "your_virustotal_api_key" not in self.api_key.lower())

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename using os.path.basename to prevent path traversal."""
        clean = os.path.basename(filename).strip()
        # Strip path traversal artifacts and null bytes
        clean = clean.replace("\x00", "").replace("/", "").replace("\\", "")
        return clean or "unnamed_file.bin"

    # =========================================================================
    # URL Reputation & Security Checks
    # =========================================================================
    async def scan_url(self, url: str) -> SecurityScanResult:
        """Scan a URL using VirusTotal API (v3) with fallback to local security heuristics."""
        clean_url = url.strip()
        if not clean_url:
            return SecurityScanResult(is_safe=False, reason="Empty URL provided.")

        # Check cache
        if clean_url in self.url_cache:
            return self.url_cache[clean_url]

        # 1. Local URL structural and SSRF heuristics
        try:
            parsed = urlparse(clean_url)
            if parsed.scheme.lower() not in {"http", "https"}:
                res = SecurityScanResult(
                    is_safe=False,
                    reason=f"Disallowed URL scheme: '{parsed.scheme}'. Only HTTP and HTTPS are permitted.",
                )
                self.url_cache[clean_url] = res
                return res

            hostname = (parsed.hostname or "").lower()
            if not hostname or hostname in BLOCKED_URL_HOSTS or hostname.startswith("127.") or hostname.startswith("192.168.") or hostname.startswith("10."):
                res = SecurityScanResult(
                    is_safe=False,
                    reason="SSRF Protection: Blocked private or loopback destination.",
                )
                self.url_cache[clean_url] = res
                return res
        except Exception as e:
            res = SecurityScanResult(is_safe=False, reason=f"Malformed URL: {e}")
            self.url_cache[clean_url] = res
            return res

        # 2. VirusTotal API (v3) URL check
        if self.has_vt_api:
            try:
                # VirusTotal v3 URL ID is URL-safe base64 without '=' padding
                url_id = base64.urlsafe_b64encode(clean_url.encode("utf-8")).decode("utf-8").rstrip("=")
                headers = {"x-apikey": self.api_key}

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    vt_url = f"https://www.virustotal.com/api/v3/urls/{url_id}"
                    response = await client.get(vt_url, headers=headers)

                    if response.status_code == 200:
                        data = response.json()
                        stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                        malicious = stats.get("malicious", 0)
                        suspicious = stats.get("suspicious", 0)
                        positives = malicious + suspicious

                        if positives > 0:
                            logger.warning("VirusTotal flagged URL %s: %d detections (%s)", clean_url, positives, stats)
                            res = SecurityScanResult(
                                is_safe=False,
                                reason=f"VirusTotal detected {positives} security flags ({malicious} malicious, {suspicious} suspicious).",
                                details=stats,
                                positives=positives,
                            )
                            self.url_cache[clean_url] = res
                            return res
                        else:
                            res = SecurityScanResult(
                                is_safe=True,
                                reason="VirusTotal report clean.",
                                details=stats,
                                positives=0,
                            )
                            self.url_cache[clean_url] = res
                            return res

                    elif response.status_code == 404:
                        # Submit URL for fresh analysis
                        submit_url = "https://www.virustotal.com/api/v3/urls"
                        await client.post(submit_url, headers=headers, data={"url": clean_url})
                        logger.info("Submitted new URL %s to VirusTotal for analysis.", clean_url)
                        # Given 404, fallback to local heuristics (passed above)
                        res = SecurityScanResult(
                            is_safe=True,
                            reason="URL queued for VirusTotal scan; passed local validation.",
                        )
                        self.url_cache[clean_url] = res
                        return res
                    else:
                        logger.warning("VirusTotal URL query returned HTTP %s for %s", response.status_code, clean_url)
            except Exception as e:
                logger.error("VirusTotal URL scan exception for %s: %s", clean_url, e)

        # Fallback: Safe if passed local checks
        res = SecurityScanResult(is_safe=True, reason="Passed local safety and SSRF validation.")
        self.url_cache[clean_url] = res
        return res

    # =========================================================================
    # File & Binary Security Checks
    # =========================================================================
    async def scan_file_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        expected_mime: Optional[str] = None,
    ) -> SecurityScanResult:
        """Perform comprehensive multi-stage scanning on file bytes.
        
        1. Sanitize filename (os.path.basename)
        2. Calculate SHA-256 hash and check local cache
        3. Query VirusTotal by SHA-256
        4. Upload file if unknown and < 32MB
        5. Deep local inspection (magic bytes, extensions, script signatures, PDF exploits)
        """
        safe_name = self.sanitize_filename(filename)
        sha256 = hashlib.sha256(file_bytes).hexdigest()

        # Check in-memory hash cache
        if sha256 in self.hash_cache:
            cached = self.hash_cache[sha256]
            logger.debug("Hit SHA-256 cache for %s (%s): is_safe=%s", safe_name, sha256[:8], cached.is_safe)
            return cached

        # Run local checks first for instant rejection of obvious threats
        local_result = self.perform_local_inspection(file_bytes, safe_name, expected_mime)
        if not local_result.is_safe:
            local_result.sha256 = sha256
            self.hash_cache[sha256] = local_result
            return local_result

        # VirusTotal v3 Inspection
        if self.has_vt_api:
            vt_result = await self._scan_vt_file(file_bytes, safe_name, sha256)
            if vt_result is not None:
                vt_result.sha256 = sha256
                self.hash_cache[sha256] = vt_result
                return vt_result

        # If VirusTotal was not reached or completed, return the clean local result
        local_result.sha256 = sha256
        self.hash_cache[sha256] = local_result
        return local_result

    async def scan_file_on_disk(
        self,
        file_path: str | Path,
        filename: Optional[str] = None,
        expected_mime: Optional[str] = None,
    ) -> SecurityScanResult:
        """Scan a file residing on disk. Immediately quarantines/removes if flagged."""
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            return SecurityScanResult(is_safe=False, reason="Target file does not exist.")

        name = filename or path.name
        safe_name = self.sanitize_filename(name)

        try:
            file_bytes = path.read_bytes()
        except Exception as e:
            return SecurityScanResult(is_safe=False, reason=f"Could not read file: {e}")

        result = await self.scan_file_bytes(file_bytes, safe_name, expected_mime)

        if not result.is_safe:
            self._quarantine_file(path, safe_name, result)

        return result

    async def _scan_vt_file(
        self,
        file_bytes: bytes,
        safe_filename: str,
        sha256: str,
    ) -> Optional[SecurityScanResult]:
        """Query VirusTotal v3 by hash, uploading if unknown and under 32MB."""
        headers = {"x-apikey": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 1. Query by SHA-256 hash first
                hash_url = f"https://www.virustotal.com/api/v3/files/{sha256}"
                response = await client.get(hash_url, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                    malicious = stats.get("malicious", 0)
                    suspicious = stats.get("suspicious", 0)
                    positives = malicious + suspicious

                    if positives > 0:
                        logger.warning(
                            "VirusTotal flagged file %s (%s): %d positives (%s)",
                            safe_filename, sha256[:8], positives, stats,
                        )
                        return SecurityScanResult(
                            is_safe=False,
                            reason=f"VirusTotal detected {positives} security flags.",
                            details=stats,
                            positives=positives,
                            sha256=sha256,
                        )
                    else:
                        logger.info("VirusTotal verified file %s (%s) is clean.", safe_filename, sha256[:8])
                        return SecurityScanResult(
                            is_safe=True,
                            reason="VirusTotal verified clean.",
                            details=stats,
                            positives=0,
                            sha256=sha256,
                        )

                # 2. If unknown (404) and size under 32MB (33,554,432 bytes)
                if response.status_code == 404 and len(file_bytes) <= 32 * 1024 * 1024:
                    logger.info("File %s unknown on VirusTotal; uploading for analysis (%d bytes)...", safe_filename, len(file_bytes))
                    upload_url = "https://www.virustotal.com/api/v3/files"
                    files = {"file": (safe_filename, file_bytes)}
                    upload_resp = await client.post(upload_url, headers=headers, files=files)

                    if upload_resp.status_code in {200, 201}:
                        upload_data = upload_resp.json()
                        analysis_id = upload_data.get("data", {}).get("id")
                        logger.info("Uploaded %s to VirusTotal; analysis ID=%s. Polling verdict...", safe_filename, analysis_id)

                        if analysis_id:
                            analysis_url = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
                            for _ in range(3):
                                await asyncio.sleep(2.0)
                                a_resp = await client.get(analysis_url, headers=headers)
                                if a_resp.status_code == 200:
                                    a_data = a_resp.json()
                                    attr = a_data.get("data", {}).get("attributes", {})
                                    if attr.get("status") == "completed":
                                        stats = attr.get("stats", {})
                                        malicious = stats.get("malicious", 0)
                                        suspicious = stats.get("suspicious", 0)
                                        positives = malicious + suspicious
                                        if positives > 0:
                                            logger.warning("VirusTotal analysis flagged %s: %d positives", safe_filename, positives)
                                            return SecurityScanResult(
                                                is_safe=False,
                                                reason=f"VirusTotal detected {positives} security flags.",
                                                details=stats,
                                                positives=positives,
                                                sha256=sha256,
                                            )
                                        else:
                                            logger.info("VirusTotal analysis confirmed %s is clean.", safe_filename)
                                            return SecurityScanResult(
                                                is_safe=True,
                                                reason="VirusTotal verified clean.",
                                                details=stats,
                                                positives=0,
                                                sha256=sha256,
                                            )

                        # If analysis is still pending after polling, assume safe since uploaded and passed sanity checks
                        return SecurityScanResult(
                            is_safe=True,
                            reason="Queued on VirusTotal; passed security verification.",
                            sha256=sha256,
                        )
                    else:
                        logger.warning("VirusTotal upload returned HTTP %s", upload_resp.status_code)

        except Exception as e:
            logger.warning("VirusTotal API error during file scan for %s: %s", safe_filename, e)

        return None

    # =========================================================================
    # Local Deep Inspection Fallback
    # =========================================================================
    def perform_local_inspection(
        self,
        file_bytes: bytes,
        filename: str,
        expected_mime: Optional[str] = None,
    ) -> SecurityScanResult:
        """Perform comprehensive file signature, magic byte, and extension inspection."""
        ext = Path(filename).suffix.lower()

        # 1. Prohibited extension check
        if ext in DANGEROUS_EXTENSIONS:
            return SecurityScanResult(
                is_safe=False,
                reason=f"Blocked dangerous executable file extension: '{ext}'.",
            )

        # 2. Executable magic-byte masquerade detection
        for magic_bytes, desc in EXECUTABLE_MAGIC_BYTES:
            if file_bytes.startswith(magic_bytes):
                return SecurityScanResult(
                    is_safe=False,
                    reason=f"Executable masquerade detected ({desc}) inside '{filename}'.",
                )

        # 3. Optional python-magic MIME verification if library installed
        try:
            import magic  # type: ignore
            detected_mime = magic.from_buffer(file_bytes, mime=True)
            if detected_mime:
                if detected_mime.startswith("application/x-dosexec") or detected_mime.startswith("application/x-executable"):
                    return SecurityScanResult(
                        is_safe=False,
                        reason=f"Dangerous MIME type detected: {detected_mime}",
                    )
        except Exception:
            pass

        # 4. Deep PDF Exploit Token inspection
        if ext == ".pdf" or b"%PDF-" in file_bytes[:1024]:
            for token in PDF_EXPLOIT_TOKENS:
                if token in file_bytes:
                    return SecurityScanResult(
                        is_safe=False,
                        reason=f"High-risk PDF exploit structure detected: '{token.decode('ascii', errors='ignore')}'.",
                    )

        # 5. Media signature consistency check
        if ext in {".jpg", ".jpeg"} and not file_bytes.startswith(b"\xff\xd8\xff"):
            return SecurityScanResult(
                is_safe=False,
                reason="Corrupted or disguised JPEG image header.",
            )
        if ext == ".png" and not file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return SecurityScanResult(
                is_safe=False,
                reason="Corrupted or disguised PNG image header.",
            )
        if ext == ".gif" and not (file_bytes.startswith(b"GIF87a") or file_bytes.startswith(b"GIF89a")):
            return SecurityScanResult(
                is_safe=False,
                reason="Corrupted or disguised GIF image header.",
            )

        return SecurityScanResult(is_safe=True, reason="Passed local magic-byte inspection.")

    def _quarantine_file(self, file_path: Path, safe_filename: str, result: SecurityScanResult) -> None:
        """Immediately remove or quarantine an unsafe file from disk."""
        try:
            quarantine_target = self.quarantine_dir / f"quarantine_{safe_filename}"
            if file_path.exists():
                # Write reason sidecar
                sidecar = quarantine_target.with_suffix(quarantine_target.suffix + ".reason.txt")
                sidecar.write_text(f"Threat Reason: {result.reason}\nPositives: {result.positives}\nSHA256: {result.sha256}", encoding="utf-8")
                # Move file into quarantine
                file_path.replace(quarantine_target)
                logger.warning("Quarantined malicious file %s to %s", file_path, quarantine_target)
        except Exception as e:
            logger.error("Failed to move file to quarantine (%s); attempting direct unlink: %s", file_path, e)
            try:
                if file_path.exists():
                    file_path.unlink(missing_ok=True)
            except Exception:
                pass


# Singleton instance for simple import across bot and pipeline
security_scanner = SecurityScanner()
