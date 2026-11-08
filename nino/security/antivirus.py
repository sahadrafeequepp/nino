"""Antivirus & Content Scanner: Deep payload inspection, magic-byte verification, and quarantine."""

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

logger = logging.getLogger("nino.security.antivirus")


class AntivirusThreatDetected(Exception):
    """Raised when malicious content, executable masquerading, or exploit payloads are detected."""
    pass


# Prohibited executable and script extensions
DANGEROUS_EXTENSIONS = {
    ".exe", ".dll", ".sys", ".com", ".scr", ".pif",
    ".bat", ".cmd", ".ps1", ".psm1", ".vbs", ".vbe",
    ".js", ".jse", ".wsf", ".wsh", ".hta", ".cpl",
    ".msi", ".msp", ".reg", ".jar", ".bin",
}

# Magic-byte headers for compiled binaries and executables
EXECUTABLE_MAGIC_BYTES = [
    (b"MZ", "Windows PE executable/DLL"),
    (b"\x7fELF", "Linux ELF executable"),
    (b"\xfe\xed\xfa\xce", "Mach-O 32-bit executable"),
    (b"\xfe\xed\xfa\xcf", "Mach-O 64-bit executable"),
    (b"\xce\xfa\xed\xfe", "Mach-O executable (reverse byte order)"),
    (b"\xcf\xfa\xed\xfe", "Mach-O 64-bit (reverse byte order)"),
    (b"\xca\xfe\xba\xbe", "Java Class or Universal Mach-O binary"),
    (b"#!", "Shell script interpreter"),
]

# Patterns for malicious command cradles and obfuscated execution in text/documents
MALICIOUS_SCRIPT_PATTERNS = [
    re.compile(r"powershell(?:\.exe)?\s+.*(?:-enc|-encodedcommand|-e\s+|bypass|hidden)", re.IGNORECASE),
    re.compile(r"Invoke-Expression|(?:\bIEX\b)", re.IGNORECASE),
    re.compile(r"DownloadString|DownloadFile|Net\.WebClient", re.IGNORECASE),
    re.compile(r"WScript\.Shell|Shell\.Application", re.IGNORECASE),
    re.compile(r"certutil(?:\.exe)?\s+.*-(?:urlcache|decode)", re.IGNORECASE),
    re.compile(r"rundll32(?:\.exe)?\s+.*javascript:", re.IGNORECASE),
    re.compile(r"cmd(?:\.exe)?\s+/c\s+.*start", re.IGNORECASE),
]

# High-risk PDF exploit tokens (active execution actions only)
PDF_EXPLOIT_TOKENS = [
    b"/Type /Action /S /JavaScript",
    b"/JavaScript",
    b"/Launch",
]


class ScanResult:
    """Structured report of an antivirus scan."""

    def __init__(
        self,
        is_clean: bool,
        threat_name: str | None = None,
        details: str = "",
        quarantined_path: str | None = None,
        sha256: str = "",
    ):
        self.is_clean = is_clean
        self.threat_name = threat_name
        self.details = details
        self.quarantined_path = quarantined_path
        self.sha256 = sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_clean": self.is_clean,
            "threat_name": self.threat_name,
            "details": self.details,
            "quarantined_path": self.quarantined_path,
            "sha256": self.sha256,
        }

    def __repr__(self) -> str:
        status = "CLEAN" if self.is_clean else f"INFECTED({self.threat_name})"
        return f"<ScanResult status={status} sha256={self.sha256[:8]}...>"


class AntivirusScanner:
    """Local Antivirus scanner for inspecting files, web payloads, and ingested documents."""

    def __init__(
        self,
        quarantine_dir: str = "data/quarantine",
        enable_windows_defender: bool = False,
    ):
        self.quarantine_dir = Path(quarantine_dir)
        self.enable_windows_defender = enable_windows_defender
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def compute_sha256(self, data: bytes) -> str:
        """Compute SHA-256 hash of byte buffer."""
        return hashlib.sha256(data).hexdigest()

    def scan_bytes(
        self,
        data: bytes,
        filename: str = "stream_payload",
    ) -> ScanResult:
        """Scan raw byte buffer for executable magic bytes, PDF exploits, and malicious script tokens."""
        sha256 = self.compute_sha256(data)

        # 1. File extension check if filename is supplied
        ext = Path(filename).suffix.lower()
        if ext in DANGEROUS_EXTENSIONS:
            return ScanResult(
                is_clean=False,
                threat_name="ProhibitedDangerousExtension",
                details=f"File extension '{ext}' is classified as an executable/script security hazard.",
                sha256=sha256,
            )

        # 2. Magic-byte verification (detecting executable masquerading)
        # Even if file is named note.txt or doc.pdf, check if header is an executable
        for magic, desc in EXECUTABLE_MAGIC_BYTES:
            if data.startswith(magic):
                # Shell script '#!' is allowed only if explicitly expected; in notes/documents, it's blocked
                return ScanResult(
                    is_clean=False,
                    threat_name="ExecutableMasquerade",
                    details=f"Binary header matches forbidden executable format: {desc}.",
                    sha256=sha256,
                )

        # 3. PDF Exploit Scanner
        if ext == ".pdf" or data.startswith(b"%PDF"):
            for token in PDF_EXPLOIT_TOKENS:
                if token in data:
                    return ScanResult(
                        is_clean=False,
                        threat_name="PDFEmbeddedExploit",
                        details=f"PDF document contains prohibited dynamic execution token '{token.decode('latin1')}'.",
                        sha256=sha256,
                    )

        # 4. Text/Script analysis for payload cradles
        try:
            sample_text = data[:1048576].decode("utf-8", errors="ignore")
            for pattern in MALICIOUS_SCRIPT_PATTERNS:
                match = pattern.search(sample_text)
                if match:
                    return ScanResult(
                        is_clean=False,
                        threat_name="MaliciousCommandCradle",
                        details=f"Detected malicious execution signature: '{match.group(0)[:60]}'.",
                        sha256=sha256,
                    )
        except Exception:
            pass

        return ScanResult(is_clean=True, sha256=sha256)

    def scan_file(self, file_path: str | Path, quarantine_on_threat: bool = True) -> ScanResult:
        """Scan a file on disk. If infected and quarantine_on_threat is True, safely quarantines it."""
        path = Path(file_path)
        if not path.is_file():
            return ScanResult(
                is_clean=False,
                threat_name="FileNotFound",
                details=f"Target file does not exist: {path}",
            )

        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception as e:
            return ScanResult(
                is_clean=False,
                threat_name="FileReadError",
                details=f"Failed to read file for scanning: {e}",
            )

        result = self.scan_bytes(data, filename=path.name)

        # Optional: Windows Defender live scan if enabled and on Windows
        if result.is_clean and self.enable_windows_defender and os.name == "nt":
            wd_result = self._scan_with_windows_defender(path)
            if not wd_result.is_clean:
                result = wd_result

        if not result.is_clean and quarantine_on_threat:
            quarantined = self.quarantine_file(path, result)
            result.quarantined_path = str(quarantined)

        return result

    def quarantine_file(self, file_path: Path, result: ScanResult) -> Path:
        """Safely quarantine a file by isolating it in quarantine_dir."""
        dest_filename = f"{result.sha256[:12]}_{file_path.name}.quarantined"
        dest_path = self.quarantine_dir / dest_filename
        metadata_path = self.quarantine_dir / f"{dest_filename}.meta.json"

        try:
            shutil.move(str(file_path), str(dest_path))
            meta = {
                "original_path": str(file_path),
                "threat_name": result.threat_name,
                "details": result.details,
                "sha256": result.sha256,
            }
            metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.warning("File safely quarantined: %s -> %s (Threat: %s)", file_path, dest_path, result.threat_name)
        except Exception as e:
            logger.error("Failed to move file to quarantine: %s", e)

        return dest_path

    def _scan_with_windows_defender(self, path: Path) -> ScanResult:
        """Invoke Windows Defender MpCmdRun or PowerShell Start-MpScan."""
        try:
            cmd = [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"Start-MpScan -ScanPath '{path.resolve()}' -ScanType QuickScan",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if res.returncode != 0:
                return ScanResult(
                    is_clean=False,
                    threat_name="WindowsDefenderDetection",
                    details=f"Windows Defender flagged item or scan error: {res.stderr[:200]}",
                )
        except Exception as e:
            logger.debug("Windows Defender scan pass-through: %s", e)

        return ScanResult(is_clean=True)


# Global default scanner instance
antivirus_scanner = AntivirusScanner()
