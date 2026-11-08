#!/usr/bin/env python3
"""Automated Secret and Sensitive Data Scanner for Git Commits & Upgrades.

Scans staged files (or all tracked files) to guarantee that:
1. No environment files (.env, .env.local, etc.) are committed.
2. No database or SQLite files (*.db, *.sqlite, etc.) are committed.
3. No user conversation logs, stickers, or personal media (bot_saved_data/, etc.) are committed.
4. No server audit or interaction logs (logs/, audit.jsonl, interactions.jsonl) are committed.
5. No hardcoded API keys, tokens (Telegram, OpenAI, Gemini, VirusTotal, etc.), or private keys exist.

Usage:
    python scripts/check_secrets.py --staged      (used by pre-commit hook)
    python scripts/check_secrets.py --pre-push    (used by pre-push hook)
    python scripts/check_secrets.py --all         (full scan of git tracked files)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Files / Paths strictly forbidden from being staged or tracked
FORBIDDEN_FILE_PATTERNS = [
    # Environment credentials
    (re.compile(r"(^|/)\.env(\.[^/]+)?$", re.IGNORECASE), "Environment configuration (.env) file containing secret keys"),
    (re.compile(r"(^|/)telegram_bot/\.env$", re.IGNORECASE), "Telegram bot environment file containing tokens"),
    
    # SQLite and Database binaries / transaction journals
    (re.compile(r"\.(db|sqlite|sqlite3|db-shm|db-wal)$", re.IGNORECASE), "Database or SQLite storage file"),
    
    # Bot saved conversations, user info, and downloaded media
    (re.compile(r"(^|/)bot_saved_data/", re.IGNORECASE), "Bot saved conversation data, sticker cache, or personal media"),
    (re.compile(r"(^|/)(interactions|reply|users)\.jsonl?$", re.IGNORECASE), "User conversation history or metadata"),
    
    # Server and audit logs
    (re.compile(r"(^|/)logs/.*\.log$", re.IGNORECASE), "Server execution log file"),
    (re.compile(r"(^|/)audit\.jsonl$", re.IGNORECASE), "System audit trail containing IP addresses or requests"),
    
    # Vault / private notes
    (re.compile(r"(^|/)vault/.", re.IGNORECASE), "Private notes vault storage"),
    
    # Private cryptographic keys
    (re.compile(r"\.(pem|key|pkcs12|pfx|id_rsa|id_ed25519)$", re.IGNORECASE), "Private cryptographic key file"),
]

# Exceptions: Files allowed despite pattern matches
WHITELISTED_FILES = {
    ".env.example",
    "telegram_bot/.env.example",
    "tests/test_audit.jsonl",
    "data/.gitkeep",
}

# Regex patterns for high-entropy secrets and credential leaks in code contents
SECRET_PATTERNS = [
    (
        re.compile(r"\b[0-9]{8,11}:[a-zA-Z0-9_-]{35}\b"),
        "Telegram Bot API Token (e.g. 123456789:ABCdefGHI...)",
    ),
    (
        re.compile(r"VIRUSTOTAL_API_KEY\s*=\s*['\"][a-f0-9]{64}['\"]", re.IGNORECASE),
        "Hardcoded 64-character VirusTotal API Key",
    ),
    (
        re.compile(r"\bsk-(?:live-|proj-)?[a-zA-Z0-9]{20,}\b"),
        "OpenAI API Secret Key",
    ),
    (
        re.compile(r"\bAIza[0-9A-Za-z-_]{35}\b"),
        "Google / Gemini API Key",
    ),
    (
        re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b"),
        "GitHub Personal Access Token",
    ),
    (
        re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|DSA|PGP)? PRIVATE KEY-----"),
        "Cryptographic Private Key block",
    ),
]

# Allowed placeholder tokens and test mock fixtures that should not trigger secret alerts
SAFE_PLACEHOLDERS = [
    "your_telegram_bot_token_here",
    "your_telegram_user_id_here",
    "change-me",
    "placeholder",
    "dummy",
    "sample",
    "example",
    "00000000",
    "pragma: allowlist secret",
]

TEST_FIXTURE_PATTERNS = [
    "1234567890",
    "abcdefghijklmnopqrstuvwx",
    "AKIAIOSFODNN7EXAMPLE",
]


def is_whitelisted(file_path: str) -> bool:
    norm = file_path.replace("\\", "/").lstrip("./")
    return norm in WHITELISTED_FILES


def get_staged_files() -> List[str]:
    """Get list of files currently staged in Git (Added, Copied, Modified)."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [f.strip() for f in out.splitlines() if f.strip()]
    except Exception:
        return []


def get_all_tracked_files() -> List[str]:
    """Get all tracked files in Git."""
    try:
        out = subprocess.check_output(
            ["git", "ls-files"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [f.strip() for f in out.splitlines() if f.strip()]
    except Exception:
        return []


def get_file_content(file_path: str, staged: bool = False) -> str:
    """Retrieve content of file, either from git staged index or working tree."""
    if staged:
        try:
            return subprocess.check_output(
                ["git", "show", f":{file_path}"],
                text=True,
                encoding="utf-8",
                errors="ignore",
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    path = Path(file_path)
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    return ""


def scan_file(file_path: str, staged: bool = False) -> List[Tuple[str, str, int]]:
    """Scan a single file for path and content violations. Returns [(type, message, line_num)]."""
    violations = []
    norm_path = file_path.replace("\\", "/").lstrip("./")

    if is_whitelisted(norm_path):
        return []

    # 1. Path & Filename Checks
    for pattern, reason in FORBIDDEN_FILE_PATTERNS:
        if pattern.search(norm_path):
            violations.append(("BLOCKED_PATH", f"{reason} (matched: {norm_path})", 0))
            return violations

    # 2. Content Checks (Skip binary formats)
    ext = Path(norm_path).suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".tar", ".gz", ".pyc"}:
        return []

    content = get_file_content(file_path, staged=staged)
    if not content:
        return []

    lines = content.splitlines()
    for line_idx, line in enumerate(lines, 1):
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("#"):
            continue

        # Skip lines that explicitly contain our safe placeholder tokens
        if any(safe in line for safe in SAFE_PLACEHOLDERS):
            continue

        # In unit test files, allow synthetic sequential dummy patterns used for scrubber testing
        if norm_path.startswith("tests/") and any(t_pat in line for t_pat in TEST_FIXTURE_PATTERNS):
            continue

        for sec_pattern, description in SECRET_PATTERNS:
            match = sec_pattern.search(line)
            if match:
                matched_val = match.group(0)
                # Mask secret for safe display
                if len(matched_val) > 8:
                    masked = matched_val[:4] + "..." + matched_val[-4:]
                else:
                    masked = "***"
                violations.append((
                    "SECRET_LEAK",
                    f"{description} found: '{masked}'",
                    line_idx,
                ))

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Git Secret & Sensitive Data Pre-Commit Guard")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true", help="Scan only git staged files (default for pre-commit)")
    group.add_argument("--pre-push", action="store_true", help="Scan all tracked files before push")
    group.add_argument("--all", action="store_true", help="Scan all git tracked files")
    args = parser.parse_args()

    staged_mode = args.staged or (not args.pre_push and not args.all)
    if staged_mode:
        files = get_staged_files()
        scan_label = "staged changes"
    else:
        files = get_all_tracked_files()
        scan_label = "tracked repository files"

    if not files:
        print(f"🛡️  [Security Guard] No {scan_label} to scan. Clean!")
        return 0

    print(f"🛡️  [Security Guard] Scanning {len(files)} {scan_label} for secrets, databases, and logs...")

    total_violations = 0
    for file_path in files:
        file_violations = scan_file(file_path, staged=staged_mode)
        if file_violations:
            for v_type, v_msg, v_line in file_violations:
                total_violations += 1
                if v_line > 0:
                    print(f"  ❌ [{v_type}] {file_path}:{v_line} - {v_msg}")
                else:
                    print(f"  ❌ [{v_type}] {file_path} - {v_msg}")

    if total_violations > 0:
        print("\n" + "=" * 70)
        print("🚨 COMMIT / PUSH BLOCKED: SENSITIVE DATA DETECTED!")
        print("=" * 70)
        print("The files listed above contain sensitive secrets, private databases, or logs.")
        print("To resolve this:")
        print("  1. If a forbidden file (.env, .db, bot_saved_data/) was staged:")
        print("     git reset HEAD <file>")
        print("  2. If hardcoded credentials exist in source code:")
        print("     Move credentials to .env and use os.getenv('KEY') in your code.")
        print("=" * 70)
        return 1

    print("✅ [Security Guard] All files verified clean. Safe to commit and push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
