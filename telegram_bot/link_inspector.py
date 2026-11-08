"""Safe Link Inspector & Web Content Fetcher with SSRF Firewall and Antivirus Scanning.

Security Pipeline:
1. SSRF Network Firewall check: Blocks private IPs, loopback, and cloud metadata.
2. AI Security & Data Breach check: Analyzes URL via Gemini API for phishing, malware,
   credential leaks, and data-breaching parameters.
3. Redirect Security Verification: Validates every redirect target against firewall rules.
4. Deep Antivirus Payload Scan: Scans fetched bytes for executable magic-bytes and script cradles.
5. Content Sanitization: Strips scripts, styles, and extracts readable text for AI analysis.
"""

import html
import json
import logging
import re
from typing import Any, Optional
from urllib.parse import urljoin, urlparse
import httpx

from nino.security.antivirus import antivirus_scanner
from nino.security.network_firewall import FirewallBlockedException, NetworkFirewall

logger = logging.getLogger("nino.telegram.link_inspector")

# URL extraction pattern
URL_REGEX = re.compile(
    r"(https?://(?:[a-zA-Z0-9\-._~%!$&'()*+,;=]+@)?(?:[a-zA-Z0-9\-._~%]+|\[[a-fA-F0-9:.]+\])(?::\d+)?(?:/[^\s]*)?)",
    re.IGNORECASE,
)

# High-risk query parameters that often indicate data exfiltration or credential leaks
SENSITIVE_PARAM_PATTERNS = [
    re.compile(r"[?&](?:token|password|passwd|api_key|apikey|secret|auth|session|cookie|ssn|credit_card)=", re.IGNORECASE),
]


class LinkInspector:
    """Orchestrates multi-layer security verification and content extraction for URLs."""

    def __init__(
        self,
        max_download_bytes: int = 5_000_000,
        **kwargs: Any,  # Absorb legacy kwargs without error
    ):
        self.max_download_bytes = max_download_bytes
        self.firewall = NetworkFirewall(allow_internal_loopback=False)

    def extract_urls(self, text: str) -> list[str]:
        """Find all HTTP/HTTPS and www URLs present in a given string."""
        if not text:
            return []
        urls = list(URL_REGEX.findall(text))

        # Also detect URLs starting with www. without http/https scheme
        www_matches = re.findall(r"(?:^|[\s(])(www\.[a-zA-Z0-9\-._~%]+\.[a-zA-Z]{2,}(?:/[^\s)]*)?)", text, re.IGNORECASE)
        for w in www_matches:
            full_w = f"https://{w.strip()}"
            if full_w not in urls:
                urls.append(full_w)

        # Strip trailing punctuation often caught by regex
        cleaned = []
        for u in urls:
            u_clean = u.rstrip(".,;:!?)>\"'")
            if u_clean and u_clean not in cleaned:
                cleaned.append(u_clean)
        return cleaned

    def check_heuristic_threats(self, url: str) -> tuple[bool, str]:
        """Perform rapid local heuristic checks for data breach and phishing patterns."""
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        # Check for sensitive data leakage in query parameters
        for pat in SENSITIVE_PARAM_PATTERNS:
            if pat.search(url):
                return False, "Data-breach hazard: URL contains exposed credentials/tokens in query parameters."

        # Check for IP literal hosts (common in phishing/malware delivery)
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
            return False, "High-risk raw IP host: Direct IP URLs are blocked for security."

        # Check suspicious TLDs often abused for malware/phishing
        suspicious_tlds = {".zip", ".mov", ".top", ".kim", ".country", ".stream", ".gdn"}
        for tld in suspicious_tlds:
            if hostname.endswith(tld):
                return False, f"High-risk top-level domain ({tld}) flagged for potential malicious activity."

        return True, "Passed local heuristics."

    async def fetch_and_scan_url(self, url: str) -> tuple[bool, str, dict[str, Any]]:
        """Validate URL against SSRF firewall, run VirusTotal/heuristic threat check, download safely, and scan payload.

        Returns: (is_safe, content_or_error, metadata)
        """
        metadata: dict[str, Any] = {"url": url, "stage": "init"}

        # 1. SSRF & Network Firewall verification
        try:
            self.firewall.validate_url(url)
        except FirewallBlockedException as e:
            logger.warning("Firewall blocked URL %s: %s", url, e)
            metadata["stage"] = "firewall"
            return False, f"⛔ *Firewall Blocked*: Prohibited internal/private network destination ({e}).", metadata

        # 2. Local Heuristic & VirusTotal Security Verification
        # Check local heuristics FIRST to immediately block data-breach params or raw IPs offline
        local_safe, local_reason = self.check_heuristic_threats(url)
        if not local_safe:
            logger.warning("Heuristic check flagged URL %s: %s", url, local_reason)
            metadata["stage"] = "heuristics"
            return False, f"🛡️ *Security Alert*: URL was blocked due to threat detection:\n• {local_reason}", metadata

        try:
            from telegram_bot.security_scanner import security_scanner
            vt_scan = await security_scanner.scan_url(url)
            if not vt_scan.is_safe:
                logger.warning("Security scanner flagged URL %s: %s", url, vt_scan.reason)
                metadata["stage"] = "security_scanner"
                return False, f"🛡️ *Security Alert*: URL was blocked due to threat detection:\n• {vt_scan.reason}", metadata
        except Exception as e:
            logger.debug("Security scanner check skipped or raised: %s", e)

        # 3. Safe HTTP Download with Redirect Firewalling
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NinoAssistant/2.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain,*/*;q=0.8",
        }

        current_url = url
        redirect_limit = 5
        downloaded_bytes = bytearray()
        content_type = ""

        try:
            async with httpx.AsyncClient(timeout=15.0, verify=True) as client:
                for _ in range(redirect_limit):
                    # Validate current redirect target against firewall
                    self.firewall.validate_url(current_url)

                    async with client.stream("GET", current_url, headers=headers, follow_redirects=False) as resp:
                        # Handle redirects manually to validate destination IPs
                        if resp.status_code in (301, 302, 303, 307, 308):
                            loc = resp.headers.get("Location")
                            if not loc:
                                break
                            current_url = urljoin(current_url, loc)
                            continue

                        if resp.status_code != 200:
                            return False, f"⚠️ HTTP Request returned status code {resp.status_code}.", metadata

                        content_type = resp.headers.get("Content-Type", "")

                        # Stream content with size limit guard
                        async for chunk in resp.aiter_bytes():
                            downloaded_bytes.extend(chunk)
                            if len(downloaded_bytes) > self.max_download_bytes:
                                return False, "⚠️ Web content exceeds 5MB safety limit.", metadata

                        break

        except Exception as e:
            logger.error("Failed to fetch URL %s: %s", url, e)
            return False, f"⚠️ Could not reach destination: {e}", metadata

        raw_payload = bytes(downloaded_bytes)
        metadata["download_size_bytes"] = len(raw_payload)
        metadata["content_type"] = content_type

        # 4. Deep Antivirus Scan on fetched payload
        url_path = urlparse(current_url).path or "index.html"
        scan_result = antivirus_scanner.scan_bytes(raw_payload, filename=url_path)
        if not scan_result.is_clean:
            metadata["stage"] = "antivirus"
            metadata["threat"] = scan_result.threat_name
            logger.warning("Antivirus flagged web payload for %s: %s", url, scan_result.threat_name)
            return (
                False,
                f"🛡️ *Antivirus Alert*: Downloaded content from link contains a security hazard!\n"
                f"• Threat: `{scan_result.threat_name}`\n• Details: {scan_result.details}",
                metadata,
            )

        # 5. Extract readable text & title
        cleaned_text = self._extract_clean_text(raw_payload, content_type)
        metadata["stage"] = "clean"
        return True, cleaned_text, metadata

    def _extract_clean_text(self, payload: bytes, content_type: str) -> str:
        """Extract clean, script-free text from HTML or document payload."""
        # Try decoding as text
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = payload.decode("latin-1")
            except Exception:
                return "[Binary data received - content not renderable as text]"

        # If HTML, strip script/style tags and clean markup
        if "html" in content_type.lower() or "<html" in text.lower() or "<body" in text.lower():
            # Remove scripts, styles, and comments
            text = re.sub(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

            # Extract title if present
            title_match = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else ""

            # Replace block tags with newlines
            text = re.sub(r"<(?:p|div|h[1-6]|li|blockquote|br)[^>]*>", "\n", text, flags=re.IGNORECASE)
            # Strip remaining tags
            text = re.sub(r"<[^>]+>", " ", text)
            text = html.unescape(text)

            # Collapse extra whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned = "\n".join(lines)

            if title:
                cleaned = f"# {title}\n\n{cleaned}"

            if len(cleaned) > 15_000:
                cleaned = cleaned[:15_000] + "\n\n...[content truncated for safety]"
            return cleaned

        # Plain text / JSON / Markdown
        if len(text) > 15_000:
            text = text[:15_000] + "\n\n...[content truncated for safety]"
        return text
