"""Unit tests for Antivirus scanner, Network SSRF Firewall, and Search Anonymization."""

import io
from pathlib import Path
import pytest

from nino.security.antivirus import (
    AntivirusScanner,
    antivirus_scanner,
    AntivirusThreatDetected,
    ScanResult,
)
from nino.security.network_firewall import (
    NetworkFirewall,
    network_firewall,
    FirewallBlockedException,
)
from nino.security.privacy_gateway import PrivacyGateway, BlockedSecretException


# --- Network SSRF Firewall Tests ---

def test_firewall_blocks_private_ip():
    fw = NetworkFirewall()
    assert fw.is_ip_blocked("127.0.0.1") is True
    assert fw.is_ip_blocked("10.0.0.5") is True
    assert fw.is_ip_blocked("192.168.1.1") is True
    assert fw.is_ip_blocked("169.254.169.254") is True  # Cloud metadata
    assert fw.is_ip_blocked("8.8.8.8") is False         # Public IP


def test_firewall_blocks_prohibited_schemes():
    fw = NetworkFirewall()
    with pytest.raises(FirewallBlockedException, match="Scheme 'file' is prohibited"):
        fw.validate_url("file:///etc/passwd")

    with pytest.raises(FirewallBlockedException, match="Scheme 'gopher' is prohibited"):
        fw.validate_url("gopher://127.0.0.1:70")


def test_firewall_blocks_loopback_hostname():
    fw = NetworkFirewall(allow_internal_loopback=False)
    with pytest.raises(FirewallBlockedException, match="SSRF attempt"):
        fw.validate_url("http://localhost:8000/api")


# --- Antivirus Scanner Tests ---

def test_antivirus_blocks_executable_extension():
    scanner = AntivirusScanner()
    res = scanner.scan_bytes(b"dummy data", filename="malware.exe")
    assert res.is_clean is False
    assert res.threat_name == "ProhibitedDangerousExtension"


def test_antivirus_blocks_masqueraded_pe_executable():
    scanner = AntivirusScanner()
    # Windows PE header starts with MZ
    fake_doc = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 100
    res = scanner.scan_bytes(fake_doc, filename="innocent_note.txt")
    assert res.is_clean is False
    assert res.threat_name == "ExecutableMasquerade"


def test_antivirus_blocks_pdf_embedded_js():
    scanner = AntivirusScanner()
    malicious_pdf = b"%PDF-1.5\n1 0 obj\n<< /Type /Action /S /JavaScript /JS (app.alert(1)) >>\nendobj"
    res = scanner.scan_bytes(malicious_pdf, filename="contract.pdf")
    assert res.is_clean is False
    assert res.threat_name == "PDFEmbeddedExploit"


def test_antivirus_blocks_powershell_cradle():
    scanner = AntivirusScanner()
    payload = b"powershell.exe -enc aQB3AHIAIABoAHQAdABw... -ExecutionPolicy Bypass"
    res = scanner.scan_bytes(payload, filename="instructions.txt")
    assert res.is_clean is False
    assert res.threat_name == "MaliciousCommandCradle"


def test_antivirus_passes_clean_note(tmp_path):
    scanner = AntivirusScanner(quarantine_dir=str(tmp_path / "quarantine"))
    clean_note = tmp_path / "shopping.md"
    clean_note.write_text("# Shopping List\n- Apples\n- Almond milk\n", encoding="utf-8")

    res = scanner.scan_file(clean_note)
    assert res.is_clean is True
    assert clean_note.exists()


def test_antivirus_quarantines_infected_file(tmp_path):
    quarantine_dir = tmp_path / "quarantine"
    scanner = AntivirusScanner(quarantine_dir=str(quarantine_dir))

    bad_file = tmp_path / "report.pdf"
    bad_file.write_bytes(b"%PDF-1.4\n/Launch << /F (cmd.exe) >>\n")

    res = scanner.scan_file(bad_file, quarantine_on_threat=True)
    assert res.is_clean is False
    assert res.threat_name == "PDFEmbeddedExploit"
    assert not bad_file.exists()  # Moved to quarantine
    assert res.quarantined_path is not None
    assert Path(res.quarantined_path).exists()


# --- Privacy Gateway Query Anonymization Tests ---

def test_anonymize_search_query():
    query = "Please search for side effects of ibuprofen 200mg my email is test@test.com"
    anon = PrivacyGateway.anonymize_search_query(query)
    assert "test@test.com" not in anon
    assert "[REDACTED_EMAIL]" in anon
    assert "ibuprofen 200mg" in anon


def test_anonymize_search_query_blocks_secret():
    secret_query = "look up docs for API key sk-proj-1234567890abcdefghijklmnopqrstuvwx"
    with pytest.raises(BlockedSecretException):
        PrivacyGateway.anonymize_search_query(secret_query)
