"""Tests for Privacy Gateway secret blocking, PII redaction, and egress control."""

import pytest
from nino.security.privacy_gateway import (
    PrivacyGateway,
    BlockedSecretException,
    SecurityException,
)


@pytest.fixture
def gateway():
    return PrivacyGateway(allow_external=False)


@pytest.fixture
def enabled_gateway():
    return PrivacyGateway(allow_external=True)


# --- Secret Blocking Tests ---

def test_blocks_aws_access_key(gateway):
    secret_text = "Here is my key AKIAIOSFODNN7EXAMPLE for testing"
    with pytest.raises(BlockedSecretException, match="sensitive credential or secret key"):
        gateway.sanitize(secret_text)


def test_blocks_private_key_header(gateway):
    secret_text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0...\n-----END RSA PRIVATE KEY-----"  # pragma: allowlist secret
    with pytest.raises(BlockedSecretException, match="sensitive credential or secret key"):
        gateway.sanitize(secret_text)


def test_blocks_openai_api_key(gateway):
    secret_text = "Please use sk-proj-1234567890abcdefghijklmnopqrstuvwx to query the API"
    with pytest.raises(BlockedSecretException, match="sensitive credential or secret key"):
        gateway.sanitize(secret_text)


def test_blocks_anthropic_api_key(gateway):
    secret_text = "Key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
    with pytest.raises(BlockedSecretException, match="sensitive credential or secret key"):
        gateway.sanitize(secret_text)


def test_blocks_github_token(gateway):
    secret_text = "Token: ghp_123456789012345678901234567890123456"
    with pytest.raises(BlockedSecretException, match="sensitive credential or secret key"):
        gateway.sanitize(secret_text)


def test_blocks_jwt_token(gateway):
    secret_text = "Auth: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.sdfsdfds"
    with pytest.raises(BlockedSecretException, match="sensitive credential or secret key"):
        gateway.sanitize(secret_text)


# --- PII Redaction Tests ---

def test_redacts_email_address(gateway):
    text = "Reach me at alice.smith@example.org or bob@corp.co.uk"
    sanitized = gateway.sanitize(text)
    assert "alice.smith@example.org" not in sanitized
    assert "bob@corp.co.uk" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized


def test_redacts_phone_numbers(gateway):
    text = "Call me at +1 (555) 234-5678 or 555-876-5432"
    sanitized = gateway.sanitize(text)
    assert "555" not in sanitized
    assert "[REDACTED_PHONE]" in sanitized


def test_redacts_ip_addresses(gateway):
    text = "Server IP is 192.168.1.100 and dns 10.0.0.1"
    sanitized = gateway.sanitize(text)
    assert "192.168.1.100" not in sanitized
    assert "10.0.0.1" not in sanitized
    assert "[REDACTED_IP]" in sanitized


def test_redacts_credit_card(gateway):
    text = "Payment card 4111 2222 3333 4444 should not leak"
    sanitized = gateway.sanitize(text)
    assert "4111" not in sanitized
    assert "[REDACTED_CARD]" in sanitized


def test_redacts_ssn(gateway):
    text = "My SSN is 000-12-3456"
    sanitized = gateway.sanitize(text)
    assert "000-12-3456" not in sanitized
    assert "[REDACTED_SSN]" in sanitized


# --- Egress & Logging Tests ---

@pytest.mark.asyncio
async def test_external_provider_disabled_by_default(gateway):
    messages = [{"role": "user", "content": "Hello world"}]
    with pytest.raises(SecurityException, match="External AI providers are disabled"):
        await gateway.dispatch_external("anthropic", messages)


@pytest.mark.asyncio
async def test_outbound_logging_metadata_only(enabled_gateway):
    messages = [
        {"role": "user", "content": "Contact alice@test.com regarding project X"}
    ]
    sanitized = await enabled_gateway.dispatch_external("custom-external", messages)

    # Content was redacted
    assert "[REDACTED_EMAIL]" in sanitized[0]["content"]
    assert "alice@test.com" not in sanitized[0]["content"]

    # Check logged metadata
    assert len(enabled_gateway.outbound_log) == 1
    log_entry = enabled_gateway.outbound_log[0]
    assert log_entry["target_provider"] == "custom-external"
    assert log_entry["message_count"] == 1
    assert log_entry["content_included"] is False
    assert "contact" not in str(log_entry).lower()
