"""Unit tests for Piper TTS voice catalog, alias resolution, and switching."""

import pytest
from nino.voice.tts import (
    VOICE_CATALOG,
    resolve_voice_name,
    get_voice_info,
    is_voice_downloaded,
    PiperTTSAdapter,
)


def test_voice_catalog_contains_expected_voices():
    assert "kristin" in VOICE_CATALOG
    assert "cori" in VOICE_CATALOG
    assert "amy" in VOICE_CATALOG
    assert "ryan" in VOICE_CATALOG
    assert "alan" in VOICE_CATALOG


def test_resolve_voice_name_aliases():
    assert resolve_voice_name("kristin") == "en_US-kristin-medium"
    assert resolve_voice_name("cori") == "en_GB-cori-high"
    assert resolve_voice_name("ryan") == "en_US-ryan-medium"
    assert resolve_voice_name("1") == "en_US-kristin-medium"
    assert resolve_voice_name("en_US-amy-medium") == "en_US-amy-medium"


def test_get_voice_info():
    info = get_voice_info("kristin")
    assert info["name"] == "Kristin"
    assert info["gender"] == "Female"
    assert "US" in info["accent"]

    info_ryan = get_voice_info("ryan")
    assert info_ryan["name"] == "Ryan"
    assert info_ryan["gender"] == "Male"


def test_is_voice_downloaded():
    # Pre-downloaded voices in data/models/piper/
    assert is_voice_downloaded("kristin") is True
    assert is_voice_downloaded("cori") is True
    assert is_voice_downloaded("ryan") is True


def test_piper_adapter_initializes_with_alias():
    adapter = PiperTTSAdapter(voice="kristin")
    assert adapter.voice == "en_US-kristin-medium"
