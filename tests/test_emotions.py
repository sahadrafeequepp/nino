"""Tests for EmotionParser, tag extraction, stripping, and streaming chunk boundary handling."""

import pytest
from nino.persona.emotions import EmotionParser


def test_parse_complete_emotion_tag():
    """Verify standard [emotion:<name>] tag extraction and clean text generation."""
    parser = EmotionParser()
    text = "[emotion:happy] Good morning! It is great to see you."
    res = parser.parse_text(text)

    assert res.emotion == "happy"
    assert res.clean_text == "Good morning! It is great to see you."


def test_unknown_emotion_tag_stripped():
    """Verify unknown emotion tags are stripped and fallback to default or None."""
    parser = EmotionParser()
    text = "[emotion:flying] Look at that bird."
    res = parser.parse_text(text)

    # flying is not in whitelist
    assert res.emotion is None
    assert res.clean_text == "Look at that bird."


def test_streaming_chunk_boundary_split():
    """Verify emotion tag split across multiple streaming chunks is properly buffered and parsed."""
    parser = EmotionParser()

    # Chunk 1 contains start of tag
    out1 = parser.feed_chunk("Hello [emo")
    assert out1.clean_text == "Hello "
    assert out1.emotion is None

    # Chunk 2 completes the tag and contains rest of sentence
    out2 = parser.feed_chunk("tion:excited] world!")
    # "excited" not in standard whitelist so emotion is None, tag is stripped
    assert out2.clean_text == "world!"

    # Reset and test with valid tag
    parser.reset()
    out1 = parser.feed_chunk("[emotion:")
    assert out1.clean_text == ""
    assert out1.emotion is None

    out2 = parser.feed_chunk("surprised] Wow, that works!")
    assert out2.emotion == "surprised"
    assert out2.clean_text == "Wow, that works!"


def test_multiple_emotions_in_text():
    """Verify that multiple emotion tags in a single string are stripped cleanly."""
    parser = EmotionParser()
    text = "[emotion:thinking] Let me see... [emotion:happy] Here is the solution!"
    res = parser.parse_text(text)

    assert res.clean_text == "Let me see... Here is the solution!"
    assert res.emotion in ["thinking", "happy"]


def test_case_insensitivity_and_whitespace():
    """Verify emotion tag handles mixed case and surrounding whitespace."""
    parser = EmotionParser()
    text = "[emotion:  Relaxed  ] Take a deep breath."
    res = parser.parse_text(text)

    assert res.emotion == "relaxed"
    assert res.clean_text == "Take a deep breath."
