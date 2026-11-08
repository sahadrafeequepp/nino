"""Tests for sentence boundary splitting and streaming order in voice pipeline."""

import asyncio
import pytest
from nino.voice.streamer import SentenceBoundarySplitter, split_text_to_sentences


def test_split_text_to_sentences_standard():
    text = "Hello there! My name is Nino. How can I assist you today?"
    sentences = split_text_to_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello there!"
    assert sentences[1] == "My name is Nino."
    assert sentences[2] == "How can I assist you today?"


def test_sentence_splitter_preserves_abbreviations_and_decimals():
    text = "Dr. Smith arrived at 3.14 PM with Mr. Johnson. The project is ready."
    sentences = split_text_to_sentences(text)
    assert len(sentences) == 2
    assert "Dr. Smith arrived at 3.14 PM with Mr. Johnson." in sentences[0]
    assert sentences[1] == "The project is ready."


def test_sentence_splitter_handles_newlines():
    text = "First paragraph content.\n\nSecond paragraph content follows here."
    sentences = split_text_to_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "First paragraph content."
    assert sentences[1] == "Second paragraph content follows here."


@pytest.mark.asyncio
async def test_streaming_order_yields_first_sentence_early():
    """Verify that sentence 1 is emitted before later tokens arrive."""
    splitter = SentenceBoundarySplitter(min_sentence_chars=10)

    # Simulated slow token stream
    token_sequence = [
        "The ", "weather ", "is ", "sunny ", "today. ",  # Sentence 1 complete
        "Tomorrow ", "it ", "might ", "rain ", "heavily."  # Sentence 2 complete
    ]

    emitted_sentences = []
    tokens_seen_at_emission = []

    tokens_processed = 0
    for token in token_sequence:
        tokens_processed += 1
        results = splitter.process_chunk(token)
        for s in results:
            emitted_sentences.append(s)
            tokens_seen_at_emission.append(tokens_processed)

    for s in splitter.flush():
        emitted_sentences.append(s)
        tokens_seen_at_emission.append(tokens_processed)

    assert len(emitted_sentences) == 2
    assert emitted_sentences[0] == "The weather is sunny today."
    assert emitted_sentences[1] == "Tomorrow it might rain heavily."

    # Sentence 1 was emitted at token 5 (halfway through the stream!)
    assert tokens_seen_at_emission[0] == 5
    assert tokens_seen_at_emission[1] == 10
