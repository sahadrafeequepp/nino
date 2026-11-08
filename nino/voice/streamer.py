"""Sentence boundary detection and streaming pipeline for low-latency TTS."""

import re
from typing import AsyncIterator

# Common abbreviations that should NOT trigger a sentence split
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "vs", "etc", "eg", "ie", "al", "fig", "inc", "ltd", "corp",
    "approx", "apt", "dept", "est", "min", "max",
}


class SentenceBoundarySplitter:
    """Splits an incoming stream of tokens into discrete sentences for pipelined TTS."""

    def __init__(self, min_sentence_chars: int = 5):
        self.min_sentence_chars = min_sentence_chars
        self.buffer = ""

    def _is_abbreviation(self, text: str, punct_index: int) -> bool:
        """Check if the word immediately preceding the period is a known abbreviation."""
        prefix = text[:punct_index].rstrip()
        tokens = prefix.split()
        if not tokens:
            return False
        last_word = tokens[-1].lower().rstrip(".")
        return last_word in ABBREVIATIONS

    def _is_decimal_number(self, text: str, punct_index: int) -> bool:
        """Check if period is part of a number (e.g., 3.14)."""
        if punct_index > 0 and punct_index + 1 < len(text):
            prev_char = text[punct_index - 1]
            next_char = text[punct_index + 1]
            if prev_char.isdigit() and next_char.isdigit():
                return True
        return False

    def process_chunk(self, chunk: str) -> list[str]:
        """Add a token chunk to buffer and extract any complete sentences."""
        self.buffer += chunk
        sentences = []

        while True:
            # Check for double newline as immediate paragraph break
            if "\n\n" in self.buffer:
                idx = self.buffer.index("\n\n")
                candidate = self.buffer[:idx].strip()
                self.buffer = self.buffer[idx + 2:].lstrip()
                if candidate:
                    sentences.append(candidate)
                continue

            # Find all potential sentence ending marks (. ! ?) followed by whitespace
            search_start = 0
            found_sentence = False

            while search_start < len(self.buffer):
                match = re.search(r"([.!?]+)(\s+)", self.buffer[search_start:])
                if not match:
                    break

                abs_punct_idx = search_start + match.start(1)
                abs_punct_end = search_start + match.end(1)
                abs_match_end = search_start + match.end()

                # 1. Decimal check
                if self.buffer[abs_punct_idx] == "." and self._is_decimal_number(self.buffer, abs_punct_idx):
                    search_start = abs_match_end
                    continue

                # 2. Abbreviation check
                if self.buffer[abs_punct_idx] == "." and self._is_abbreviation(self.buffer, abs_punct_idx):
                    search_start = abs_match_end
                    continue

                # Valid sentence boundary found
                candidate = self.buffer[:abs_punct_end].strip()
                self.buffer = self.buffer[abs_match_end:]

                if len(candidate) >= self.min_sentence_chars:
                    sentences.append(candidate)
                    found_sentence = True
                break

            if not found_sentence:
                break

        return sentences

    def flush(self) -> list[str]:
        """Flush remaining text in buffer when token stream completes."""
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining:
            return [remaining]
        return []

    async def split_stream(self, token_stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """Asynchronously yield complete sentences from an incoming token stream."""
        async for token in token_stream:
            sentences = self.process_chunk(token)
            for s in sentences:
                yield s

        for s in self.flush():
            yield s


def split_text_to_sentences(text: str) -> list[str]:
    """Utility to split complete text into sentences."""
    splitter = SentenceBoundarySplitter()
    sentences = splitter.process_chunk(text)
    sentences.extend(splitter.flush())
    return sentences
