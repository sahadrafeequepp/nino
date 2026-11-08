"""Real-time synchronized audio playback and terminal word printing."""

import asyncio
import re
import sys
import threading
import time
from typing import Any, Callable, Optional

try:
    import sounddevice as sd
except ImportError:
    sd = None


def calculate_word_delays(text: str, total_duration: float) -> list[tuple[str, float]]:
    """Distribute total audio duration across words based on character length and punctuation pauses.

    This aligns terminal word printing with acoustic speech output (including silence pauses
    after commas, periods, question marks, and action directions).
    """
    if not text or not text.strip():
        return []

    words = text.strip().split()
    if not words:
        return []

    if len(words) == 1:
        return [(words[0], max(0.05, total_duration))]

    if total_duration <= 0:
        return [(w, 0.0) for w in words]

    weights: list[float] = []
    words_since_pause = 0

    from nino.voice.tts import NON_ENDING_WORDS, CONJUNCTIONS_AND_CONNECTORS

    for i, w in enumerate(words):
        # Check for stage action directions like *chuckles*, *sighs*
        if w.startswith("*") and w.endswith("*"):
            weights.append(3.5)
            words_since_pause = 0
            continue

        clean = re.sub(r"[^a-zA-Z0-9]", "", w)
        w_len = max(len(clean), 1)
        weight = float(w_len)
        words_since_pause += 1

        # Natural pauses calibrated to acoustic pauses in PiperTTS
        has_explicit_pause = False
        if w.endswith("..."):
            weight += 8.0
            has_explicit_pause = True
        elif w.endswith(("?", "!")):
            weight += 6.0
            has_explicit_pause = True
        elif w.endswith("."):
            weight += 5.0
            has_explicit_pause = True
        elif w.endswith((",", ";", ":", "--", "—")):
            weight += 3.5
            has_explicit_pause = True

        if has_explicit_pause:
            words_since_pause = 0
        else:
            # Pacing: Account for 4-5 word breather pauses even if text lacked explicit comma
            remaining = len(words) - (i + 1)
            if remaining > 1:
                clean_cur = clean.lower()
                next_w = words[i + 1].lower().strip(".,!?:;\"'()[]")
                if clean_cur not in NON_ENDING_WORDS:
                    if words_since_pause >= 4 and next_w in CONJUNCTIONS_AND_CONNECTORS:
                        weight += 3.5
                        words_since_pause = 0
                    elif words_since_pause >= 5:
                        weight += 3.5
                        words_since_pause = 0

        weights.append(weight)

    total_weight = sum(weights)
    if total_weight <= 0:
        uniform_delay = total_duration / len(words)
        return [(w, uniform_delay) for w in words]

    return [(w, total_duration * (wt / total_weight)) for w, wt in zip(words, weights)]


class SyncedAudioPlayer:
    """Manages persistent sounddevice output stream and synchronizes word printing with audio."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._stream: Optional[Any] = None
        self._lock = threading.Lock()

    def _ensure_stream(self):
        """Open and start a persistent RawOutputStream if not already active."""
        if sd is None:
            return None
        with self._lock:
            if self._stream is None or getattr(self._stream, "closed", True):
                try:
                    self._stream = sd.RawOutputStream(
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype="int16",
                    )
                    self._stream.start()
                except Exception:
                    self._stream = None
            return self._stream

    def play_and_print(
        self,
        text: str,
        audio_bytes: bytes,
        on_word: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Synchronously play audio bytes through speakers while printing words in real time."""
        if not audio_bytes:
            if text and text.strip():
                if on_word:
                    on_word(text)
                else:
                    sys.stdout.write(text.strip() + " ")
                    sys.stdout.flush()
            return

        stream = self._ensure_stream()
        if stream is None:
            # Fallback if no audio device
            if text and text.strip():
                sys.stdout.write(text.strip() + " ")
                sys.stdout.flush()
            return

        duration = len(audio_bytes) / (self.sample_rate * 2)
        word_delays = calculate_word_delays(text, duration)

        started_event = threading.Event()

        def _play_worker():
            try:
                started_event.set()
                stream.write(audio_bytes)
            except Exception:
                pass

        th = threading.Thread(target=_play_worker, daemon=True)
        th.start()
        started_event.wait(timeout=1.0)

        # Print words in sync with the audio duration
        for word, delay in word_delays:
            if on_word:
                on_word(word)
            else:
                sys.stdout.write(word + " ")
                sys.stdout.flush()
            time.sleep(delay)

        th.join(timeout=2.0)

    async def play_and_print_async(
        self,
        text: str,
        audio_bytes: bytes,
        on_word: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Asynchronously play audio bytes through speakers while printing words in real time."""
        if not audio_bytes:
            if text and text.strip():
                if on_word:
                    on_word(text)
                else:
                    sys.stdout.write(text.strip() + " ")
                    sys.stdout.flush()
            return

        stream = self._ensure_stream()
        if stream is None:
            if text and text.strip():
                sys.stdout.write(text.strip() + " ")
                sys.stdout.flush()
            return

        duration = len(audio_bytes) / (self.sample_rate * 2)
        word_delays = calculate_word_delays(text, duration)

        started_event = threading.Event()

        def _play_worker():
            try:
                started_event.set()
                stream.write(audio_bytes)
            except Exception:
                pass

        th = threading.Thread(target=_play_worker, daemon=True)
        th.start()
        await asyncio.to_thread(started_event.wait, 1.0)

        for word, delay in word_delays:
            if on_word:
                on_word(word)
            else:
                sys.stdout.write(word + " ")
                sys.stdout.flush()
            await asyncio.sleep(delay)

        await asyncio.to_thread(th.join, 2.0)

    def close(self) -> None:
        """Safely close the persistent output stream."""
        with self._lock:
            if self._stream is not None:
                try:
                    if not getattr(self._stream, "closed", True):
                        self._stream.stop()
                        self._stream.close()
                except Exception:
                    pass
                self._stream = None


class ThinkingIndicator:
    """Displays an animated 'Nino: ...' indicator on the terminal while thinking."""

    def __init__(self, prefix: str = "Nino: "):
        self.prefix = prefix
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_active = False

    def _run(self):
        frames = [".  ", ".. ", "..."]
        idx = 0
        while not self._stop_event.is_set():
            sys.stdout.write(f"\r{self.prefix}{frames[idx % 3]}")
            sys.stdout.flush()
            idx += 1
            if self._stop_event.wait(0.35):
                break

    def start(self):
        """Start the animated dots in background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._is_active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, clear_line: bool = True):
        """Stop animation and clear the indicator line ONLY if it was actively animating."""
        if not self._is_active and not (self._thread and self._thread.is_alive()):
            return

        self._is_active = False
        if self._thread is not None:
            self._stop_event.set()
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._thread = None

        if clear_line:
            # Clear only the exact width of prefix + animated dots, never beyond
            erase_len = len(self.prefix) + 4
            sys.stdout.write("\r" + " " * erase_len + "\r")
            sys.stdout.flush()
