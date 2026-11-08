"""Duplex and barge-in controller for audio input/output isolation."""

from enum import Enum
import logging
from typing import Callable, Optional

logger = logging.getLogger("nino.voice.duplex")


class DuplexMode(str, Enum):
    HALF_DUPLEX = "half_duplex"
    BARGE_IN = "barge_in"  # Experimental: allows interrupting Nino; best with headphones


class AssistantVoiceState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class DuplexController:
    """Controls microphone intake gating and barge-in interruption during playback."""

    def __init__(
        self,
        mode: DuplexMode = DuplexMode.HALF_DUPLEX,
        on_barge_in: Optional[Callable[[], None]] = None,
    ):
        self.mode = mode
        self.state = AssistantVoiceState.IDLE
        self.on_barge_in = on_barge_in
        self._consecutive_speech_frames_during_speaking = 0
        self.barge_in_threshold_frames = 3  # ~100ms of sustained speech to trigger barge-in

    def set_state(self, new_state: AssistantVoiceState) -> None:
        """Update voice state and reset speech counters."""
        logger.debug("Duplex state transition: %s -> %s", self.state, new_state)
        self.state = new_state
        if new_state != AssistantVoiceState.SPEAKING:
            self._consecutive_speech_frames_during_speaking = 0

    def can_process_microphone(self) -> bool:
        """Determine whether incoming microphone audio should be processed for speech."""
        if self.state != AssistantVoiceState.SPEAKING:
            return True

        # In half-duplex, mic is strictly muted while speaking to prevent feedback
        if self.mode == DuplexMode.HALF_DUPLEX:
            return False

        # In barge-in mode, mic audio is monitored for user interruptions
        return True

    def check_barge_in(self, is_speech: bool) -> bool:
        """Check if incoming frame should trigger a barge-in interruption."""
        if self.state != AssistantVoiceState.SPEAKING or self.mode != DuplexMode.BARGE_IN:
            return False

        if is_speech:
            self._consecutive_speech_frames_during_speaking += 1
            if self._consecutive_speech_frames_during_speaking >= self.barge_in_threshold_frames:
                logger.info("Barge-in triggered! User interrupted assistant playback.")
                if self.on_barge_in:
                    self.on_barge_in()
                self.set_state(AssistantVoiceState.LISTENING)
                return True
        else:
            self._consecutive_speech_frames_during_speaking = max(0, self._consecutive_speech_frames_during_speaking - 1)

        return False
