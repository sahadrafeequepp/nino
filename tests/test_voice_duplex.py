"""Tests for half-duplex microphone muting and barge-in interruption logic."""

import pytest
from nino.voice.duplex import AssistantVoiceState, DuplexController, DuplexMode


def test_half_duplex_mutes_mic_during_speaking():
    """In half-duplex, microphone frames are discarded while Nino is speaking."""
    controller = DuplexController(mode=DuplexMode.HALF_DUPLEX)

    # When Idle or Listening: Mic is open
    controller.set_state(AssistantVoiceState.IDLE)
    assert controller.can_process_microphone() is True

    controller.set_state(AssistantVoiceState.LISTENING)
    assert controller.can_process_microphone() is True

    controller.set_state(AssistantVoiceState.PROCESSING)
    assert controller.can_process_microphone() is True

    # When Speaking: Mic is muted
    controller.set_state(AssistantVoiceState.SPEAKING)
    assert controller.can_process_microphone() is False

    # When done speaking: Mic is unmuted
    controller.set_state(AssistantVoiceState.IDLE)
    assert controller.can_process_microphone() is True


def test_barge_in_mode_keeps_mic_open():
    """In barge-in mode, mic is monitored even while speaking."""
    controller = DuplexController(mode=DuplexMode.BARGE_IN)
    controller.set_state(AssistantVoiceState.SPEAKING)
    assert controller.can_process_microphone() is True


def test_barge_in_interruption_triggers_callback():
    """When user speaks during playback in barge-in mode, playback is cancelled."""
    interrupted = False

    def on_barge_in():
        nonlocal interrupted
        interrupted = True

    controller = DuplexController(mode=DuplexMode.BARGE_IN, on_barge_in=on_barge_in)
    controller.set_state(AssistantVoiceState.SPEAKING)

    # Isolated noise frame does not trigger barge-in
    assert controller.check_barge_in(is_speech=False) is False
    assert interrupted is False

    # 1 speech frame
    assert controller.check_barge_in(is_speech=True) is False
    assert interrupted is False

    # 2 speech frames
    assert controller.check_barge_in(is_speech=True) is False
    assert interrupted is False

    # 3rd sustained speech frame triggers barge-in!
    assert controller.check_barge_in(is_speech=True) is True
    assert interrupted is True
    assert controller.state == AssistantVoiceState.LISTENING
