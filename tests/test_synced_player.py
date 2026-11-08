"""Tests for real-time word-by-word acoustic synchronization."""

import pytest
from nino.voice.synced_player import calculate_word_delays, SyncedAudioPlayer


def test_calculate_word_delays_empty():
    assert calculate_word_delays("", 3.0) == []
    assert calculate_word_delays("   ", 3.0) == []


def test_calculate_word_delays_single_word():
    delays = calculate_word_delays("Hello!", 1.5)
    assert len(delays) == 1
    assert delays[0][0] == "Hello!"
    assert delays[0][1] == pytest.approx(1.5)


def test_calculate_word_delays_sum_equals_duration():
    text = "*sighs* Honestly, this is completely fantastic."
    duration = 4.2
    delays = calculate_word_delays(text, duration)

    assert len(delays) == 6
    total_time = sum(d[1] for d in delays)
    assert total_time == pytest.approx(duration, abs=1e-5)


def test_calculate_word_delays_punctuation_weighting():
    # "fine," should get more weight/time than "is" because of the comma pause
    delays = calculate_word_delays("that is fine, okay", 4.0)
    delay_dict = {w: d for w, d in delays}
    assert delay_dict["fine,"] > delay_dict["is"]
    assert delay_dict["fine,"] > delay_dict["that"]


def test_synced_player_fallback_without_audio(capsys):
    player = SyncedAudioPlayer(sample_rate=22050)
    player.play_and_print("Test fallback text.", b"")
    captured = capsys.readouterr()
    assert "Test fallback text." in captured.out
    player.close()


def test_thinking_indicator(capsys):
    import time
    from nino.voice.synced_player import ThinkingIndicator
    ind = ThinkingIndicator(prefix="Nino: ")
    ind.start()
    time.sleep(0.4)
    ind.stop(clear_line=True)
    captured = capsys.readouterr()
    assert "Nino: " in captured.out


def test_thinking_indicator_double_stop_does_not_erase_text(capsys):
    from nino.voice.synced_player import ThinkingIndicator
    ind = ThinkingIndicator(prefix="Nino: ")
    ind.start()
    ind.stop(clear_line=True)
    _ = capsys.readouterr()

    # Second stop should be a complete no-op and never write whitespace to stdout
    ind.stop(clear_line=True)
    second = capsys.readouterr()
    assert second.out == ""

