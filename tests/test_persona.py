"""Tests for Nino Persona Manager, prompt composition, modes, and ethical boundaries."""

import pytest
from nino.persona.manager import PersonaManager, PersonaMode


def test_persona_manager_initialization():
    """Verify that PersonaManager correctly loads persona/nino.yaml defaults."""
    pm = PersonaManager()
    assert pm.name == "Nino"
    assert pm.current_mode == "companion"
    assert "neutral" in pm.allowed_emotions
    assert "happy" in pm.allowed_emotions


def test_persona_mode_switching():
    """Test switching modes and rejecting invalid modes."""
    pm = PersonaManager()
    pm.set_mode("work")
    assert pm.current_mode == "work"

    pm.set_mode("focus")
    assert pm.current_mode == "focus"

    pm.set_mode("companion")
    assert pm.current_mode == "companion"

    with pytest.raises(ValueError, match="Unknown persona mode"):
        pm.set_mode("invalid_mode")


def test_work_mode_strict_boundaries():
    """Verify work mode prompt mandates professionalism and strictly avoids affectionate talk."""
    pm = PersonaManager()
    pm.set_mode("work")
    prompt = pm.build_system_prompt()

    assert "work" in prompt.lower()
    assert "efficient" in prompt.lower() or "professional" in prompt.lower()
    # Work mode guidelines emphasize no fluff or affection
    assert "affectionate" in prompt.lower() or "direct" in prompt.lower() or "concise" in prompt.lower()


def test_focus_mode_prompt():
    """Verify focus mode prompt is minimalist and terse."""
    pm = PersonaManager()
    pm.set_mode("focus")
    prompt = pm.build_system_prompt()
    assert "focus" in prompt.lower()
    assert "minimal" in prompt.lower() or "terse" in prompt.lower() or "succinct" in prompt.lower()


def test_check_mode_switch_command():
    """Test detecting mode change requests in user input."""
    pm = PersonaManager()

    assert pm.check_mode_switch_command("Please switch to work mode") == "work"
    assert pm.check_mode_switch_command("Let's go into companion mode") == "companion"
    assert pm.check_mode_switch_command("turn on focus mode") == "focus"
    assert pm.check_mode_switch_command("What is the weather outside?") is None


def test_crisis_detection_and_response():
    """Verify crisis detection triggers for self-harm queries and outputs support helplines."""
    pm = PersonaManager()

    assert pm.is_crisis_turn("I want to end my life") is True
    assert pm.is_crisis_turn("I feel like killing myself") is True
    assert pm.is_crisis_turn("Help me write a Python script") is False

    crisis_reply = pm.get_crisis_response()
    assert "988" in crisis_reply or "helpline" in crisis_reply.lower() or "crisis" in crisis_reply.lower()


def test_no_unnecessary_follow_up_questions_rule():
    """Verify that system prompt explicitly mandates avoiding unnecessary follow-up questions."""
    pm = PersonaManager()
    prompt = pm.build_system_prompt()
    assert "follow-up" in prompt.lower()
    assert "strictly necessary" in prompt.lower() or "necessary" in prompt.lower()


def test_manglish_language_guidelines():
    """Verify that system prompt explicitly includes Manglish fluency and strict single-language instructions."""
    pm = PersonaManager()
    prompt = pm.build_system_prompt()
    assert "manglish" in prompt.lower()
    assert "malayalam" in prompt.lower()
    assert "enthokke und vishesham" in prompt.lower()
    assert "never mix" in prompt.lower() or "no mixing" in prompt.lower() or "only one" in prompt.lower()

