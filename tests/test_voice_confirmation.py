"""Tests for spoken tool confirmation parsing and safety enforcement."""

import pytest
from nino.memory.db import MemoryDB
from nino.tools.confirmation import ConfirmationManager
from nino.voice.confirmation_flow import (
    ConfirmationIntent,
    SpokenConfirmationHandler,
    parse_spoken_confirmation,
)


def test_spoken_confirmation_affirmative():
    """Verify that only clear affirmative phrases return APPROVED."""
    assert parse_spoken_confirmation("yes") == ConfirmationIntent.APPROVED
    assert parse_spoken_confirmation("YES") == ConfirmationIntent.APPROVED
    assert parse_spoken_confirmation("yeah") == ConfirmationIntent.APPROVED
    assert parse_spoken_confirmation("confirm") == ConfirmationIntent.APPROVED
    assert parse_spoken_confirmation("approve") == ConfirmationIntent.APPROVED
    assert parse_spoken_confirmation("do it") == ConfirmationIntent.APPROVED
    assert parse_spoken_confirmation("go ahead") == ConfirmationIntent.APPROVED
    assert parse_spoken_confirmation("proceed") == ConfirmationIntent.APPROVED
    assert parse_spoken_confirmation("sure!") == ConfirmationIntent.APPROVED


def test_spoken_confirmation_negative():
    """Verify that explicit rejection phrases return DENIED."""
    assert parse_spoken_confirmation("no") == ConfirmationIntent.DENIED
    assert parse_spoken_confirmation("NO!") == ConfirmationIntent.DENIED
    assert parse_spoken_confirmation("nope") == ConfirmationIntent.DENIED
    assert parse_spoken_confirmation("cancel") == ConfirmationIntent.DENIED
    assert parse_spoken_confirmation("stop") == ConfirmationIntent.DENIED
    assert parse_spoken_confirmation("don't") == ConfirmationIntent.DENIED
    assert parse_spoken_confirmation("deny") == ConfirmationIntent.DENIED
    assert parse_spoken_confirmation("abort") == ConfirmationIntent.DENIED


def test_spoken_confirmation_ambiguous_never_approves():
    """Critical safety rule: Ambiguous, mumbled, or conflicting words MUST NOT approve."""
    ambiguous_inputs = [
        "maybe",
        "uhh",
        "ummm",
        "what did you say",
        "yes or no",
        "yes but wait a second",
        "no problem",
        "I don't know",
        "surely not",
        "can you repeat that",
        "hello?",
        "",
        "   ",
        "definitely maybe",
        "yes delete everything now please hurry up wait no",
    ]

    for user_input in ambiguous_inputs:
        intent = parse_spoken_confirmation(user_input)
        assert intent != ConfirmationIntent.APPROVED, f"Input '{user_input}' improperly approved!"
        assert intent in (ConfirmationIntent.AMBIGUOUS, ConfirmationIntent.DENIED)


def test_spoken_confirmation_handler_integration(tmp_path):
    """Test full handler approval, denial, and ambiguous safety cancellation."""
    db_path = str(tmp_path / "test_conf.db")
    memory_db = MemoryDB(db_path)
    conf_mgr = ConfirmationManager(memory_db)
    handler = SpokenConfirmationHandler(conf_mgr)

    # 1. Test Approved Flow
    conf1 = conf_mgr.create_confirmation("task_delete", {"task_id": "123"}, summary="Delete task #123", tier="SENSITIVE")
    approved, msg = handler.process_spoken_reply(conf1.id, "yes")
    assert approved is True
    assert "confirmed" in msg.lower()
    assert conf_mgr.get_confirmation(conf1.id).status == "approved"

    # 2. Test Denied Flow
    conf2 = conf_mgr.create_confirmation("calendar_delete_event", {"event_id": "evt_1"}, summary="Delete event", tier="SENSITIVE")
    approved, msg = handler.process_spoken_reply(conf2.id, "no")
    assert approved is False
    assert "cancelled" in msg.lower()
    assert conf_mgr.get_confirmation(conf2.id).status == "denied"

    # 3. Test Ambiguous Flow (Mumbled/Unclear reply) -> Safely cancelled!
    conf3 = conf_mgr.create_confirmation("run_python_code", {"code": "import os"}, summary="Run python code", tier="SENSITIVE")
    approved, msg = handler.process_spoken_reply(conf3.id, "uhh maybe later")
    assert approved is False
    assert "unclear" in msg.lower() or "safety" in msg.lower()
    assert conf_mgr.get_confirmation(conf3.id).status == "denied"
