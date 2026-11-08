"""Tests for the clock package modules."""

import pytest
from datetime import datetime, timezone
from clock import (
    ClockTool,
    LocationClock,
    STANDARD_TIMEZONE_MAP,
    clock,
    format_short_time,
    get_current_time_str,
    get_time_conversational_reply,
    is_time_query,
)


def test_location_clock_initialization():
    loc = LocationClock(
        default_city="Vadakara",
        default_region="Kerala",
        default_country="India",
        default_lat=11.5978,
        default_lon=75.5814,
        default_timezone="Asia/Kolkata",
    )
    assert loc.city == "Vadakara"
    assert loc.timezone_name == "Asia/Kolkata"
    assert loc.latitude == 11.5978
    assert loc.longitude == 75.5814

    dt = loc.now()
    assert dt.tzinfo is not None
    assert "+05:30" in dt.strftime("%z") or "IST" in dt.strftime("%Z")


def test_gps_synchronization():
    loc = LocationClock()
    res = loc.sync_from_gps(35.6762, 139.6503, "Tokyo, Japan", "Asia/Tokyo")
    assert loc.city == "Tokyo, Japan"
    assert loc.timezone_name == "Asia/Tokyo"
    assert loc.latitude == 35.6762
    assert "local_time" in res


def test_formatting_utilities():
    ts = get_current_time_str()
    assert any(day in ts for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    assert ":" in ts

    short = format_short_time()
    assert ("AM" in short or "PM" in short)
    assert ":" in short


def test_time_intent_detection():
    assert is_time_query("nino rell me the time") is True
    assert is_time_query("nino tell me the time") is True
    assert is_time_query("what time is it") is True
    assert is_time_query("what's today's date") is True
    assert is_time_query("time ethrayaayi machan") is True
    assert is_time_query("samayam ethra") is True
    assert is_time_query("write me a python script") is False


def test_conversational_replies():
    reply_en = get_time_conversational_reply("tell me the time")
    assert "It is currently" in reply_en
    assert "Vadakara" in reply_en

    reply_manglish = get_time_conversational_reply("time ethrayaayi machan")
    assert "machan" in reply_manglish
    assert "Vadakara" in reply_manglish


def test_clock_tool():
    import asyncio
    tool = ClockTool()
    output = asyncio.run(tool.run())
    assert "Current Local Time:" in output
    assert "Location:" in output
    assert "Vadakara" in output
