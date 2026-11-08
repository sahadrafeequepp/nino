"""Time and date query intent detection and conversational response generation."""

import re
from typing import Optional
from clock.location_clock import LocationClock, clock

TIME_QUERY_REGEX = re.compile(
    r"\b(?:"
    r"what(?:'s|\s+is)?\s+(?:the\s+)?(?:current\s+)?time|"
    r"what\s+time\s+is\s+it|"
    r"(?:tell|rell|give)\s+me\s+(?:the\s+)?time|"
    r"current\s+time|"
    r"time\s+(?:now|please|right\s+now)|"
    r"what(?:'s|\s+is)?\s+(?:the\s+)?date|"
    r"what(?:'s|\s+is)?\s+today(?:'s)?\s+date|"
    r"today(?:'s)?\s+date|"
    r"what\s+day\s+is\s+(?:it|today)|"
    r"which\s+day\s+is\s+(?:it|today)|"
    r"samayam\s+(?:ethra|entha|parayu)|"
    r"time\s+(?:ethra|entha|para|parayu)|"
    r"ethra\s+samayam\s+aayi|"
    r"ethra\s+aayi\s+time|"
    r"time\s+ethrayaayi"
    r")\b|"
    r"^(?:nino[, ]*)?(?:the\s+)?time\??$|"
    r"^(?:nino[, ]*)?what\s+time\??$",
    re.IGNORECASE,
)


def is_time_query(text: str) -> bool:
    """Return True if user input is asking for current time, date, or day."""
    if not text:
        return False
    return bool(TIME_QUERY_REGEX.search(text.strip()))


def get_time_conversational_reply(query: str, location_clock: Optional[LocationClock] = None) -> str:
    """Generate a direct, conversational response with exact real-time clock data."""
    clk = location_clock or clock
    dt = clk.now()
    time_12 = dt.strftime("%I:%M %p").lstrip("0")
    time_24 = dt.strftime("%H:%M")
    day_name = dt.strftime("%A")
    date_str = dt.strftime("%d %B %Y")
    tz_abbr = dt.strftime("%Z") or "IST"
    period = clk.period_of_day(dt.hour).split()[0]

    q_lower = query.lower()
    is_manglish = any(w in q_lower for w in ("machan", "bro", "samayam", "ethra", "para", "parayu", "entha", "aayi"))
    is_date_only = any(w in q_lower for w in ("date", "day")) and not any(w in q_lower for w in ("time", "samayam"))

    if is_manglish:
        if is_date_only:
            return f"Innu {day_name}, {date_str} aanu machan! Vadakara-yil standard {tz_abbr}."
        return f"Ippo exact time {time_12} aayi machan ({day_name}, {date_str})! Vadakara-yil {period.lower()} time."

    if is_date_only:
        return f"Today is {day_name}, {date_str} ({clk.city}, {tz_abbr})."

    return f"It is currently {time_12} ({period}, {time_24} hrs) on {day_name}, {date_str} here in {clk.city} ({tz_abbr})."
