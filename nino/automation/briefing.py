"""Daily briefing generation aggregating calendar events, due tasks, reminders, and optional weather."""

from collections.abc import Callable
from datetime import datetime
import logging
from typing import Any, Optional
import httpx

from nino.config import Settings
from nino.memory.db import MemoryDB
from nino.models.base import LLMAdapter

logger = logging.getLogger("nino.automation.briefing")


class DailyBriefingGenerator:
    """Aggregates user's agenda for today and generates a factual, pressure-free morning summary."""

    def __init__(
        self,
        memory_db: MemoryDB,
        settings: Settings,
        adapter: Optional[LLMAdapter] = None,
        clock_fn: Optional[Callable[[], datetime]] = None,
    ):
        self.db = memory_db
        self.settings = settings
        self.adapter = adapter
        self.clock = clock_fn or datetime.now

    def now(self) -> datetime:
        return self.clock()

    def get_todays_calendar_events(self, today_str: Optional[str] = None) -> list[dict[str, Any]]:
        """Retrieve calendar events scheduled for today."""
        target_date = today_str or self.now().strftime("%Y-%m-%d")
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, title, start_time, end_time, location, description
                FROM calendar_events
                WHERE date(start_time) = date(?) OR start_time LIKE ?
                ORDER BY start_time ASC
                """,
                (target_date, f"{target_date}%"),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_pending_tasks(self) -> list[dict[str, Any]]:
        """Retrieve active pending tasks."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, title, due_date, priority
                FROM tasks
                WHERE status = 'pending'
                ORDER BY priority DESC, id ASC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_due_reminders(self, today_str: Optional[str] = None) -> list[dict[str, Any]]:
        """Retrieve reminders for today."""
        target_date = today_str or self.now().strftime("%Y-%m-%d")
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, message, trigger_time, urgent
                FROM reminders
                WHERE status = 'pending' AND (date(trigger_time) = date(?) OR trigger_time LIKE ?)
                ORDER BY trigger_time ASC
                """,
                (target_date, f"{target_date}%"),
            )
            return [dict(row) for row in cursor.fetchall()]

    async def get_weather_forecast(self) -> Optional[str]:
        """Fetch weather if enabled using coarse location."""
        if not self.settings.weather_enabled or not self.settings.weather_coarse_location:
            return None

        loc = self.settings.weather_coarse_location.strip()
        try:
            # Parse coarse lat,lon if provided (e.g. "37.7,-122.4")
            if "," in loc:
                parts = [float(p.strip()) for p in loc.split(",")]
                lat = round(parts[0], 1)
                lon = round(parts[1], 1)
                url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
                async with httpx.AsyncClient(timeout=4.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        curr = data.get("current_weather", {})
                        temp = curr.get("temperature")
                        wind = curr.get("windspeed")
                        return f"{temp}°C, wind {wind} km/h"
        except Exception as e:
            logger.warning("Weather fetch error (coarse loc=%s): %s", loc, e)
        return None

    async def generate_briefing(self) -> str:
        """Compose the daily briefing message."""
        today = self.now()
        date_str = today.strftime("%A, %B %d, %Y")
        today_iso = today.strftime("%Y-%m-%d")

        events = self.get_todays_calendar_events(today_iso)
        tasks = self.get_pending_tasks()
        reminders = self.get_due_reminders(today_iso)
        weather = await self.get_weather_forecast()

        # Build clean factual context
        sections = [f"Daily Briefing for {date_str}."]

        if weather:
            sections.append(f"Weather: {weather}.")

        if events:
            event_lines = []
            for ev in events:
                time_part = ev["start_time"].split("T")[1][:5] if "T" in ev["start_time"] else ev["start_time"]
                event_lines.append(f"- {ev['title']} at {time_part}")
            sections.append(f"Calendar ({len(events)} events):\n" + "\n".join(event_lines))
        else:
            sections.append("Calendar: No events scheduled for today.")

        if tasks:
            task_lines = [f"- {t['title']} ({t['priority']})" for t in tasks[:5]]
            sections.append(f"Pending Tasks ({len(tasks)}):\n" + "\n".join(task_lines))
        else:
            sections.append("Tasks: No pending tasks.")

        if reminders:
            reminder_lines = [f"- {r['message']}" for r in reminders]
            sections.append(f"Reminders ({len(reminders)}):\n" + "\n".join(reminder_lines))

        raw_briefing = "\n\n".join(sections)

        # If LLM adapter is available, we can optionally polish it into a concise spoken paragraph
        if self.adapter:
            try:
                prompt = (
                    "Summarize the following schedule into a friendly, clear, and factual morning briefing. "
                    "Keep it strictly under 4 sentences. Do not use artificial urgency, guilt, or engagement bait.\n\n"
                    f"{raw_briefing}"
                )
                summary = await self.adapter.chat([{"role": "user", "content": prompt}])
                if summary and len(summary.strip()) > 10:
                    return summary.strip()
            except Exception as e:
                logger.warning("LLM briefing generation failed, using direct template: %s", e)

        return raw_briefing
