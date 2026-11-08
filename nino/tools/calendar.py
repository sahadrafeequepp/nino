"""Local SQLite calendar tool with pure-Python RFC 5545 ICS export and import."""

import re
import uuid
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

from nino.memory.db import MemoryDB
from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier
from nino.tools.taint import TaintTracker


class CalendarCreateInput(BaseModel):
    title: str = Field(..., description="Event title or summary")
    start_time: str = Field(..., description="Start time (e.g. YYYY-MM-DD HH:MM or ISO-8601)")
    end_time: str = Field(..., description="End time (e.g. YYYY-MM-DD HH:MM or ISO-8601)")
    location: str | None = Field(default=None, description="Event location or link")
    description: str | None = Field(default=None, description="Detailed event description")


class CalendarCreateEventTool(BaseTool):
    name = "calendar_create_event"
    description = "Create a new event on the local calendar."
    args_schema = CalendarCreateInput
    tier = PermissionTier.REVERSIBLE

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(
        self,
        title: str,
        start_time: str,
        end_time: str,
        location: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO calendar_events (title, start_time, end_time, location, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (title, start_time, end_time, location or "", description or "", now, now),
            )
            event_id = cursor.lastrowid
        return {
            "status": "created",
            "event_id": event_id,
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "message": f"Calendar event #{event_id} '{title}' scheduled from {start_time} to {end_time}.",
        }


class CalendarListInput(BaseModel):
    start_date: str | None = Field(default=None, description="Optional start filter (e.g. YYYY-MM-DD)")
    end_date: str | None = Field(default=None, description="Optional end filter (e.g. YYYY-MM-DD)")


class CalendarListEventsTool(BaseTool):
    name = "calendar_list_events"
    description = "List scheduled calendar events within an optional date range."
    args_schema = CalendarListInput
    tier = PermissionTier.READ_ONLY

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, start_date: str | None = None, end_date: str | None = None) -> str:
        query = "SELECT id, title, start_time, end_time, location, description FROM calendar_events"
        params: list[Any] = []
        clauses = []

        if start_date:
            clauses.append("start_time >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("start_time <= ?")
            params.append(end_date)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY start_time ASC"

        with self.db.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        if not rows:
            return "No scheduled calendar events found."

        lines = [f"Found {len(rows)} event(s):"]
        for r in rows:
            loc = f" @ {r['location']}" if r["location"] else ""
            lines.append(f"- #{r['id']}: '{r['title']}' ({r['start_time']} -> {r['end_time']}){loc}")
            if r["description"]:
                lines.append(f"  Description: {r['description']}")

        return "\n".join(lines)


class CalendarUpdateInput(BaseModel):
    event_id: int = Field(..., description="ID of the event to update")
    title: str | None = Field(default=None, description="New title")
    start_time: str | None = Field(default=None, description="New start time")
    end_time: str | None = Field(default=None, description="New end time")
    location: str | None = Field(default=None, description="New location")
    description: str | None = Field(default=None, description="New description")


class CalendarUpdateEventTool(BaseTool):
    name = "calendar_update_event"
    description = "Update details of an existing calendar event."
    args_schema = CalendarUpdateInput
    tier = PermissionTier.REVERSIBLE

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(
        self,
        event_id: int,
        title: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        location: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM calendar_events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            if not row:
                return {"status": "error", "message": f"Calendar event #{event_id} not found."}

            new_title = title if title is not None else row["title"]
            new_start = start_time if start_time is not None else row["start_time"]
            new_end = end_time if end_time is not None else row["end_time"]
            new_loc = location if location is not None else row["location"]
            new_desc = description if description is not None else row["description"]

            conn.execute(
                """
                UPDATE calendar_events
                SET title = ?, start_time = ?, end_time = ?, location = ?, description = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_title, new_start, new_end, new_loc, new_desc, now, event_id),
            )

        return {
            "status": "updated",
            "event_id": event_id,
            "title": new_title,
            "message": f"Calendar event #{event_id} '{new_title}' updated successfully.",
        }


class CalendarDeleteInput(BaseModel):
    event_id: int = Field(..., description="ID of the calendar event to delete")


class CalendarDeleteEventTool(BaseTool):
    name = "calendar_delete_event"
    description = "Delete a scheduled calendar event. Requires confirmation."
    args_schema = CalendarDeleteInput
    tier = PermissionTier.SENSITIVE

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, event_id: int) -> dict[str, Any]:
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT title FROM calendar_events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            if not row:
                return {"status": "error", "message": f"Calendar event #{event_id} not found."}

            conn.execute("DELETE FROM calendar_events WHERE id = ?", (event_id,))

        return {
            "status": "deleted",
            "event_id": event_id,
            "title": row["title"],
            "message": f"Calendar event #{event_id} '{row['title']}' was deleted.",
        }


class CalendarExportICSInput(BaseModel):
    event_ids: list[int] | None = Field(default=None, description="Optional list of event IDs to export (exports all if omitted)")


class CalendarExportICSTool(BaseTool):
    name = "calendar_export_ics"
    description = "Export calendar events to standard RFC 5545 iCalendar (.ics) format."
    args_schema = CalendarExportICSInput
    tier = PermissionTier.READ_ONLY

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, event_ids: list[int] | None = None) -> str:
        with self.db.get_connection() as conn:
            if event_ids:
                placeholders = ",".join("?" for _ in event_ids)
                cursor = conn.execute(f"SELECT * FROM calendar_events WHERE id IN ({placeholders})", event_ids)
            else:
                cursor = conn.execute("SELECT * FROM calendar_events ORDER BY start_time ASC")
            rows = cursor.fetchall()

        now_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Nino Local AI Assistant//Calendar//EN",
            "CALSCALE:GREGORIAN",
        ]

        def format_ics_dt(dt_str: str) -> str:
            clean = re.sub(r"[^0-9]", "", dt_str)
            if len(clean) >= 14:
                return clean[:14] + "Z"
            if len(clean) >= 8:
                return clean[:8]
            return clean

        for r in rows:
            lines.append("BEGIN:VEVENT")
            lines.append(f"UID:nino-event-{r['id']}-{now_stamp}@localhost")
            lines.append(f"DTSTAMP:{now_stamp}")
            lines.append(f"DTSTART:{format_ics_dt(r['start_time'])}")
            lines.append(f"DTEND:{format_ics_dt(r['end_time'])}")
            lines.append(f"SUMMARY:{r['title']}")
            if r["location"]:
                lines.append(f"LOCATION:{r['location']}")
            if r["description"]:
                lines.append(f"DESCRIPTION:{r['description']}")
            lines.append("END:VEVENT")

        lines.append("END:VCALENDAR")
        return "\r\n".join(lines)


class CalendarImportICSInput(BaseModel):
    ics_content: str = Field(..., description="Raw text of an RFC 5545 iCalendar (.ics) file")


class CalendarImportICSTool(BaseTool):
    name = "calendar_import_ics"
    description = "Import events from an RFC 5545 iCalendar (.ics) string into the local calendar."
    args_schema = CalendarImportICSInput
    tier = PermissionTier.REVERSIBLE

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, ics_content: str) -> dict[str, Any]:
        # Unfold lines according to RFC 5545 (line continuation starting with space/tab)
        unfolded = re.sub(r"\r?\n[ \t]", "", ics_content)
        lines = unfolded.splitlines()

        events_imported = 0
        in_event = False
        curr_event: dict[str, str] = {}
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection() as conn:
            for line in lines:
                line = line.strip()
                if line == "BEGIN:VEVENT":
                    in_event = True
                    curr_event = {}
                elif line == "END:VEVENT":
                    in_event = False
                    title = curr_event.get("SUMMARY", "Untitled Imported Event")
                    start = curr_event.get("DTSTART", now)
                    end = curr_event.get("DTEND", start)
                    loc = curr_event.get("LOCATION", "")
                    desc = curr_event.get("DESCRIPTION", "")
                    conn.execute(
                        """
                        INSERT INTO calendar_events (title, start_time, end_time, location, description, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (title, start, end, loc, desc, now, now),
                    )
                    events_imported += 1
                elif in_event and ":" in line:
                    key, _, val = line.partition(":")
                    clean_key = key.split(";")[0].upper()
                    curr_event[clean_key] = val

        return {
            "status": "imported",
            "events_imported": events_imported,
            "message": f"Successfully imported {events_imported} event(s) into calendar.",
        }
