"""Tool registry for registering and discovering Nino tools."""

from typing import Any
from nino.tools.base import BaseTool


class ToolRegistry:
    """Registry maintaining available tools and their schema representations."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def to_ollama_tools(self) -> list[dict[str, Any]]:
        """Return all tools formatted for Ollama / OpenAI tool calling."""
        return [tool.to_ollama_tool() for tool in self._tools.values()]

    def get_relevant_tools(self, query: str | None = None) -> list[BaseTool]:
        """Select a focused subset of tools relevant to the query to preserve token budget and speed up inference."""
        if not self._tools:
            return []

        if not query:
            return list(self._tools.values())

        q = query.lower()
        matched: set[str] = set()

        category_map = {
            ("weather", "forecast", "rain", "snow", "temperature", "climate", "outside", "hot", "cold", "degree"): [
                "get_weather_forecast", "get_location", "get_current_time"
            ],
            ("location", "gps", "where am i", "city", "coordinates", "place", "set location", "my location", "store location", "home"): [
                "get_location", "set_location", "get_weather_forecast"
            ],

            ("task", "todo", "tasks", "checklist"): [
                "task_create", "task_list", "task_complete", "task_delete"
            ],
            ("reminder", "remind", "alarm", "schedule notification"): [
                "reminder_create", "reminder_list", "reminder_cancel", "reminder_snooze", "get_current_time"
            ],
            ("note", "notes", "write down", "memo", "remember this"): [
                "notes_create", "notes_search", "notes_read", "notes_append"
            ],
            ("calendar", "event", "meeting", "appointment", "schedule"): [
                "calendar_create_event", "calendar_list_events", "calendar_update_event", "calendar_delete_event", "get_current_time"
            ],
            ("calculate", "math", "sum", "multiply", "divide", "minus", "+", "-", "*", "/"): [
                "calculator"
            ],
            ("time", "date", "clock", "today", "now", "year", "month", "day"): [
                "get_current_time"
            ],
            ("file", "find file", "search file", "read file", "document"): [
                "file_search", "file_read"
            ],
            ("focus", "pomodoro", "timer", "work session"): [
                "focus_start", "focus_stop", "focus_status"
            ],
            ("light", "switch", "lamp", "plug", "home assistant", "ha ", "thermostat"): [
                "ha_list_entities", "ha_get_state", "ha_turn_on", "ha_turn_off", "ha_set_value", "ha_security_control"
            ],
            ("camera", "screenshot", "screen", "capture", "photo", "picture", "image"): [
                "screenshot_capture", "camera_snapshot", "image_describe"
            ],
            ("code", "python", "script", "execute"): [
                "code_execution"
            ],
            ("subagent", "delegate", "agent"): [
                "delegate_task"
            ],
        }

        for keywords, tool_names in category_map.items():
            if any(k in q for k in keywords):
                for tn in tool_names:
                    if tn in self._tools:
                        matched.add(tn)

        # If specific tools matched, return them (guaranteeing core essentials like get_current_time are handy)
        if matched:
            return [self._tools[name] for name in matched if name in self._tools]

        # Default fallback: return top everyday tools in compact mode
        core_fallback = [
            "get_current_time", "calculator", "get_weather_forecast", "get_location",
            "task_create", "task_list", "notes_create", "notes_search"
        ]
        return [self._tools[n] for n in core_fallback if n in self._tools]

    def get_system_prompt_spec(self, query: str | None = None, compact: bool = True) -> str:
        """Generate text specification of available tools for LLM prompting."""
        tools = self.get_relevant_tools(query) if query else list(self._tools.values())
        if not tools:
            return "No tools currently registered."
        if compact:
            specs = [tool.format_compact_spec() for tool in tools]
        else:
            specs = [tool.format_text_spec() for tool in tools]
        return "\n".join(specs)

