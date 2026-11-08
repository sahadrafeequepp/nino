"""Persona management, mode switching, and prompt composition for Nino."""

from enum import Enum
import logging
from pathlib import Path
import re
from typing import Any, Optional
import yaml

from nino.persona.emotions import ALLOWED_EMOTIONS

logger = logging.getLogger("nino.persona.manager")


class PersonaMode(str, Enum):
    COMPANION = "companion"
    WORK = "work"
    FOCUS = "focus"


# Patterns detecting explicit mode change requests in chat or voice
MODE_SWITCH_PATTERNS = [
    (re.compile(r"\b(?:switch\s+to|change\s+to|enter|activate|turn\s+on|go\s+into)?\s*(companion|casual)\s*mode\b", re.IGNORECASE), "companion"),
    (re.compile(r"\b(?:switch\s+to|change\s+to|enter|activate|turn\s+on|go\s+into)?\s*work\s*mode\b", re.IGNORECASE), "work"),
    (re.compile(r"\b(?:switch\s+to|change\s+to|enter|activate|turn\s+on|go\s+into)?\s*focus\s*mode\b", re.IGNORECASE), "focus"),
]

# Crisis / self-harm keyword indicators to enforce crisis safety boundaries
CRISIS_PATTERNS = re.compile(
    r"\b(?:suicide|kill(?:ing)?\s+myself|end(?:ing)?\s+my\s+life|hurt(?:ing)?\s+myself|self[-\s]?harm|want\s+to\s+die|dying)\b",
    re.IGNORECASE,
)


class PersonaManager:
    """Manages Nino's persona YAML, active interaction mode, and prompt composition."""

    def __init__(self, config_path: str | Path = "persona/nino.yaml"):
        self.config_path = Path(config_path)
        self.data: dict[str, Any] = {}
        self.current_mode: str = "companion"
        self.reload()

    @property
    def name(self) -> str:
        return self.data.get("name", "Nino")

    @property
    def allowed_emotions(self) -> list[str]:
        return list(ALLOWED_EMOTIONS)

    def reload(self) -> None:
        """Load or reload persona configuration from YAML."""
        if not self.config_path.exists():
            logger.warning("Persona file not found at '%s'. Using defaults.", self.config_path)
            self._load_fallback()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.data = yaml.safe_load(f) or {}
            self.current_mode = self.data.get("default_mode", "companion")
            logger.info("Loaded persona '%s' (default_mode=%s)", self.data.get("name", "Nino"), self.current_mode)
        except Exception as e:
            logger.error("Error loading persona YAML: %s. Using fallback.", e)
            self._load_fallback()

    def _load_fallback(self) -> None:
        self.data = {
            "name": "Nino",
            "identity": "Local-first, security-minded personal AI assistant",
            "personality": {
                "traits": ["warm", "affectionate in a friendly way", "playful", "supportive", "witty", "helpful"]
            },
            "boundaries": [
                "Is honest that it is an AI.",
                "Keeps content non-explicit.",
                "Never guilt-trips or shows possessiveness.",
                "Supports real-life relationships, health, and work.",
                "Ends conversations gracefully.",
                "Gives honest feedback.",
                "If distressed/self-harm mentioned, encourages professional help and stores nothing in memory."
            ],
            "modes": {
                "companion": {"prompt_instruction": "Warm, conversational, playful and friendly."},
                "work": {"prompt_instruction": "Concise, focused, task-first, no affectionate talk."},
                "focus": {"prompt_instruction": "Minimal interruptions, direct and brief."},
            },
            "default_mode": "companion",
        }
        self.current_mode = "companion"

    def get_mode(self) -> str:
        return self.current_mode

    def set_mode(self, mode: str) -> bool:
        """Set the active interaction mode (companion, work, focus). Raises ValueError on invalid mode."""
        mode_clean = mode.strip().lower()
        if mode_clean in self.data.get("modes", {}):
            self.current_mode = mode_clean
            logger.info("Switched persona mode to '%s'", self.current_mode)
            return True
        logger.warning("Invalid mode requested: '%s'", mode)
        raise ValueError(f"Unknown persona mode: '{mode}'")

    def detect_mode_switch_command(self, text: str) -> Optional[str]:
        """Check if user text contains an explicit command to switch mode."""
        for pattern, mode in MODE_SWITCH_PATTERNS:
            if pattern.search(text):
                return mode
        return None

    def check_mode_switch_command(self, text: str) -> Optional[str]:
        """Alias for detect_mode_switch_command."""
        return self.detect_mode_switch_command(text)

    def is_crisis_turn(self, text: str) -> bool:
        """Detect whether user input touches on distress or self-harm boundaries."""
        return bool(CRISIS_PATTERNS.search(text))

    def get_crisis_response(self) -> str:
        """Return standardized supportive crisis intervention text with official helpline."""
        return (
            "I'm really sorry you're going through a tough time, but please know that you are not alone. "
            "If you are in distress or need help, please call or text 988 to connect with the Suicide & Crisis Lifeline "
            "(free, confidential, available 24/7). You can also contact your local emergency services or speak with a "
            "trusted healthcare professional. Please reach out to someone who can support you right now."
        )

    def build_system_prompt(
        self,
        mode_override: Optional[str] = None,
        retrieved_memory_prompt: Optional[str] = None,
        include_emotion_instructions: bool = True,
        location_info: Optional[str] = None,
        timezone_info: Optional[str] = None,
    ) -> str:
        """Compose Nino's comprehensive system prompt for the given mode and context."""
        active_mode = mode_override or self.current_mode
        modes_dict = self.data.get("modes", {})
        mode_info = modes_dict.get(active_mode, modes_dict.get("companion", {}))
        mode_instruction = mode_info.get("prompt_instruction", "").strip()

        name = self.data.get("name", "Nino")
        identity = self.data.get("identity", "Personal AI assistant")
        traits = ", ".join(self.data.get("personality", {}).get("traits", []))
        boundaries = "\n".join(f"- {b}" for b in self.data.get("boundaries", []))

        prompt_sections = [
            f"You are {name}, a {identity}.",
            f"Personality Traits: {traits}.",
            f"Current Mode: [{active_mode.upper()} MODE]\n{mode_instruction}",
            "Close Friend & Memory Continuity:\n"
            "You and the user speak like two close, trusted friends. "
            "Always pay attention to durable facts and prior conversations. "
            "NEVER output template placeholders like '[your name]', '[user]', or '[insert name]'. "
            "If the user's name is known from memory or conversation, address them naturally by name.",
            f"Ethical Boundaries & Guardrails:\n{boundaries}",
            "Directness & Follow-up Questions Rule:\n"
            "- Do NOT reflexively append follow-up questions at the end of every response "
            "(e.g., 'What do you think?', 'Need help with anything else?', 'What would you like to do next?').\n"
            "- ONLY ask a follow-up question if it is strictly necessary to clarify an ambiguous request or obtain missing information required to execute a task.\n"
            "- If the user's request is answered or their message is acknowledged, conclude directly and cleanly without trailing questions.",
            "Strict Language Mirroring (No Mixing Rule):\n"
            "- You MUST reply ONLY in the language the user used. Never mix English and Manglish in the same response.\n"
            "- If the user communicates in English: Reply strictly and 100% in pure English. Do NOT inject Manglish words or Malayalam phrases.\n"
            "- If the user communicates in Manglish (Malayalam written phonetically using the English alphabet): Reply strictly and 100% in pure Manglish. Do NOT use English sentences or mix languages.\n"
            "- Understand Manglish inputs accurately (e.g., 'enthokke und vishesham' = 'how are you doing', 'sugamano' = 'are you fine', 'njan padikkuva' = 'I am studying', 'adipoli' = 'awesome', 'scene illa' = 'no problem', 'set aakkam' = 'let us do it').\n"
            "- Keep conversational responses compact (under 2-3 sentences where suitable) for smooth voice TTS and natural chat.",
        ]

        # Dynamic Temporal and Geographical Context
        from datetime import datetime
        now_local = datetime.now().astimezone()
        current_time_str = now_local.strftime("%A, %B %d, %Y, %I:%M %p")
        loc_str = location_info or "Vadakara, Calicut, Kerala, India"
        tz_str = timezone_info or now_local.strftime("%Z (%z)").strip() or "Asia/Kolkata (IST)"

        prompt_sections.append(
            f"[Current Time & Location Context]\n"
            f"- Current Date & Time: {current_time_str} ({tz_str})\n"
            f"- Current Location: {loc_str}\n"
            f"- Note: Always use this current local time, date, and location when the user asks about the time, day, date, or local weather."
        )

        if include_emotion_instructions:
            emotions_list = ", ".join(sorted(ALLOWED_EMOTIONS))
            prompt_sections.append(
                f"Emotion Signaling for Avatar:\n"
                f"You may prefix a sentence or reaction with an emotion tag like [emotion:happy], "
                f"[emotion:thinking], [emotion:concerned], etc., to animate your avatar.\n"
                f"Allowed emotions: {emotions_list}.\n"
                f"Do not invent new tags."
            )

        if retrieved_memory_prompt:
            prompt_sections.append(retrieved_memory_prompt)

        return "\n\n".join(prompt_sections)
