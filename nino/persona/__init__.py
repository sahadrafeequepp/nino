"""Nino Persona & Emotion Signaling Subsystem."""

from nino.persona.emotions import (
    ALLOWED_EMOTIONS,
    EmotionParser,
    strip_emotion_tags,
)
from nino.persona.manager import PersonaManager

__all__ = [
    "PersonaManager",
    "EmotionParser",
    "ALLOWED_EMOTIONS",
    "strip_emotion_tags",
]
