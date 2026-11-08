"""Spoken confirmation flow for sensitive tools in voice interactions."""

from enum import Enum
import logging
import re
from typing import Optional

from nino.tools.confirmation import Confirmation, ConfirmationManager

logger = logging.getLogger("nino.voice.confirmation")


class ConfirmationIntent(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    AMBIGUOUS = "ambiguous"


# Explicit single-phrase or clean affirmative patterns
AFFIRMATIVE_REGEX = re.compile(
    r"^(?:yes|yeah|yep|confirm|approved?|do\s+it|go\s+ahead|proceed|sure|affirmative)$",
    re.IGNORECASE,
)

# Explicit negative patterns
NEGATIVE_REGEX = re.compile(
    r"^(?:no|nope|cancel|stop|don'?t|deny|denied|abort|reject|never\s*mind|do\s+not)$",
    re.IGNORECASE,
)


def parse_spoken_confirmation(text: str) -> ConfirmationIntent:
    """Parse spoken user reply for tool confirmation.

    Strict rules:
    - Only a clear, unambiguous affirmation returns APPROVED.
    - An explicit denial returns DENIED.
    - Any mumbled, uncertain, rambling, or ambiguous speech (e.g. 'maybe',
      'uhh', 'yes but wait', 'what?', 'yes or no') returns AMBIGUOUS and
      MUST NOT count as approval.
    """
    if not text or not text.strip():
        return ConfirmationIntent.AMBIGUOUS

    # Clean punctuation and normalize whitespace
    cleaned = re.sub(r"[^\w\s]", "", text).strip().lower()

    if not cleaned:
        return ConfirmationIntent.AMBIGUOUS

    if AFFIRMATIVE_REGEX.match(cleaned):
        return ConfirmationIntent.APPROVED

    if NEGATIVE_REGEX.match(cleaned):
        return ConfirmationIntent.DENIED

    # Any other speech is considered ambiguous / mumbled
    logger.info("Spoken confirmation input '%s' considered ambiguous / unconfirmed", text)
    return ConfirmationIntent.AMBIGUOUS


class SpokenConfirmationHandler:
    """Handles verbal confirmation prompting and validation for sensitive tool execution."""

    def __init__(self, confirmation_manager: ConfirmationManager):
        self.confirmation_manager = confirmation_manager

    def build_spoken_prompt(self, confirmation: Confirmation) -> str:
        """Generate a concise, clear spoken prompt for the user."""
        tool_name = confirmation.tool_name.replace("_", " ")
        args_summary = ", ".join(f"{k} {v}" for k, v in confirmation.arguments.items()) if confirmation.arguments else ""

        if args_summary:
            action_desc = f"execute {tool_name} with {args_summary}"
        else:
            action_desc = f"execute {tool_name}"

        return (
            f"Nino needs your approval to {action_desc}. "
            f"Please say yes to confirm, or no to cancel."
        )

    def process_spoken_reply(
        self,
        confirmation_id: str,
        spoken_text: str,
    ) -> tuple[bool, str]:
        """Process spoken reply against a pending confirmation.

        Returns:
            (is_approved: bool, feedback_message: str)
        """
        intent = parse_spoken_confirmation(spoken_text)

        if intent == ConfirmationIntent.APPROVED:
            success = self.confirmation_manager.approve(confirmation_id)
            if success:
                logger.info("Spoken confirmation approved for ID '%s'", confirmation_id)
                return True, "Action confirmed. Executing now."
            else:
                return False, "Approval failed because the confirmation expired or is invalid."

        elif intent == ConfirmationIntent.DENIED:
            self.confirmation_manager.deny(confirmation_id)
            logger.info("Spoken confirmation denied for ID '%s'", confirmation_id)
            return False, "Action cancelled as requested."

        else:
            # Ambiguous or mumbled
            self.confirmation_manager.deny(confirmation_id)
            logger.warning(
                "Spoken confirmation for ID '%s' was ambiguous ('%s') and was safely rejected",
                confirmation_id,
                spoken_text,
            )
            return (
                False,
                "Confirmation was unclear or mumbled. For your safety, the action was cancelled.",
            )
