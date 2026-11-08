"""Piper TTS adapter implementation for fast local neural speech synthesis."""

import asyncio
import logging
from pathlib import Path
import re
import threading
from typing import AsyncIterator, Optional

from nino.voice.interfaces import TTSAdapter

logger = logging.getLogger("nino.voice.tts.piper")

HUGGINGFACE_PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

VOICE_CATALOG: dict[str, dict[str, str]] = {
    "kristin": {
        "id": "en_US-kristin-medium",
        "name": "Kristin",
        "accent": "US English",
        "gender": "Female",
        "style": "Conversational, warm, smooth & natural (Recommended)",
    },
    "cori": {
        "id": "en_GB-cori-high",
        "name": "Cori",
        "accent": "British English",
        "gender": "Female",
        "style": "High-fidelity, clear, crisp & articulate",
    },
    "amy": {
        "id": "en_US-amy-medium",
        "name": "Amy",
        "accent": "US English",
        "gender": "Female",
        "style": "Bright, energetic, upbeat American",
    },
    "lessac": {
        "id": "en_US-lessac-medium",
        "name": "Lessac",
        "accent": "US English",
        "gender": "Female",
        "style": "Articulate, balanced standard American",
    },
    "ryan": {
        "id": "en_US-ryan-medium",
        "name": "Ryan",
        "accent": "US English",
        "gender": "Male",
        "style": "Conversational, natural, friendly American male",
    },
    "alan": {
        "id": "en_GB-alan-medium",
        "name": "Alan",
        "accent": "British English",
        "gender": "Male",
        "style": "Calm, distinguished British gentleman",
    },
    "joe": {
        "id": "en_US-joe-medium",
        "name": "Joe",
        "accent": "US English",
        "gender": "Male",
        "style": "Casual, relaxed American male",
    },
}


def resolve_voice_name(name_or_alias: str) -> str:
    """Resolve a voice alias, 1-based index, or full model ID to a Piper model ID."""
    if not name_or_alias:
        return "en_US-kristin-medium"
    cleaned = str(name_or_alias).strip().lower()
    if cleaned.isdigit():
        idx = int(cleaned) - 1
        keys = list(VOICE_CATALOG.keys())
        if 0 <= idx < len(keys):
            return VOICE_CATALOG[keys[idx]]["id"]
    if cleaned in VOICE_CATALOG:
        return VOICE_CATALOG[cleaned]["id"]
    for info in VOICE_CATALOG.values():
        if cleaned == info["id"].lower():
            return info["id"]
    return name_or_alias.strip()


def get_voice_info(name_or_alias: str) -> dict[str, str]:
    """Retrieve metadata for a given voice name or alias."""
    resolved = resolve_voice_name(name_or_alias)
    for alias, info in VOICE_CATALOG.items():
        if info["id"] == resolved:
            return {"alias": alias, **info}
    return {
        "alias": name_or_alias,
        "id": resolved,
        "name": resolved,
        "accent": "Custom",
        "gender": "Unknown",
        "style": "Custom model",
    }


def is_voice_downloaded(name_or_alias: str, model_dir: str | Path = "data/models/piper") -> bool:
    """Check if the given voice model .onnx and .onnx.json files exist on disk."""
    voice_id = resolve_voice_name(name_or_alias)
    md = Path(model_dir)
    return (md / f"{voice_id}.onnx").is_file() and (md / f"{voice_id}.onnx.json").is_file()

# Comprehensive Unicode emoji pattern covering all modern emoji planes, pictographs, dingbats, and joiners
# NOTE: General punctuation (\u2000-\u206f) is explicitly excluded so contractions (curly apostrophes),
# dashes, and ellipses are preserved and normalized rather than stripped.
EMOJI_PATTERN = re.compile(
    "["
    "\U00010000-\U0010ffff"  # All Supplementary Planes (emoticons, pictographs, symbols, flags)
    "\u200d"                 # Zero-width joiner
    "\ufe0e\ufe0f"           # Variation selectors
    "\u20d0-\u20ff"          # Combining marks for symbols
    "\u2100-\u214f"          # Letterlike symbols
    "\u2190-\u21ff"          # Arrows
    "\u2200-\u22ff"          # Mathematical operators
    "\u2300-\u23ff"          # Misc technical symbols
    "\u2460-\u24ff"          # Enclosed alphanumerics
    "\u25a0-\u25ff"          # Geometric shapes
    "\u2600-\u27bf"          # Misc symbols & dingbats
    "\u2900-\u297f"          # Supplemental arrows
    "\u2b00-\u2bff"          # Misc symbols and arrows
    "\u3030\u303d"           # Wavy dash / part alternation
    "\u3297\u3299"           # Circled ideographs
    "]+",
    flags=re.UNICODE,
)

# Text emoticons and Discord/Slack shortcode emojis
SHORTCODE_EMOJI_PATTERN = re.compile(r":[a-zA-Z0-9_+-]+:")
TEXT_EMOTICON_PATTERN = re.compile(
    r"(?<!\w)(?::[-~]?[)D(P/\\|*oOpP3sS]|;[-~]?[)D(P/\\|*oOpP]|<3|</3|XD|xd|xD|-_-|\^_\^|T_T|o_O|O_o)(?!\w)"
)


# Mapping common stage directions / asterisk actions to emotional vocal profiles
ACTION_WORDS = {
    # Playful / laughter / witty amusement
    "chuckle": "playful",
    "chuckles": "playful",
    "chuckling": "playful",
    "laugh": "playful",
    "laughs": "playful",
    "laughing": "playful",
    "giggle": "playful",
    "giggles": "playful",
    "giggling": "playful",
    "snicker": "playful",
    "snickers": "playful",
    "grin": "playful",
    "grins": "playful",
    "wink": "playful",
    "winks": "playful",
    "smiles": "happy",
    "smile": "happy",
    "smiling": "happy",
    # Sarcastic / smug / dry wit / the cool one
    "smirk": "sarcastic",
    "smirks": "sarcastic",
    "smirking": "sarcastic",
    "sigh": "sarcastic",
    "sighs": "sarcastic",
    "sighing": "sarcastic",
    "roll eyes": "sarcastic",
    "rolls eyes": "sarcastic",
    "rolling eyes": "sarcastic",
    "groan": "sarcastic",
    "groans": "sarcastic",
    "shrug": "sarcastic",
    "shrugs": "sarcastic",
    # Whisper / confidential / soft
    "whisper": "whisper",
    "whispers": "whisper",
    "whispering": "whisper",
    "softly": "whisper",
    "leans in": "whisper",
    # Surprise / sudden
    "gasp": "surprised",
    "gasps": "surprised",
    "eyes widen": "surprised",
    # Thinking / hesitant / thoughtful pause
    "pause": "thinking",
    "pauses": "thinking",
    "hesitate": "thinking",
    "hesitates": "thinking",
    "hmm": "thinking",
    "clears throat": "thinking",
    "ponders": "thinking",
    # Concern / gentle support
    "concerned": "concerned",
    "frowns": "concerned",
    "worried": "concerned",
}

# Prosody presets tuned for smooth, effortless, natural conversational speech (no forced straining or erratic duration)
EMOTION_PROSODY = {
    "neutral": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w_scale": 0.8, "volume": 1.0},
    "playful": {"length_scale": 0.98, "noise_scale": 0.667, "noise_w_scale": 0.8, "volume": 1.0},
    "sarcastic": {"length_scale": 1.01, "noise_scale": 0.667, "noise_w_scale": 0.8, "volume": 1.0},
    "happy": {"length_scale": 0.98, "noise_scale": 0.667, "noise_w_scale": 0.8, "volume": 1.0},
    "thinking": {"length_scale": 1.02, "noise_scale": 0.667, "noise_w_scale": 0.8, "volume": 1.0},
    "whisper": {"length_scale": 1.04, "noise_scale": 0.600, "noise_w_scale": 0.75, "volume": 0.85},
    "concerned": {"length_scale": 1.02, "noise_scale": 0.667, "noise_w_scale": 0.8, "volume": 1.0},
    "surprised": {"length_scale": 0.96, "noise_scale": 0.667, "noise_w_scale": 0.8, "volume": 1.0},
}


ACTION_KEYWORDS = (
    "chuckle", "laugh", "smile", "grin", "sigh", "shrug", "nod",
    "wink", "smirk", "giggle", "snicker", "frown", "gasp", "eye",
    "eyes", "glint", "finger", "avatar", "click", "snap", "twirl",
    "look", "glance", "whisper", "lean", "head", "throat", "breath",
    "face", "hand", "walk", "clears", "pause", "hesitate",
)


def extract_actions_and_emotions(text: str) -> tuple[str, str, bool]:
    """Parse out actions enclosed in asterisks (e.g. *chuckles*, *sighs*) and [emotion:tag].

    Crucially, action descriptions are NOT voiced out as literal words; instead their emotional
    intent is mapped and an action pause flag is returned.
    Regular emphasized words (e.g. *super* in 'super smooth') retain the word and strip the asterisks.
    """
    if not text:
        return "", "neutral", False

    detected_emotion = "neutral"
    had_action_pause = False

    # 1. Check explicit [emotion:xxx] tags
    m_tag = re.search(r"\[emotion:([a-zA-Z_-]+)\]", text, re.IGNORECASE)
    if m_tag:
        raw_tag = m_tag.group(1).lower()
        if raw_tag in EMOTION_PROSODY:
            detected_emotion = raw_tag
        text = re.sub(r"\[emotion:[a-zA-Z_-]+\]", "", text, flags=re.IGNORECASE)

    # 2. Check asterisks for stage directions vs emphasis
    def replace_asterisks(match):
        nonlocal detected_emotion, had_action_pause
        inner = match.group(1).strip()
        inner_lower = inner.lower()

        # Check if inner text contains recognized action words / keywords
        is_action = any(k in inner_lower for k in ACTION_KEYWORDS) or any(
            act in inner_lower for act in ACTION_WORDS
        )

        if is_action:
            for act, emo in ACTION_WORDS.items():
                if act in inner_lower:
                    detected_emotion = emo
                    break
            had_action_pause = True
            # Stage direction is NOT voiced out! Insert a natural pause marker
            return " , "

        # Otherwise it was just word emphasis (e.g. *really*, *super*, *bold text*) -> preserve the word!
        return f" {inner} "

    processed = re.sub(r"\*([^*]+)\*", replace_asterisks, text)
    return processed, detected_emotion, had_action_pause


HEADLINE_PAUSE_TOKEN = " @@NINOPAUSE@@ "


def prepare_headings_with_pauses(text: str) -> str:
    """Detect headlines and topic headings, inserting explicit pause tokens before and after.

    Ensures that transitioning from normal text to a headline causes an organic pause,
    and transitioning from a headline to normal body text causes a distinct cadence drop and pause.
    """
    if not text or not text.strip():
        return ""

    t = text

    def clean_heading_text(val: str) -> str:
        s = val.strip()
        # Strip decorative formatting inside the heading
        s = re.sub(r"[*_~`#]+", "", s).strip()
        if not s:
            return ""
        # Preserve inquiries and exclamations
        if s.endswith(("?", "!")):
            term = s[-1]
            s = s[:-1].rstrip(".:;,- ") + term
        else:
            s = s.rstrip(".:;,- ") + "."
        return s

    # 1. HTML headings: <h1>...</h1> through <h6>...</h6>
    t = re.sub(
        r"(?i)<h[1-6][^>]*>(.*?)</h[1-6]>",
        lambda m: f"{HEADLINE_PAUSE_TOKEN}{clean_heading_text(m.group(1))}{HEADLINE_PAUSE_TOKEN}",
        t,
    )

    # 2. Markdown ATX headings (# ... ######)
    t = re.sub(
        r"(?m)^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*$",
        lambda m: f"{HEADLINE_PAUSE_TOKEN}{clean_heading_text(m.group(1))}{HEADLINE_PAUSE_TOKEN}",
        t,
    )

    # 3. Setext headings (underlined with === or ---)
    t = re.sub(
        r"(?m)^([^\n#\-*+`@][^\n]{1,60})\n[ \t]*[=-]{3,}[ \t]*$",
        lambda m: f"{HEADLINE_PAUSE_TOKEN}{clean_heading_text(m.group(1))}{HEADLINE_PAUSE_TOKEN}",
        t,
    )

    # 4. Standalone bold topic heading lines: **Topic** or **Topic:** or - **Topic:**
    t = re.sub(
        r"(?m)^[ \t]*(?:[-*+•·‣⁃]\s+|\d+\.\s+)?(?:\*\*|__)(.+?)(?:\*\*|__):?[ \t]*$",
        lambda m: f"{HEADLINE_PAUSE_TOKEN}{clean_heading_text(m.group(1))}{HEADLINE_PAUSE_TOKEN}",
        t,
    )

    # 5. Lead-in bold topic headings in paragraphs/list items: **Topic:** Normal text begins...
    t = re.sub(
        r"(?m)^[ \t]*([-*+•·‣⁃]\s+|\d+\.\s+)?(?:\*\*|__)([^:*\n]{2,55}:?)(?:\*\*|__):?[ \t]+([^\n]+)$",
        lambda m: f"{HEADLINE_PAUSE_TOKEN}{clean_heading_text(m.group(2))}{HEADLINE_PAUSE_TOKEN}{m.group(3).strip()}",
        t,
    )

    return t


def clean_text_for_speech(text: str) -> str:
    """Preprocess and sanitize text for natural, human-sounding speech synthesis."""
    if not text or not text.strip():
        return ""

    # 0. Detect headlines and format with pause tokens
    t = prepare_headings_with_pauses(text)

    # 1. Strip code blocks completely
    t = re.sub(r"```[\s\S]*?```", "", t)

    # 2. Strip inline code backticks while keeping inner text words
    t = re.sub(r"`([^`]+)`", r"\1", t)

    # 3. Normalize unicode quotation marks, apostrophes, dashes, and ellipses
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("…", "...")
    t = re.sub(r"\s*[—–]+\s*", ", ", t)

    # 3. Markdown links: [display text](http...) -> keep display text only
    t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", t)

    # 4. Remove raw URLs
    t = re.sub(r"https?://\S+", "", t)

    # 5. Remove HTML tags
    t = re.sub(r"<[^>]+>", "", t)

    # 6. Remove markdown headers (#, ##, ###) at start of lines if any left
    t = re.sub(r"(?m)^\s*#{1,6}\s*", "", t)

    # 7. Remove bullet points and numbered list markers at line start
    t = re.sub(r"(?m)^\s*[-*+•·‣⁃]\s+", "", t)
    t = re.sub(r"(?m)^\s*\d+\.\s+", "", t)

    # 8. Remove remaining markdown bold/strike markers
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = re.sub(r"~~([^~]+)~~", r"\1", t)

    # 9. Strip all emojis, pictographs, shortcodes, and text emoticons
    t = EMOJI_PATTERN.sub("", t)
    t = SHORTCODE_EMOJI_PATTERN.sub("", t)
    t = TEXT_EMOTICON_PATTERN.sub("", t)

    # 10. Convert HEADLINE_PAUSE_TOKEN into natural sentence pause dots
    t = t.replace(HEADLINE_PAUSE_TOKEN.strip(), " ... ")
    t = t.replace("@@NINOPAUSE@@", " ... ")

    # 11. Convert symbols into natural conversational words
    t = re.sub(r"\s*&\s*", " and ", t)
    t = re.sub(r"(?<=\w)/(?=\w)", " or ", t)
    t = re.sub(r"\s*[/|\\]\s*", " or ", t)
    t = re.sub(r"(\d+)\s*%", r"\1 percent", t)
    t = re.sub(r"\$(\d+)", r"\1 dollars", t)
    t = re.sub(r"[@#^~_=]+", " ", t)

    # 12. Strip brackets, quotes, braces that sound robotic when read by TTS (preserve apostrophes in contractions like I'm, don't)
    t = re.sub(r'[*\[\]{}<>()"«»]', " ", t)

    # 13. Turn decorative dashes surrounded by spaces into a natural breath pause
    t = re.sub(r"\s+[-–—]+\s+", ", ", t)

    # 14. Normalize punctuation
    t = re.sub(r"\s+([,.:;!?])", r"\1", t)
    t = re.sub(r"\?+", "?", t)
    t = re.sub(r"!+", "!", t)
    t = re.sub(r"(\s*\.\.\.\s*)+", " ... ", t)

    # 15. Clean leading commas or dangling punctuation from stripped prefixes
    t = re.sub(r"^\s*[,;.]\s*", "", t)

    # 16. Normalize multiple whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def segment_with_pauses(text: str) -> list[tuple[str, int]]:
    """Split clean text into speakable units and assign appropriate natural pause durations.

    Pause durations (in milliseconds):
    - Comma (','), Semicolon (';') : 200ms (clause breath pause)
    - Dash ('--')                  : 250ms (conversational pivot)
    - Period ('.'), Exclamation ('!'): 400ms (sentence completion pause)
    - Question mark ('?')          : 450ms (inquiry pause)
    - Ellipsis ('...')             : 550ms (thoughtful dramatic pause)
    """
    if not text:
        return []

    # Split on punctuation marks that are followed by whitespace or end of string
    raw_segments = re.split(r"(?<=[,.;:!?\n])\s+", text)
    result: list[tuple[str, int]] = []

    for seg in raw_segments:
        s = seg.strip()
        if not s:
            continue

        # Determine appropriate pause based on trailing punctuation
        pause_ms = 0
        if s.endswith("..."):
            pause_ms = 550
        elif s.endswith("?"):
            pause_ms = 450
        elif s.endswith(".") or s.endswith("!"):
            pause_ms = 400
        elif s.endswith(",") or s.endswith(";"):
            pause_ms = 200
        elif s.endswith("--") or s.endswith("—"):
            pause_ms = 250
        else:
            # End of segment without punctuation: give default clause pause
            pause_ms = 200

        result.append((s, pause_ms))

    return result


CONJUNCTIONS_AND_CONNECTORS = {
    "and", "but", "or", "so", "yet", "because", "while", "although",
    "since", "which", "that", "when", "where", "if", "with",
    "before", "after", "then", "also", "plus", "however", "though",
}

NON_ENDING_WORDS = {
    # Articles & determiners
    "the", "a", "an", "this", "that", "these", "those",
    # Possessives & Pronouns
    "my", "your", "his", "her", "our", "their", "its",
    "i", "you", "he", "she", "it", "we", "they",
    # Prepositions (prefer pausing BEFORE them, not immediately after)
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "into", "through",
    # Auxiliaries & modals
    "can", "will", "could", "would", "should", "might", "must", "may",
    "is", "are", "was", "were", "be", "been",
}

PAUSE_PUNCTUATION = {",", ";", ":", "—", "--", ".", "!", "?", "…"}


def insert_breather_pauses(text: str, min_words: int = 4, max_words: int = 5) -> str:
    """Insert natural conversational breath pauses after every 4-5 words.

    Human speakers naturally chunk long continuous speech into short phrasal breath
    groups of 4-5 words, taking a subtle breath intake before continuing. This
    function identifies optimal phrasal boundaries (preferring conjunctions and
    clause connectors) and inserts gentle prosodic pause markers while preserving
    short natural sentences and avoiding awkward cuts after articles or prepositions.
    """
    if not text or not text.strip():
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    result_sentences = []

    for s in sentences:
        words = s.split()
        if len(words) <= max_words:
            result_sentences.append(s)
            continue

        new_words = []
        words_since_pause = 0

        for i, w in enumerate(words):
            new_words.append(w)
            words_since_pause += 1

            # If word already ends with pause punctuation, reset counter
            if any(w.endswith(p) for p in PAUSE_PUNCTUATION):
                words_since_pause = 0
                continue

            # Don't place a breather if only 1 word remains before sentence ends
            remaining = len(words) - (i + 1)
            if remaining <= 1:
                continue

            clean_cur = w.lower().strip(".,!?:;\"'()[]")
            next_w = words[i + 1].lower().strip(".,!?:;\"'()[]")

            # Avoid placing a pause right after an article, possessive, preposition, or modal
            if clean_cur in NON_ENDING_WORDS:
                continue

            should_breathe = False
            # 1. Optimal point: after at least min_words (4) if next word is a conjunction / clause connector
            if words_since_pause >= min_words and next_w in CONJUNCTIONS_AND_CONNECTORS:
                should_breathe = True
            # 2. Reached max_words (5)
            elif words_since_pause >= max_words:
                should_breathe = True

            if should_breathe:
                new_words[-1] = w + ","
                words_since_pause = 0

        result_sentences.append(" ".join(new_words))

    return " ".join(result_sentences)


class PiperTTSAdapter(TTSAdapter):
    """Local text-to-speech adapter powered by Piper."""

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        speed: float = 1.0,
        model_dir: str | Path = "data/models/piper",
        use_cuda: bool = False,
    ):
        self.voice = resolve_voice_name(voice)
        self.speed = speed
        self.model_dir = Path(model_dir)
        self.use_cuda = use_cuda
        self._voice_instance = None
        self._sample_rate: int = 22050
        self._synth_lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _resolve_model_paths(self) -> tuple[Path, Path]:
        """Resolve .onnx and .onnx.json paths for the configured voice."""
        direct_path = Path(self.voice)
        if direct_path.is_file() and direct_path.suffix == ".onnx":
            config_path = direct_path.with_suffix(".onnx.json")
            if not config_path.is_file():
                config_path = Path(str(direct_path) + ".json")
            return direct_path, config_path

        onnx_file = self.model_dir / f"{self.voice}.onnx"
        json_file = self.model_dir / f"{self.voice}.onnx.json"
        return onnx_file, json_file

    def download_voice_if_needed(self) -> tuple[Path, Path]:
        """Download stock Piper voice files if not already present."""
        import urllib.request
        onnx_path, json_path = self._resolve_model_paths()

        if onnx_path.exists() and json_path.exists():
            return onnx_path, json_path

        self.model_dir.mkdir(parents=True, exist_ok=True)
        parts = self.voice.split("-")
        if len(parts) >= 3:
            locale = parts[0]
            name = parts[1]
            quality = parts[2]
            lang = locale.split("_")[0]
            base_url = f"{HUGGINGFACE_PIPER_VOICES_URL}/{lang}/{locale}/{name}/{quality}/{self.voice}"
            onnx_url = f"{base_url}.onnx"
            json_url = f"{base_url}.onnx.json"

            logger.info("Downloading Piper voice model '%s' from %s", self.voice, base_url)
            try:
                urllib.request.urlretrieve(onnx_url, onnx_path)
                urllib.request.urlretrieve(json_url, json_path)
                logger.info("Piper voice '%s' downloaded successfully to %s", self.voice, onnx_path)
                return onnx_path, json_path
            except Exception as e:
                logger.warning("Could not auto-download Piper voice '%s': %s", self.voice, e)

        return onnx_path, json_path

    def _ensure_voice(self):
        if self._voice_instance is None:
            import piper
            onnx_path, json_path = self._resolve_model_paths()

            if not onnx_path.exists():
                onnx_path, json_path = self.download_voice_if_needed()

            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"Piper voice model not found at '{onnx_path}'. "
                    f"Download '{self.voice}.onnx' and '{self.voice}.onnx.json' into '{self.model_dir}'."
                )

            logger.info("Loading Piper voice from '%s'", onnx_path)
            self._voice_instance = piper.PiperVoice.load(
                model_path=str(onnx_path),
                config_path=str(json_path) if json_path.exists() else None,
                use_cuda=self.use_cuda,
            )
            if hasattr(self._voice_instance, "config") and hasattr(self._voice_instance.config, "sample_rate"):
                self._sample_rate = self._voice_instance.config.sample_rate

        return self._voice_instance

    def _make_silence(self, ms: int) -> bytes:
        """Generate raw 16-bit PCM silence bytes for the specified duration in ms."""
        if ms <= 0:
            return b""
        num_samples = int(self._sample_rate * (ms / 1000.0))
        return b"\x00\x00" * num_samples

    def _make_breath_pause(
        self,
        ms: int = 450,
        is_question: bool = False,
        is_thoughtful: bool = False,
    ) -> bytes:
        """Generate an organic, natural human breath pause with subtle vocal tract aspiration.

        Instead of mechanical digital zero silence, this models a natural human breathing
        intake and continuous room presence between sentences.
        """
        if ms <= 0:
            return b""

        # Adaptive human pause duration:
        # Questions get a contemplative comprehension pause (~520ms)
        # Thoughtful / ellipsis pauses get ~600ms
        # Normal sentence endings get ~450ms
        target_ms = ms
        if is_thoughtful:
            target_ms = max(ms, 600)
        elif is_question:
            target_ms = max(ms, 520)

        num_samples = int(self._sample_rate * (target_ms / 1000.0))
        if num_samples <= 0:
            return b""

        try:
            import numpy as np

            # Fast bandpass filtered noise modeling natural oral/nasal breath resonance (~1.8kHz)
            white = np.random.normal(0, 1, num_samples).astype(np.float32)
            t_k = np.arange(-15, 16, dtype=np.float32)
            f_c = 1800.0 / self._sample_rate
            f_w = 1000.0 / self._sample_rate
            kernel = 2 * f_w * np.sinc(2 * f_w * t_k) * np.cos(2 * np.pi * f_c * t_k) * np.hamming(31)
            breath = np.convolve(white, kernel, mode="same")

            max_val = float(np.max(np.abs(breath))) or 1.0
            breath = breath / max_val

            # Organic human inhalation profile (asymmetric breath expansion peaked at ~65% before next sentence)
            t = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
            env = np.sin(np.pi * (t ** 0.85)) ** 2

            fade_len = min(int(self._sample_rate * 0.025), num_samples // 4)
            if fade_len > 0:
                env[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                env[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

            # Subtle acoustic room floor + whispery breath intake (-38dB peak level)
            room_floor = 0.002 * (1.0 - env)
            scaled = (breath * env * 0.014 + white * room_floor)
            int16_samples = np.clip(scaled * 32767.0, -32768, 32767).astype(np.int16)
            return int16_samples.tobytes()
        except Exception:
            return b"\x00\x00" * num_samples

    def _build_synthesis_config(self, emotion: str = "neutral"):
        """Construct a Piper SynthesisConfig optimized for humane, organic, and emotional prosody."""
        try:
            from piper.config import SynthesisConfig
            preset = EMOTION_PROSODY.get(emotion, EMOTION_PROSODY["neutral"])
            # Apply base speed scaling
            base_length = preset["length_scale"] / max(0.2, self.speed)
            return SynthesisConfig(
                length_scale=base_length,
                noise_scale=preset["noise_scale"],
                noise_w_scale=preset["noise_w_scale"],
                volume=preset["volume"],
                normalize_audio=True,
            )
        except Exception:
            return None

    def _synthesize_sync(self, text: str, emotion_override: Optional[str] = None) -> bytes:
        """Synthesize text with natural continuous prosody and pauses around headlines."""
        if not text or not text.strip():
            return b""

        # 1. Parse actions/stage directions and determine emotion
        processed_text, detected_emotion, had_action = extract_actions_and_emotions(text)
        active_emotion = emotion_override or detected_emotion

        voice = self._ensure_voice()
        syn_config = self._build_synthesis_config(active_emotion)

        # 2. Insert pause tokens around headlines and topic headings
        segmented_text = prepare_headings_with_pauses(processed_text)
        audio_parts: list[bytes] = []

        pause_token = HEADLINE_PAUSE_TOKEN.strip()
        if pause_token in segmented_text:
            raw_sections = [s.strip() for s in segmented_text.split(pause_token) if s.strip()]
            for idx, sec in enumerate(raw_sections):
                clean_sec = clean_text_for_speech(sec)
                if not clean_sec:
                    continue
                with self._synth_lock:
                    for chunk in voice.synthesize(clean_sec, syn_config=syn_config):
                        if chunk.audio_int16_bytes:
                            audio_parts.append(chunk.audio_int16_bytes)
                        elif chunk._audio_int16_array is not None:
                            audio_parts.append(chunk._audio_int16_array.tobytes())

                # Insert natural breath pause between headline and normal text transitions
                if idx < len(raw_sections) - 1:
                    pause_bytes = self._make_breath_pause(ms=500)
                    if pause_bytes:
                        audio_parts.append(pause_bytes)
        else:
            clean_text = clean_text_for_speech(processed_text)
            if not clean_text:
                return b""

            with self._synth_lock:
                for chunk in voice.synthesize(clean_text, syn_config=syn_config):
                    if chunk.audio_int16_bytes:
                        audio_parts.append(chunk.audio_int16_bytes)
                    elif chunk._audio_int16_array is not None:
                        audio_parts.append(chunk._audio_int16_array.tobytes())

        return b"".join(audio_parts)

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text into raw 16-bit PCM bytes."""
        if not text or not text.strip():
            return b""
        return await asyncio.to_thread(self._synthesize_sync, text)

    async def synthesize_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Synthesize sentences sequentially as they arrive from the text stream with proper pauses."""
        async for sentence in text_stream:
            if not sentence or not sentence.strip():
                continue
            audio_bytes = await self.synthesize(sentence)
            if audio_bytes:
                yield audio_bytes
