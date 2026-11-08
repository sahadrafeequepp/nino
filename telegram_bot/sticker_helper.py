"""Sticker emotion classification, complementary reaction matching, and cache helper for Nino Telegram Bot."""

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("nino.stickers")

# Categorized emotion mapping for standard Telegram emoji tags
EMOTION_CATEGORIES: dict[str, list[str]] = {
    "joy": ["😂", "🤣", "😹", "😆", "😄", "😃", "😀", "😁", "😸", "lol", "haha"],
    "love": ["❤️", "💖", "💕", "🥰", "😍", "😘", "😻", "💓", "💘", "💌", "😚", "😙", "love", "heart"],
    "cool": ["😎", "🕶️", "😏", "🔥", "✨", "🤙", "💅", "🥂", "🆒", "cool", "fire"],
    "agree": ["👍", "👌", "✅", "🙌", "👏", "🤝", "💯", "🫡", "🆗", "yes", "agree"],
    "sad": ["😢", "😭", "🥺", "😿", "💔", "😥", "😓", "😞", "😔", "💧", "sad", "crying"],
    "angry": ["😡", "🤬", "👿", "💢", "😤", "😠", "😾", "angry", "mad"],
    "surprised": ["😱", "😲", "🤯", "😳", "🙀", "😮", "🫢", "👀", "🚨", "shock", "wow"],
    "thinking": ["🤔", "🧐", "🤨", "💭", "❓", "❔", "💡", "think", "hmm"],
    "greeting": ["👋", "🙋", "🙋‍♂️", "🙋‍♀️", "✋", "hi", "hello", "hey"],
    "celebrate": ["🎉", "🥳", "🎊", "🍾", "🎈", "🎁", "party", "cheers"],
    "sleepy": ["😴", "🥱", "💤", "🛌", "😪", "sleep", "tired"],
    "cheeky": ["😜", "😝", "😛", "😋", "🤪", "🤡", "🤭", "👅", "sassy"],
    "shy": ["🙈", "😳", "🫣", "🥺", "shy", "blush"],
}

# Nino's persona-aligned reaction matrix: given an incoming emotion, what emotions should Nino react with?
COMPLEMENTARY_REACTIONS: dict[str, list[str]] = {
    "joy": ["joy", "cool", "celebrate", "cheeky"],
    "love": ["love", "cheeky", "cool", "joy"],
    "cool": ["cool", "agree", "joy", "cheeky"],
    "agree": ["agree", "cool", "joy"],
    "sad": ["love", "sad", "agree"],
    "angry": ["surprised", "cheeky", "thinking", "cool"],
    "surprised": ["surprised", "cool", "joy", "thinking"],
    "thinking": ["thinking", "cool", "agree"],
    "greeting": ["greeting", "joy", "cool"],
    "celebrate": ["celebrate", "joy", "cool"],
    "sleepy": ["sleepy", "greeting", "love"],
    "cheeky": ["cheeky", "cool", "joy"],
    "shy": ["love", "cheeky", "joy"],
}

# Reverse lookup: emoji -> emotion category
EMOJI_TO_EMOTION: dict[str, str] = {}
for category, emojis in EMOTION_CATEGORIES.items():
    for em in emojis:
        EMOJI_TO_EMOTION[em] = category


def detect_emotion(emoji: Optional[str], visual_description: str = "") -> str:
    """Determine the emotion category for an incoming sticker from its emoji and visual context."""
    if emoji:
        # Exact match or substring match
        clean_emoji = emoji.strip()
        if clean_emoji in EMOJI_TO_EMOTION:
            return EMOJI_TO_EMOTION[clean_emoji]
        for em, cat in EMOJI_TO_EMOTION.items():
            if em in clean_emoji:
                return cat

    # Visual description sentiment keyword fallback
    desc_lower = visual_description.lower()
    for cat, keywords in EMOTION_CATEGORIES.items():
        for kw in keywords:
            if kw in desc_lower:
                return cat

    return "joy"


def get_target_emotions(input_emotion: str) -> list[str]:
    """Retrieve prioritized list of response emotions for a given input emotion."""
    return COMPLEMENTARY_REACTIONS.get(input_emotion, ["joy", "cool", "agree"])


def pick_matching_sticker(
    stickers: list[Any],
    target_emotions: list[str],
    incoming_file_unique_id: Optional[str] = None,
) -> Optional[Any]:
    """Pick the most accurate sticker from a sticker set matching target emotions.
    
    1. Searches for stickers in target_emotions (excluding incoming sticker if alternatives exist).
    2. Searches for stickers matching any target emotion's associated emojis.
    3. If none match, picks any positive/expressive sticker from the pack.
    4. Fallback: picks any different sticker from the pack.
    """
    if not stickers:
        return None

    # Gather acceptable emojis for target emotions
    preferred_emojis: set[str] = set()
    for cat in target_emotions:
        for em in EMOTION_CATEGORIES.get(cat, []):
            preferred_emojis.add(em)

    # 1. Exact matches for preferred emotions
    primary_candidates = []
    secondary_candidates = []

    for s in stickers:
        s_emoji = getattr(s, "emoji", "") or ""
        s_uid = getattr(s, "file_unique_id", None)

        is_same = (s_uid is not None and s_uid == incoming_file_unique_id)
        matches_preferred = any(em in s_emoji for em in preferred_emojis)

        if matches_preferred:
            if not is_same:
                primary_candidates.append(s)
            else:
                secondary_candidates.append(s)

    if primary_candidates:
        return random.choice(primary_candidates)
    if secondary_candidates:
        return secondary_candidates[0]

    # 2. General friendly/positive emotions (joy, cool, agree)
    fallback_emojis: set[str] = set()
    for cat in ["joy", "cool", "agree"]:
        for em in EMOTION_CATEGORIES.get(cat, []):
            fallback_emojis.add(em)

    fallback_candidates = [
        s for s in stickers
        if any(em in (getattr(s, "emoji", "") or "") for em in fallback_emojis)
        and getattr(s, "file_unique_id", None) != incoming_file_unique_id
    ]
    if fallback_candidates:
        return random.choice(fallback_candidates)

    # 3. Any different sticker from the pack
    different_stickers = [
        s for s in stickers
        if getattr(s, "file_unique_id", None) != incoming_file_unique_id
    ]
    if different_stickers:
        return random.choice(different_stickers)

    return stickers[0]


class StickerCache:
    """Persistent storage for indexed stickers by emotion for instant, cross-pack fallback replies."""

    def __init__(self, cache_file: str | Path = "bot_saved_data/sticker_cache.json") -> None:
        self.cache_file = Path(cache_file)
        self.cache: dict[str, list[str]] = {}
        self.load()

    def load(self) -> None:
        """Load sticker cache from disk."""
        try:
            if self.cache_file.exists():
                data = json.loads(self.cache_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.cache = data
        except Exception as e:
            logger.warning("Could not read sticker cache (%s), starting fresh", e)
            self.cache = {}

    def save(self) -> None:
        """Save sticker cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to save sticker cache to %s: %s", self.cache_file, e)

    def record_sticker(self, file_id: str, emotion: str) -> None:
        """Cache a sticker file_id under an emotion category."""
        if not file_id:
            return
        if emotion not in self.cache:
            self.cache[emotion] = []
        if file_id not in self.cache[emotion]:
            self.cache[emotion].append(file_id)
            # Cap at 50 per category to prevent unbounded growth
            if len(self.cache[emotion]) > 50:
                self.cache[emotion].pop(0)
            self.save()

    def record_sticker_set(self, sticker_set: Any) -> None:
        """Index all stickers in a set into cache for future responses."""
        stickers = getattr(sticker_set, "stickers", []) or []
        updated = False
        for s in stickers:
            file_id = getattr(s, "file_id", None)
            emoji = getattr(s, "emoji", None)
            if file_id and emoji:
                cat = detect_emotion(emoji)
                if cat not in self.cache:
                    self.cache[cat] = []
                if file_id not in self.cache[cat]:
                    self.cache[cat].append(file_id)
                    if len(self.cache[cat]) > 50:
                        self.cache[cat].pop(0)
                    updated = True
        if updated:
            self.save()

    def get_sticker_for_emotions(self, target_emotions: list[str]) -> Optional[str]:
        """Find a cached sticker file_id matching any of target_emotions."""
        for cat in target_emotions:
            file_ids = self.cache.get(cat, [])
            if file_ids:
                return random.choice(file_ids)
        # Any cached sticker fallback
        for file_ids in self.cache.values():
            if file_ids:
                return random.choice(file_ids)
        return None

