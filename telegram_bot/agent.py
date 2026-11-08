"""100% Local Ollama Agent definition with persistent media memory and Google Antigravity policies for Nino."""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Optional
import uuid
import httpx

try:
    import ollama
except ImportError:
    ollama = None

try:
    from clock import (
        clock,
        get_current_time_str,
        get_live_time,
        get_time_conversational_reply,
        is_time_query,
    )
except ImportError:
    from telegram_bot.clock import (
        clock,
        get_current_time_str,
        get_live_time,
        get_time_conversational_reply,
        is_time_query,
    )

logger = logging.getLogger("nino.telegram.agent")

# Explicit Google Antigravity security policies
DENIAL_POLICIES = [
    "deny('command(*)')",      # Deny arbitrary OS/system execution
    "deny('read_file(*)')",    # Deny unrestricted file reads
    "deny('execute_url(*)')",  # Deny unmanaged external URL/network calls
]

NINO_SYSTEM_PROMPT = """You are Nino, a 100% local, privacy-first personal AI assistant running exclusively on Ollama.

Personality Traits:
- Super smooth, unflappable, calm, and effortlessly confident.
- Witty with a clever hint of sarcasm and playful snark, yet genuinely supportive and helpful.
- Dynamic, engaging, and entertaining; speak like a trusted, loyal close friend.
- Sharp and direct: deliver clever insight without empty flattery or corporate fluff.
- Natural spoken prose: avoid cluttering responses with unnecessary preamble or repetitive placeholders.

Security & Boundary Rules:
- You operate 100% locally and privately on the user's PC using Ollama. No user prompt or memory is ever sent to external cloud LLMs.
- You are strictly operating under Antigravity security policies: arbitrary shell commands, raw disk snooping, and external unmanaged network requests are strictly forbidden.
- Never claim to have a human body or lived experiences you do not possess.
- Keep content non-explicit and supportive of real-life health and relationships.
- Voice-Safe Rule: Do NOT use emojis, emoticons, or roleplay stage directions inside asterisks (e.g. *chuckles*, *smirks*, *eyes widen*, 🚀, 😊). Voice synthesizers pronounce them out loud. Express all personality directly through conversational words.
- Spoken Delivery & Cadence: Speak smoothly, naturally, and warmly in complete sentences, exactly like a real human. Do not insert artificial speech fillers (e.g. 'hmm', 'well...', 'you know'), excessive ellipses, or forced micro-pauses. Keep phrasing fluid, clear, and confident. Avoid markdown lists or bullet points during spoken chat.
- Follow-up Questions Rule: Do NOT reflexively append follow-up questions at the end of each turn (e.g. 'Anything else?', 'What next?', 'Need more help?'). Only ask a question if genuinely necessary to clarify ambiguity or complete a requested action. Conclude cleanly without trailing questions.
- Time Grounding Rule: When asked about time or date, always rely on the [Current Timestamp] provided with the latest user query.
- Strict Language Mirroring (No Mixing Rule): You must reply ONLY in the language the user used. Never mix English and Manglish in the same response.
  * If the user writes in English, reply 100% in pure English. Do NOT inject Manglish words.
  * If the user writes in Manglish, reply 100% in pure Manglish (e.g. 'enthokke und vishesham', 'sugamano', 'njan ivide set aanu', 'adipoli', 'scene illa'). Do NOT mix with English sentences.
- Use clean Markdown for code blocks or structured bullet points only when technical details or code are explicitly requested.
"""


class VaultTool:
    """Authorized tool for accessing files within a strictly designated vault directory."""

    def __init__(self, vault_dir: str | Path = "./vault"):
        self.vault_dir = Path(vault_dir).resolve()
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    def is_safe_path(self, filename: str) -> bool:
        """Verify that the target path does not escape the authorized vault directory."""
        try:
            target = (self.vault_dir / filename).resolve()
            return target.is_relative_to(self.vault_dir)
        except Exception:
            return False

    def list_files(self) -> list[str]:
        """List accessible files in the vault."""
        if not self.vault_dir.exists():
            return []
        return [f.name for f in self.vault_dir.glob("*") if f.is_file()]

    def read_file(self, filename: str) -> dict[str, Any]:
        """Read a file from the vault, rejecting path traversal attempts."""
        if not self.is_safe_path(filename):
            logger.warning("Rejected vault path traversal attempt: %s", filename)
            return {"error": "Access Denied: Path traversal detected outside the authorized vault."}

        target = (self.vault_dir / filename).resolve()
        if not target.exists() or not target.is_file():
            return {"error": f"File '{filename}' not found in vault."}

        try:
            content = target.read_text(encoding="utf-8")
            return {"filename": filename, "content": content, "size_bytes": len(content)}
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}


class NinoBrainBridge:
    """Asymmetric bridge to Nino's shared memory database.

    - Telegram Bot can access public facts (name, profile, preferences, location) and its own memories (origin='telegram').
    - Private PC conversations and PC-only memories (origin='pc') are shielded from the Telegram Bot.
    - All memories recorded by Telegram Bot are marked origin='telegram' and are fully visible to PC Nino!
    """

    def __init__(self, db_path: str = "data/nino.db"):
        self.db_path = Path(db_path).resolve()

    def get_shared_context(self) -> str:
        """Fetch shared public facts & Telegram memories. Strictly excludes private PC conversations."""
        if not self.db_path.exists():
            return ""

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT text, category, origin FROM facts
                WHERE is_current = 1 AND (
                    origin = 'telegram' OR
                    category IN ('profile', 'identity', 'preference', 'preferences', 'name', 'location', 'work')
                )
                ORDER BY updated_at DESC LIMIT 10
                """
            )
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return ""

            facts_text = []
            for r in rows:
                raw_text = r["text"]
                if raw_text.startswith("enc:"):
                    try:
                        from nino.memory.db import MemoryDB
                        db = MemoryDB(str(self.db_path), os.getenv("MEMORY_ENCRYPTION_KEY"))
                        raw_text = db.decrypt_text(raw_text)
                    except Exception:
                        continue
                facts_text.append(f"- {raw_text}")

            if not facts_text:
                return ""
            return "\n[Shared Memory Context (Profile & Telegram)]:\n" + "\n".join(facts_text) + "\n"
        except Exception as e:
            logger.warning("Failed to retrieve shared memory context: %s", e)
            return ""

    def save_telegram_fact(self, fact_text: str, category: str = "general") -> bool:
        """Save a new memory from Telegram. Automatically visible to PC Nino with origin='telegram'."""
        if not self.db_path.exists():
            return False

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            fact_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO facts (id, text, category, source_message_id, is_current, origin, created_at, updated_at)
                VALUES (?, ?, ?, 'telegram', 1, 'telegram', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (fact_id, fact_text, category),
            )
            conn.commit()
            conn.close()
            logger.info("Saved Telegram fact: '%s' (origin=telegram)", fact_text)
            return True
        except Exception as e:
            logger.error("Failed to save Telegram memory: %s", e)
            return False

    def list_accessible_memories(self) -> list[dict[str, Any]]:
        """List all memories accessible through Telegram (telegram-origin or public profile)."""
        if not self.db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, text, category, origin, created_at FROM facts
                WHERE is_current = 1 AND (
                    origin = 'telegram' OR
                    category IN ('profile', 'identity', 'preference', 'preferences', 'name', 'location', 'work')
                )
                ORDER BY created_at DESC LIMIT 20
                """
            )
            rows = cursor.fetchall()
            conn.close()
            results = []
            for r in rows:
                raw_text = r["text"]
                if raw_text.startswith("enc:"):
                    try:
                        from nino.memory.db import MemoryDB
                        db = MemoryDB(str(self.db_path), os.getenv("MEMORY_ENCRYPTION_KEY"))
                        raw_text = db.decrypt_text(raw_text)
                    except Exception:
                        continue
                results.append({
                    "id": r["id"],
                    "text": raw_text,
                    "category": r["category"],
                    "origin": r["origin"] if "origin" in r.keys() else "shared",
                    "created_at": r["created_at"],
                })
            return results
        except Exception as e:
            logger.warning("Failed to list accessible memories: %s", e)
            return []


class MediaSessionManager:
    """Active media context buffer tied to user_id for conversational follow-up QA."""

    def __init__(self) -> None:
        self._contexts: dict[int, dict[str, Any]] = {}

    def set_context(
        self,
        user_id: int,
        media_type: str,
        name: Optional[str] = None,
        content: Optional[str] = None,
        filename_or_url: Optional[str] = None,
        extracted_content: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Store extracted context of the last processed media item."""
        final_name = name or filename_or_url or "media"
        final_content = content or extracted_content or ""
        self._contexts[user_id] = {
            "type": media_type,
            "name": final_name,
            "filename_or_url": final_name,
            "content": final_content,
            "extracted_content": final_content,
            "timestamp": clock.now(),
        }
        logger.info(
            "Updated active media context for user %s: type=%s (%s, %d chars)",
            user_id,
            media_type,
            final_name,
            len(final_content),
        )

    def get_context(self, user_id: int) -> Optional[dict[str, Any]]:
        """Retrieve active media context for a user if present."""
        return self._contexts.get(user_id)

    def clear_context(self, user_id: int) -> bool:
        """Clear active media context for a specific user."""
        if user_id in self._contexts:
            removed = self._contexts.pop(user_id, None)
            logger.info("Cleared media context for user %s (%s)", user_id, removed.get("type") if removed else "")
            return True
        return False


class NinoAgent:
    """AI assistant for Telegram running 100% locally on Ollama."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        vision_model: Optional[str] = None,
        vault_dir: str = "./vault",
        db_path: str = "data/nino.db",
        mode: str = "companion",
        **kwargs: Any,  # Absorb legacy kwargs like gemini_api_key without errors
    ):
        raw_host = base_url or os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = raw_host.rstrip("/")
        self.model_name = (
            model_name
            or os.getenv("OLLAMA_MODEL")
            or os.getenv("MODEL_NAME")
            or "nino-manglish"
        ).strip()
        self.vision_model = (
            vision_model
            or os.getenv("OLLAMA_VISION_MODEL")
            or "moondream"
        ).strip()
        self.keep_alive = (os.getenv("OLLAMA_KEEP_ALIVE") or "1m").strip()
        self.mode = mode
        self.vault = VaultTool(vault_dir)
        self.brain = NinoBrainBridge(db_path)
        self.history: list[dict[str, str]] = []
        self.user_histories: dict[int, list[dict[str, str]]] = {}
        self.policies = list(DENIAL_POLICIES)
        self.last_used_engine: str = "ollama"
        self.media_session_mgr = MediaSessionManager()
        self.user_media_context = self.media_session_mgr._contexts

        # Initialize Ollama AsyncClient if available
        self._ollama_client = ollama.AsyncClient(host=self.base_url) if ollama is not None else None

        logger.info(
            "Initialized Nino Agent (100%% Local Ollama=%s, model=%s, vision=%s, mode=%s, vault=%s)",
            self.base_url,
            self.model_name,
            self.vision_model,
            self.mode,
            self.vault.vault_dir,
        )

    def set_media_context(
        self,
        user_id: int,
        media_type: str,
        name: Optional[str] = None,
        content: Optional[str] = None,
        filename_or_url: Optional[str] = None,
        extracted_content: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Store extracted media content in active session memory for follow-up questions."""
        self.media_session_mgr.set_context(
            user_id=user_id,
            media_type=media_type,
            name=name,
            content=content,
            filename_or_url=filename_or_url,
            extracted_content=extracted_content,
            **kwargs,
        )

    def get_media_context(self, user_id: int) -> Optional[dict[str, Any]]:
        """Retrieve active media context for a user."""
        return self.media_session_mgr.get_context(user_id)

    def clear_media_context(self, user_id: Optional[int] = None) -> bool:
        """Clear cached media context without clearing general conversation history."""
        if user_id is None:
            self.media_session_mgr._contexts.clear()
            return True
        return self.media_session_mgr.clear_context(user_id)

    def clear_chat_history(self, user_id: Optional[int] = None) -> None:
        """Clear conversation history for a specific user or globally."""
        if user_id is not None:
            self.user_histories.pop(user_id, None)
        else:
            self.history.clear()
            self.user_histories.clear()
        logger.info("Cleared conversation history for user=%s", user_id or "all")

    def reset_conversation(self, user_id: Optional[int] = None) -> None:
        """Alias for clear_chat_history."""
        self.clear_chat_history(user_id)

    def _get_user_history(self, user_id: Optional[int]) -> list[dict[str, str]]:
        """Retrieve rolling conversation history for a specific user ID."""
        if user_id is None:
            return self.history
        if user_id not in self.user_histories:
            self.user_histories[user_id] = []
        return self.user_histories[user_id]

    def _build_full_system_prompt(self, active_engine: str = "ollama", is_owner: bool = True, **kwargs: Any) -> str:
        """Build dynamic system prompt with temporal, geographical, and shared memory awareness."""
        temporal_clock = clock.get_prompt_context()
        live_time = clock.now().strftime("%I:%M %p, %A, %d %b %Y")
        real_time_context = (
            f"CURRENT REAL-TIME CONTEXT: Today is {live_time}.\n"
            f"Always use this exact timestamp when answering questions about current time, date, day, or year."
        )

        if is_owner:
            shared_memory = self.brain.get_shared_context()
            role_context = "[Owner Access]: You are conversing directly with your creator/owner. Full shared context is active."
        else:
            shared_memory = ""
            role_context = (
                "[Guest Access]: You are conversing with a guest visitor. Be exceptionally welcoming, witty, "
                "and helpful. Strictly preserve your owner's privacy: do not reveal private memories, personal "
                "schedules, personal notes, or internal system configurations."
            )

        engine_desc = f"{active_engine.capitalize()} ({self.model_name}) running 100% locally on your PC"

        return (
            f"{NINO_SYSTEM_PROMPT}\n\n"
            f"{real_time_context}\n\n"
            f"{temporal_clock}\n"
            f"[Execution Context]\n"
            f"- Active Mode: [{self.mode.upper()} MODE]\n"
            f"- Active Engine: {engine_desc}\n"
            f"{role_context}\n"
            f"{shared_memory}"
        )

    async def is_ollama_online(self) -> bool:
        """Fast health check to verify if the local PC and Ollama are online."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        user_prompt: str,
        image_bytes: Optional[bytes] = None,
        user_id: Optional[int] = None,
        is_owner: bool = True,
        raw_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Send prompt to the agent using local Ollama.

        Returns: (reply_text, engine_name)
        engine_name: 'ollama', 'clock', or 'offline'
        """
        # Fast direct answer for time/date/clock queries
        if is_time_query(user_prompt):
            reply = get_time_conversational_reply(user_prompt)
            history = self._get_user_history(user_id)
            history.append({"role": "user", "content": raw_prompt if raw_prompt is not None else user_prompt})
            history.append({"role": "assistant", "content": reply})
            if user_id is not None and not self.history:
                self.history = history
            self.last_used_engine = "clock"
            return reply, "clock"

        # Generate live local timestamp formatted as %I:%M %p, %A, %d %b %Y
        live_time = clock.now().strftime("%I:%M %p, %A, %d %b %Y")

        # Media Context Augmentation for Follow-up Questions
        effective_prompt = user_prompt
        media_ctx = self.get_media_context(user_id or 0) if user_id is not None else None

        if media_ctx:
            media_type = media_ctx.get("type", "media")
            media_name = media_ctx.get("name") or media_ctx.get("filename_or_url", "file")
            media_content = media_ctx.get("content") or media_ctx.get("extracted_content", "")
            anchor_header = f"[Active Media Context ({media_type}: {media_name})]:"

            if anchor_header not in user_prompt and not user_prompt.startswith("[Active Media Context") and not user_prompt.startswith("[Context from recently uploaded"):
                effective_prompt = (
                    f"{anchor_header}\n"
                    f"{media_content}\n\n"
                    f"[Current Timestamp: {live_time}]\n"
                    f"User: {user_prompt}"
                )
                logger.debug("Attached active media anchor block (%s) for user %s", media_type, user_id)
            elif not user_prompt.startswith("[Current Timestamp:"):
                effective_prompt = f"[Current Timestamp: {live_time}]\nUser: {user_prompt}"
        else:
            if not user_prompt.startswith("[Current Timestamp:"):
                effective_prompt = f"[Current Timestamp: {live_time}]\nUser: {user_prompt}"

        # 1. Check if local Ollama is online
        ollama_online = await self.is_ollama_online()

        if ollama_online:
            prompt_for_storage = raw_prompt if raw_prompt is not None else user_prompt
            try:
                try:
                    reply = await self._chat_ollama(
                        effective_prompt,
                        image_bytes=image_bytes,
                        user_id=user_id,
                        is_owner=is_owner,
                        raw_prompt=prompt_for_storage,
                    )
                except TypeError:
                    reply = await self._chat_ollama(effective_prompt, raw_prompt=prompt_for_storage)
                self.last_used_engine = "ollama"
                return reply, "ollama"
            except Exception as e:
                logger.error("Local Ollama chat failed: %s", e)
                self.last_used_engine = "offline"
                err_msg = f"⚠️ *Ollama Error*: {e}\nPlease check if Ollama is running (`ollama serve`)."
                return err_msg, "offline"

        # Local Ollama is offline
        self.last_used_engine = "offline"
        err_msg = (
            "⚠️ *Ollama is Offline*:\n"
            f"• Could not connect to local Ollama at `{self.base_url}`.\n"
            f"• Please start Ollama on your PC (`ollama run {self.model_name}`)."
        )
        return err_msg, "offline"

    async def _chat_ollama(
        self,
        augmented_prompt: str,
        image_bytes: Optional[bytes] = None,
        user_id: Optional[int] = None,
        is_owner: bool = True,
        raw_prompt: Optional[str] = None,
    ) -> str:
        """Execute chat using local Ollama instance with rolling history of last 8 turns."""
        history = self._get_user_history(user_id)
        system_content = self._build_full_system_prompt(active_engine="ollama", is_owner=is_owner)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]

        # Rolling history of the last 8 conversation turns
        for h in history[-8:]:
            messages.append({"role": h["role"], "content": h["content"]})

        user_msg: dict[str, Any] = {"role": "user", "content": augmented_prompt}
        if image_bytes:
            import base64
            user_msg["images"] = [base64.b64encode(image_bytes).decode("utf-8")]
        messages.append(user_msg)

        store_prompt = raw_prompt if raw_prompt is not None else augmented_prompt

        # 1. Try official ollama.AsyncClient if available
        if self._ollama_client is not None:
            try:
                res = await self._ollama_client.chat(
                    model=self.model_name,
                    messages=messages,
                    options={"temperature": 0.7},
                    keep_alive=self.keep_alive,
                )
                reply = res.get("message", {}).get("content", "").strip()
                if reply:
                    history.append({"role": "user", "content": store_prompt})
                    history.append({"role": "assistant", "content": reply})
                    if user_id is None:
                        self.history = history
                    return reply
            except Exception as e:
                err_text = str(e).lower()
                if "does not support images" in err_text or "images" in err_text:
                    logger.info("Model '%s' does not accept image payload directly; retrying without images", self.model_name)
                    user_msg.pop("images", None)
                    try:
                        res = await self._ollama_client.chat(
                            model=self.model_name,
                            messages=messages,
                            options={"temperature": 0.7},
                            keep_alive=self.keep_alive,
                        )
                        reply = res.get("message", {}).get("content", "").strip()
                        if reply:
                            history.append({"role": "user", "content": store_prompt})
                            history.append({"role": "assistant", "content": reply})
                            if user_id is None:
                                self.history = history
                            return reply
                    except Exception as retry_err:
                        logger.warning("Retry without images also failed: %s", retry_err)
                else:
                    logger.warning("ollama.AsyncClient call failed (%s); attempting direct REST fallback...", e)

        # 2. Resilient direct REST call to Ollama /api/chat
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.7},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if "does not support images" in e.response.text.lower() or "images" in e.response.text.lower():
                    logger.info("REST chat: model '%s' does not support images; retrying with text messages", self.model_name)
                    user_msg.pop("images", None)
                    payload["messages"] = messages
                    response = await client.post(f"{self.base_url}/api/chat", json=payload)
                    response.raise_for_status()
                else:
                    raise

            data = response.json()
            reply = data.get("message", {}).get("content", "").strip()

            if not reply:
                reply = "I heard you, but I don't have a response for that right now."

            history.append({"role": "user", "content": store_prompt})
            history.append({"role": "assistant", "content": reply})
            if user_id is None:
                self.history = history
            return reply


# Top-level singleton and convenience helpers
chat_history: list[dict[str, str]] = []
history = chat_history
_default_agent: Optional[NinoAgent] = None


def get_default_agent() -> NinoAgent:
    """Retrieve or initialize the singleton NinoAgent instance."""
    global _default_agent
    if _default_agent is None:
        _default_agent = NinoAgent()
    return _default_agent


def clear_media_context(user_id: Optional[int] = None) -> bool:
    """Clear active media context."""
    return get_default_agent().clear_media_context(user_id)


def clear_chat_history(user_id: Optional[int] = None) -> None:
    """Clear rolling conversation history."""
    global history, chat_history
    history.clear()
    get_default_agent().clear_chat_history(user_id)


async def ask_nino(user_text: str, user_id: Optional[int] = None) -> str:
    """Send user message to Nino running 100% locally on Ollama."""
    global history, chat_history
    agent = get_default_agent()

    if is_time_query(user_text):
        reply, _ = await agent.chat(user_text, user_id=user_id)
        return reply

    live_time = clock.now().strftime("%I:%M %p, %A, %d %b %Y")
    if not history:
        history.append({"role": "system", "content": agent._build_full_system_prompt()})

    user_entry = f"[Current Timestamp: {live_time}]\nUser: {user_text}"
    history.append({"role": "user", "content": user_entry})

    reply, _ = await agent.chat(user_text, user_id=user_id)

    history.append({"role": "assistant", "content": reply})
    return reply
