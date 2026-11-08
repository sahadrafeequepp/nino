"""Durable fact extraction, deduplication, sensitive data filtering, and superseding."""

import asyncio
import json
import logging
import re
import uuid
from typing import Any

from nino.memory.db import MemoryDB
from nino.memory.vector.base import Embedder, VectorStore
from nino.models.base import LLMAdapter
from nino.security.privacy_gateway import SECRET_PATTERNS, PII_RULES

logger = logging.getLogger("nino.memory.facts")

SENSITIVE_CATEGORIES = {
    "health",
    "medical",
    "finance",
    "finances",
    "financial",
    "banking",
    "national_id",
    "government_id",
    "political",
    "religious",
}

FACT_EXTRACTION_PROMPT = """You are a strict personal fact extractor.
Analyze the user's message and extract durable, permanent personal facts that the user EXPLICITLY stated about themselves.

Rules:
1. ONLY extract facts the user explicitly stated. Never guess, infer personality traits, or extrapolate.
2. If no new personal facts were stated, return an empty list: {"facts": []}.
3. Categorize each fact (e.g. "preferences", "work", "lifestyle", "contacts", "health", "finances", "identity", "education", "family").
4. If a fact explicitly updates or contradicts an existing statement, note what topic it updates in "supersedes_topic".
5. Output strict, valid JSON ONLY. No markdown fences, no explanatory text.

Output format:
{
  "facts": [
    {
      "text": "User works as a senior backend engineer",
      "category": "work",
      "supersedes_topic": "profession"
    }
  ]
}
"""


class FactManager:
    """Extracts, filters, deduplicates, and manages durable personal facts."""

    def __init__(
        self,
        memory_db: MemoryDB,
        vector_store: VectorStore,
        embedder: Embedder,
        sensitive_memory: bool = False,
    ):
        self.memory_db = memory_db
        self.vector_store = vector_store
        self.embedder = embedder
        self.sensitive_memory = sensitive_memory

    def is_sensitive_content(self, text: str, category: str) -> bool:
        """Check if content violates secret, PII, or sensitive memory restrictions."""
        # 1. Never store secrets (API keys, private keys, AWS IDs, JWTs, etc.)
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                logger.warning("Rejected storing fact containing secret pattern.")
                return True

        # 2. Never store payment cards or SSN / national IDs
        for pattern, _ in PII_RULES:
            if pattern.search(text):
                logger.warning("Rejected storing fact containing payment/national ID pattern.")
                return True

        # 3. Check for passwords / credentials
        if re.search(r"\b(password|passwd|pin code|secret key)\s*[:=]\s*\S+", text, re.IGNORECASE):
            return True

        # 4. Filter sensitive categories (health, finances, identity) unless enabled
        if not self.sensitive_memory:
            if category.lower().strip() in SENSITIVE_CATEGORIES:
                logger.info(f"Skipping sensitive category '{category}' because SENSITIVE_MEMORY=false.")
                return True

        return False

    def parse_facts_json(self, raw_output: str) -> list[dict[str, Any]] | None:
        """Parse raw model output into a list of fact dicts, extracting JSON blocks reliably."""
        text = raw_output.strip()

        # 1. Try markdown fenced code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            candidate = match.group(1).strip()
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "facts" in data and isinstance(data["facts"], list):
                    return data["facts"]
                elif isinstance(data, list):
                    return data
            except Exception:
                pass

        # 2. Try whole text direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "facts" in data and isinstance(data["facts"], list):
                return data["facts"]
            elif isinstance(data, list):
                return data
        except Exception:
            pass

        # 3. Try finding first { and last }
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                data = json.loads(text[start_idx : end_idx + 1])
                if isinstance(data, dict) and "facts" in data and isinstance(data["facts"], list):
                    return data["facts"]
            except Exception:
                pass

        return None

    async def extract_facts_from_turn(
        self,
        user_message: str,
        source_message_id: str,
        adapter: LLMAdapter,
    ) -> list[dict[str, Any]]:
        saved_facts = []

        # 1. Deterministic heuristic extraction for high-confidence identity statements
        name_match = re.search(r"\bmy name is ([A-Za-z\s]{2,40})(?:,|\.|$)", user_message, re.IGNORECASE)
        call_match = re.search(r"\b(?:u can|you can)?\s*call me ([A-Za-z]{2,25})\b", user_message, re.IGNORECASE)
        if name_match:
            full_name = name_match.group(1).strip()
            call_name = call_match.group(1).strip() if call_match else ""
            fact_desc = f"User's name is {full_name}" + (f" (prefers to be called {call_name})" if call_name else "")
            saved = await self.add_or_update_fact(text=fact_desc, category="profile", source_message_id=source_message_id)
            if saved:
                saved_facts.append(saved)
        elif call_match:
            call_name = call_match.group(1).strip()
            fact_desc = f"User prefers to be called {call_name}"
            saved = await self.add_or_update_fact(text=fact_desc, category="profile", source_message_id=source_message_id)
            if saved:
                saved_facts.append(saved)

        # Check if message is a likely personal fact candidate before waking up the LLM
        personal_patterns = [
            r"\b(i|i'm|my|mine|me|myself|we|our|ours|us|call me|remember that)\b"
        ]
        is_candidate = any(re.search(pat, user_message, re.IGNORECASE) for pat in personal_patterns)
        if not is_candidate and not (name_match or call_match):
            return saved_facts


        facts = None
        raw = None
        try:
            prompt = [
                {"role": "system", "content": FACT_EXTRACTION_PROMPT},
                {"role": "user", "content": f"User message to extract facts from:\n\"{user_message}\""},
            ]
            raw = await asyncio.wait_for(adapter.chat(prompt), timeout=60.0)
            facts = self.parse_facts_json(raw)

            # Retry once if output was malformed
            if facts is None and raw:
                logger.warning("Fact extraction JSON malformed. Retrying once with strict reminder...")
                retry_prompt = prompt + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Error: output was not valid JSON. Return ONLY the strict JSON object with {\"facts\": [...]}:"},
                ]
                raw_retry = await asyncio.wait_for(adapter.chat(retry_prompt), timeout=45.0)
                facts = self.parse_facts_json(raw_retry)
        except (asyncio.TimeoutError, TimeoutError):
            logger.debug("LLM fact extraction timed out on local CPU model. Skipping background extraction.")
            return saved_facts
        except Exception as e:
            err_msg = str(e).strip() or repr(e)
            logger.warning("LLM fact extraction error: %s", err_msg)
            facts = None

        if facts is None:
            if raw:
                logger.warning("LLM Fact extraction failed after retry.")
            return saved_facts


        for fact_item in facts:
            text = fact_item.get("text", "").strip()
            category = fact_item.get("category", "general").strip().lower()
            if not text:
                continue

            # Sensitive data filter
            if self.is_sensitive_content(text, category):
                continue

            saved_fact = await self.add_or_update_fact(
                text=text,
                category=category,
                source_message_id=source_message_id,
            )
            if saved_fact:
                saved_facts.append(saved_fact)

        return saved_facts

    async def add_or_update_fact(
        self,
        text: str,
        category: str,
        source_message_id: str | None = None,
        origin: str = "pc",
    ) -> dict[str, Any] | None:
        """Insert a fact with semantic deduplication, origin tracking, and superseding."""
        fact_vector = await self.embedder.embed(text)

        # Check existing current facts for exact match or superseding
        existing_matches = await self.vector_store.search(
            query_vector=fact_vector,
            top_k=3,
            entity_type="fact",
        )

        # If an almost identical fact already exists, skip
        for match in existing_matches:
            if match.score > 0.96 and match.metadata.get("category") == category:
                # Same fact already remembered
                return None

        # Determine if this supersedes an existing fact
        superseded_id = None
        for match in existing_matches:
            if match.metadata.get("category") == category:
                is_multi_value_category = category in {
                    "preference", "preferences", "interest", "interests", "hobbies", "skills"
                }
                has_supersede_signal = any(
                    sig in text.lower()
                    for sig in ["instead of", "no longer", "not anymore", "rather than", "changed to", "now "]
                )

                if is_multi_value_category and not has_supersede_signal:
                    # Multi-valued preferences co-exist unless almost identical (>0.92)
                    if match.score > 0.92:
                        superseded_id = match.entity_id
                        break
                else:
                    if match.score > 0.82:
                        superseded_id = match.entity_id
                        break

        fact_id = str(uuid.uuid4())
        encrypted_text = self.memory_db.encrypt_text(text)

        with self.memory_db.get_connection() as conn:
            if superseded_id:
                conn.execute(
                    "UPDATE facts SET is_current = 0, superseded_by = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (fact_id, superseded_id),
                )
            conn.execute(
                """
                INSERT INTO facts (id, text, category, source_message_id, is_current, origin, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (fact_id, encrypted_text, category, source_message_id, origin),
            )

        if superseded_id:
            # Remove superseded fact from active vector store
            await self.vector_store.delete(f"fact_{superseded_id}")

        # Index new current fact in vector store
        await self.vector_store.upsert(
            id=f"fact_{fact_id}",
            vector=fact_vector,
            entity_type="fact",
            entity_id=fact_id,
            metadata={"category": category, "text": text, "origin": origin},
        )

        return {"id": fact_id, "text": text, "category": category, "origin": origin, "superseded_by": None}


    def list_facts(
        self,
        category: str | None = None,
        current_only: bool = True,
        channel: str | None = None,
    ) -> list[dict[str, Any]]:
        """List stored facts with optional filtering and channel boundaries.

        If channel == 'telegram', only returns:
        - Facts created from Telegram (origin = 'telegram')
        - Public/identity/preference profile facts
        Explicitly hides confidential personal memories from PC conversations.
        """
        query = "SELECT id, text, category, source_message_id, is_current, superseded_by, origin, created_at FROM facts"
        clauses = []
        params = []
        if current_only:
            clauses.append("is_current = 1")
        if category:
            clauses.append("category = ?")
            params.append(category.lower().strip())
        if channel == "telegram":
            clauses.append(
                "(origin = 'telegram' OR category IN ('profile', 'identity', 'preference', 'preferences', 'name', 'location', 'work'))"
            )

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"


        with self.memory_db.get_connection() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()

        results = []
        for r in rows:
            results.append(
                {
                    "id": r["id"],
                    "text": self.memory_db.decrypt_text(r["text"]),
                    "category": r["category"],
                    "source_message_id": r["source_message_id"],
                    "is_current": bool(r["is_current"]),
                    "superseded_by": r["superseded_by"],
                    "created_at": r["created_at"],
                }
            )
        return results

    async def update_fact(self, fact_id: str, new_text: str | None = None, new_category: str | None = None) -> bool:
        """Edit an existing fact and update its vector embedding."""
        with self.memory_db.get_connection() as conn:
            cur = conn.execute("SELECT id, text, category FROM facts WHERE id = ?", (fact_id,))
            row = cur.fetchone()
            if not row:
                return False

            current_text = self.memory_db.decrypt_text(row["text"])
            current_category = row["category"]

            final_text = new_text.strip() if new_text else current_text
            final_category = new_category.strip().lower() if new_category else current_category

            encrypted = self.memory_db.encrypt_text(final_text)
            conn.execute(
                "UPDATE facts SET text = ?, category = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (encrypted, final_category, fact_id),
            )

        # Re-embed if text changed
        if new_text:
            vector = await self.embedder.embed(final_text)
            await self.vector_store.upsert(
                id=f"fact_{fact_id}",
                vector=vector,
                entity_type="fact",
                entity_id=fact_id,
                metadata={"category": final_category, "text": final_text},
            )
        return True

    async def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact from SQLite and vector store atomically."""
        with self.memory_db.get_connection() as conn:
            cur = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            deleted = cur.rowcount > 0

        if deleted:
            await self.vector_store.delete(f"fact_{fact_id}")
        return deleted

    async def delete_all(self) -> int:
        """Purge all facts from SQLite and vector store."""
        with self.memory_db.get_connection() as conn:
            cur = conn.execute("DELETE FROM facts")
            count = cur.rowcount
        await self.vector_store.delete_by_prefix("fact_")
        return count
