"""Memory tools for explicitly storing, searching, and managing durable facts in SQLite."""

import logging
from typing import Any
from pydantic import BaseModel, Field

from nino.memory.facts import FactManager
from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier

logger = logging.getLogger("nino.tools.memory_tools")


class RememberFactInput(BaseModel):
    fact: str = Field(..., description="The personal fact, preference, or detail to permanently remember about the user.")
    category: str = Field(
        default="general",
        description="Category of the fact (e.g. 'preference', 'profile', 'work', 'routine', 'family', 'general').",
    )


class RememberFactTool(BaseTool):
    name = "remember_fact"
    description = (
        "Permanently store a personal fact, preference, or detail about the user into local encrypted memory "
        "so it is retained across all conversations and restarts without being lost."
    )
    args_schema = RememberFactInput
    tier = PermissionTier.REVERSIBLE

    def __init__(self, fact_manager: FactManager):
        self.fact_manager = fact_manager

    async def run(self, fact: str, category: str = "general") -> dict[str, Any]:
        fact_clean = fact.strip()
        cat_clean = category.strip().lower() or "general"

        if self.fact_manager.is_sensitive_content(fact_clean, cat_clean):
            return {
                "status": "rejected",
                "message": "Fact contains sensitive credentials, secrets, or restricted financial data and cannot be stored.",
            }

        saved = await self.fact_manager.add_or_update_fact(text=fact_clean, category=cat_clean)
        if not saved:
            return {
                "status": "already_exists",
                "message": f"This fact is already saved in memory: '{fact_clean}'.",
            }

        return {
            "status": "success",
            "fact_id": saved.get("id"),
            "category": cat_clean,
            "fact": fact_clean,
            "message": f"Successfully remembered permanently: '{fact_clean}'.",
        }


class RecallFactsInput(BaseModel):
    query: str = Field(..., description="Search query, keyword, or topic to search for in saved memories.")
    category: str | None = Field(default=None, description="Optional category filter (e.g. 'preference', 'work', 'profile').")
    limit: int = Field(default=5, description="Maximum number of facts to recall.")


class RecallFactsTool(BaseTool):
    name = "recall_facts"
    description = (
        "Search and recall permanently saved facts, preferences, and personal details about the user from local encrypted memory."
    )
    args_schema = RecallFactsInput
    tier = PermissionTier.READ_ONLY

    def __init__(self, fact_manager: FactManager):
        self.fact_manager = fact_manager

    async def run(self, query: str, category: str | None = None, limit: int = 5) -> dict[str, Any]:
        query_clean = query.strip()
        matches = []
        try:
            query_vec = await self.fact_manager.embedder.embed(query_clean)
            matches = await self.fact_manager.vector_store.search(
                query_vector=query_vec,
                top_k=limit,
                entity_type="fact",
            )
        except Exception as e:
            logger.warning(f"Vector search failed during recall_facts, falling back to SQLite: {e}")

        retrieved = []
        seen_ids = set()

        for m in matches:
            meta = m.metadata
            cat = meta.get("category", "")
            if category and cat.lower() != category.lower().strip():
                continue
            seen_ids.add(m.entity_id)
            retrieved.append({
                "id": m.entity_id,
                "text": meta.get("text", ""),
                "category": cat,
                "score": round(m.score, 3),
            })

        all_facts = self.fact_manager.list_facts(category=category, current_only=True)
        q_lower = query_clean.lower()
        for f in all_facts:
            if f["id"] in seen_ids:
                continue
            if any(term in f["text"].lower() for term in q_lower.split()):
                retrieved.append({
                    "id": f["id"],
                    "text": f["text"],
                    "category": f["category"],
                    "score": 1.0,
                })
                seen_ids.add(f["id"])
                if len(retrieved) >= limit:
                    break

        return {
            "query": query_clean,
            "count": len(retrieved),
            "facts": retrieved,
            "message": f"Found {len(retrieved)} relevant permanent memories." if retrieved else "No matching permanent memories found.",
        }


class ForgetFactInput(BaseModel):
    fact_id: str = Field(..., description="The ID of the fact to remove from memory.")


class ForgetFactTool(BaseTool):
    name = "forget_fact"
    description = "Delete or forget a specific permanent memory from local storage by its ID."
    args_schema = ForgetFactInput
    tier = PermissionTier.SENSITIVE

    def __init__(self, fact_manager: FactManager):
        self.fact_manager = fact_manager

    async def run(self, fact_id: str) -> dict[str, Any]:
        deleted = await self.fact_manager.delete_fact(fact_id.strip())
        if deleted:
            return {"status": "success", "message": f"Permanently removed memory {fact_id}."}
        return {"status": "not_found", "message": f"Memory {fact_id} not found."}
