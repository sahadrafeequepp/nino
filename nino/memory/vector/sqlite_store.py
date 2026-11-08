"""SQLite-based vector store implementation with cosine similarity search."""

import json
import math
import struct
from typing import Any
import sqlite3

from nino.memory.db import MemoryDB
from nino.memory.vector.base import SearchResult, VectorStore


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class SQLiteVectorStore(VectorStore):
    """Vector store using SQLite for atomic persistence with facts and documents."""

    def __init__(self, memory_db: MemoryDB):
        self.memory_db = memory_db

    async def upsert(
        self,
        id: str,
        vector: list[float],
        entity_type: str,
        entity_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a vector entry."""
        # Pack floats into bytes
        blob = struct.pack(f"{len(vector)}f", *vector)
        meta_json = json.dumps(metadata)

        with self.memory_db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO vector_entries (id, entity_type, entity_id, vector, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    entity_type = excluded.entity_type,
                    entity_id = excluded.entity_id,
                    vector = excluded.vector,
                    metadata = excluded.metadata
                """,
                (id, entity_type, entity_id, blob, meta_json),
            )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        entity_type: str | None = None,
    ) -> list[SearchResult]:
        """Search top-k most similar items to query_vector using cosine similarity."""
        query = "SELECT id, entity_type, entity_id, vector, metadata FROM vector_entries"
        params = []
        if entity_type:
            query += " WHERE entity_type = ?"
            params.append(entity_type)

        scored_results: list[SearchResult] = []
        with self.memory_db.get_connection() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                blob = row["vector"]
                num_floats = len(blob) // 4
                vector = list(struct.unpack(f"{num_floats}f", blob))
                sim = _cosine_similarity(query_vector, vector)
                meta = json.loads(row["metadata"])
                scored_results.append(
                    SearchResult(
                        id=row["id"],
                        entity_type=row["entity_type"],
                        entity_id=row["entity_id"],
                        score=sim,
                        metadata=meta,
                    )
                )

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    async def delete(self, id: str) -> bool:
        """Delete a vector entry by ID."""
        with self.memory_db.get_connection() as conn:
            cur = conn.execute("DELETE FROM vector_entries WHERE id = ?", (id,))
            return cur.rowcount > 0

    async def delete_by_prefix(self, prefix: str) -> int:
        """Delete all vector entries matching ID prefix."""
        with self.memory_db.get_connection() as conn:
            cur = conn.execute("DELETE FROM vector_entries WHERE id LIKE ?", (f"{prefix}%",))
            return cur.rowcount

    async def delete_all(self) -> int:
        """Delete all vector entries."""
        with self.memory_db.get_connection() as conn:
            cur = conn.execute("DELETE FROM vector_entries")
            return cur.rowcount
