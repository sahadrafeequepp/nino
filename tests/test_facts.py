"""Tests for durable fact extraction, parsing, sensitive filtering, and superseding."""

import hashlib
import json
import math
from pathlib import Path
import re
import pytest

from nino.memory.db import MemoryDB
from nino.memory.facts import FactManager
from nino.memory.vector.base import Embedder
from nino.memory.vector.sqlite_store import SQLiteVectorStore


SEMANTIC_CLUSTERS = [
    {"home", "location", "live", "lives", "zurich", "switzerland", "city", "residence", "where"},
    {"coffee", "drink", "beverage", "roast", "pour-over", "latte", "espresso", "tea"},
    {"dog", "pet", "pets", "cat", "animal", "luna", "puppy", "hound", "retriever"},
    {"quantum", "qubits", "superposition", "shor", "algorithm", "factorization", "logarithms"},
    {"kubernetes", "k8s", "ingress", "cluster", "traffic", "service", "replica", "autoscalers", "route", "routing"},
    {"garden", "gardening", "tomato", "tomatoes", "basil", "hornworms", "planting", "watering", "companion"},
    {"python", "code", "engineer", "software", "backend", "programming", "developer", "architect"},
    {"italian", "spanish", "language", "learning", "kanji", "japanese", "education"},
]


class DeterministicFakeEmbedder(Embedder):
    """Generates repeatable normalized float vectors simulating semantic similarity for unit tests."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        lower = text.lower()
        words = set(re.findall(r"\w+", lower))

        # 1. Semantic concept clusters
        for idx, cluster in enumerate(SEMANTIC_CLUSTERS):
            if words & cluster:
                slot = (idx * 5) % self.dim
                vec[slot] += 5.0
                vec[(slot + 1) % self.dim] += 3.0

        # 2. Character trigrams for morphological and subword similarity
        for i in range(len(lower) - 2):
            tri = lower[i : i + 3]
            h = int(hashlib.md5(tri.encode()).hexdigest(), 16)
            vec[h % self.dim] += 0.5

        # 3. Word hashes
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class MockLLMAdapter:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    async def chat(self, messages, **kwargs):
        res = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return res


@pytest.fixture
def setup_fact_env(tmp_path: Path):
    db_file = tmp_path / "facts_test.db"
    db = MemoryDB(db_path=str(db_file))
    vstore = SQLiteVectorStore(memory_db=db)
    embedder = DeterministicFakeEmbedder()
    mgr = FactManager(
        memory_db=db,
        vector_store=vstore,
        embedder=embedder,
        sensitive_memory=False,
    )
    return db, vstore, embedder, mgr


@pytest.mark.asyncio
async def test_fact_extraction_valid_json(setup_fact_env):
    _, _, _, mgr = setup_fact_env
    valid_json = json.dumps({
        "facts": [
            {"text": "User lives in Seattle", "category": "location"},
            {"text": "User works as a Python engineer", "category": "work"},
        ]
    })
    mock_llm = MockLLMAdapter([valid_json])

    facts = await mgr.extract_facts_from_turn("I live in Seattle and work as a Python engineer", "msg-1", mock_llm)
    assert len(facts) == 2
    assert facts[0]["text"] == "User lives in Seattle"
    assert facts[1]["text"] == "User works as a Python engineer"


@pytest.mark.asyncio
async def test_fact_extraction_malformed_retry(setup_fact_env):
    _, _, _, mgr = setup_fact_env
    # First response malformed text, retry response valid JSON
    malformed = "Here are the facts: Seattle and Python"
    valid_retry = json.dumps({
        "facts": [{"text": "User works with Python", "category": "work"}]
    })
    mock_llm = MockLLMAdapter([malformed, valid_retry])

    facts = await mgr.extract_facts_from_turn("I code in Python", "msg-2", mock_llm)
    assert len(facts) == 1
    assert facts[0]["text"] == "User works with Python"
    assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_fact_extraction_malformed_skip(setup_fact_env):
    _, _, _, mgr = setup_fact_env
    mock_llm = MockLLMAdapter(["malformed text 1", "malformed text 2"])
    facts = await mgr.extract_facts_from_turn("statement", "msg-3", mock_llm)
    assert facts == []


@pytest.mark.asyncio
async def test_sensitive_data_filtering(setup_fact_env):
    _, _, _, mgr = setup_fact_env

    # 1. AWS Key
    assert mgr.is_sensitive_content("My key is AKIAIOSFODNN7EXAMPLE", "cloud") is True

    # 2. Credit Card
    assert mgr.is_sensitive_content("Card is 4111 2222 3333 4444", "billing") is True

    # 3. Password
    assert mgr.is_sensitive_content("password: supersecretpassword", "auth") is True

    # 4. Sensitive categories when SENSITIVE_MEMORY=false
    assert mgr.is_sensitive_content("User has asthma", "health") is True
    assert mgr.is_sensitive_content("User has $50k in bank", "finances") is True

    # Non-sensitive
    assert mgr.is_sensitive_content("User drinks black coffee", "preferences") is False


@pytest.mark.asyncio
async def test_deduplication_and_supersede(setup_fact_env):
    db, vstore, _, mgr = setup_fact_env

    # Add initial fact
    f1 = await mgr.add_or_update_fact("User is learning Spanish", "education")
    assert f1 is not None

    # Add same fact again -> should be skipped as duplicate
    f_dup = await mgr.add_or_update_fact("User is learning Spanish", "education")
    assert f_dup is None

    # Add superseding fact in same category
    f2 = await mgr.add_or_update_fact("User is now learning Italian instead of Spanish", "education")
    assert f2 is not None

    facts = mgr.list_facts(category="education", current_only=False)
    assert len(facts) >= 2

    # Verify old fact is superseded and new one is current
    current_facts = [f for f in facts if f["is_current"]]
    superseded_facts = [f for f in facts if not f["is_current"]]

    assert len(current_facts) == 1
    assert "Italian" in current_facts[0]["text"]
    assert len(superseded_facts) == 1
    assert superseded_facts[0]["superseded_by"] == f2["id"]


@pytest.mark.asyncio
async def test_delete_from_both_stores(setup_fact_env):
    db, vstore, _, mgr = setup_fact_env
    fact = await mgr.add_or_update_fact("User prefers dark mode", "preferences")
    fact_id = fact["id"]

    # Verify exists in SQLite
    facts = mgr.list_facts(category="preferences")
    assert any(f["id"] == fact_id for f in facts)

    # Delete fact
    deleted = await mgr.delete_fact(fact_id)
    assert deleted is True

    # Verify removed from SQLite
    facts_after = mgr.list_facts(category="preferences")
    assert not any(f["id"] == fact_id for f in facts_after)

    # Verify removed from vector store
    search_res = await vstore.search(await mgr.embedder.embed("dark mode"), top_k=5)
    assert not any(r.entity_id == fact_id for r in search_res)
