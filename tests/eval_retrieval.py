"""Evaluation benchmark script for Nino memory and document retrieval hit-rate."""

import asyncio
from pathlib import Path
import tempfile
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from nino.memory.db import MemoryDB
from nino.memory.facts import FactManager
from nino.memory.ingest import DocumentIngestionPipeline
from nino.memory.retrieval import RetrievalPipeline
from nino.memory.vector.sqlite_store import SQLiteVectorStore
from tests.test_facts import DeterministicFakeEmbedder


async def run_evaluation():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_file = tmp_path / "eval_memory.db"
        notes_dir = tmp_path / "eval_notes"
        notes_dir.mkdir()

        db = MemoryDB(str(db_file))
        vstore = SQLiteVectorStore(db)
        embedder = DeterministicFakeEmbedder(dim=32)

        fact_mgr = FactManager(db, vstore, embedder, sensitive_memory=True)
        pipeline = DocumentIngestionPipeline(db, vstore, embedder, chunk_size=300, chunk_overlap=50)
        retrieval = RetrievalPipeline(db, vstore, embedder, token_budget_memory=1500)

        # 1. Seed Facts
        sample_facts = [
            ("User works as a principal distributed systems architect", "work"),
            ("User lives in Zurich Switzerland", "location"),
            ("User prefers dark roast pour-over coffee", "preferences"),
            ("User has a golden retriever named Luna", "pets"),
            ("User is learning Japanese kanji characters", "education"),
            ("User prefers Neovim over VS Code for terminal editing", "preferences"),
        ]

        print("Seeding sample facts...")
        for text, cat in sample_facts:
            await fact_mgr.add_or_update_fact(text, cat)

        # 2. Seed Notes
        print("Seeding sample documents...")
        note1 = notes_dir / "quantum_notes.md"
        note1.write_text(
            "# Quantum Computing Primer\n"
            "Quantum superposition allows qubits to represent 0 and 1 simultaneously.\n"
            "Shor's algorithm solves discrete logarithms and prime factorization in polynomial time.",
            encoding="utf-8",
        )

        note2 = notes_dir / "k8s_guide.md"
        note2.write_text(
            "# Kubernetes Production Runbook\n"
            "Ingress controllers route external HTTP traffic to cluster clusterIP services.\n"
            "Horizontal Pod Autoscalers trigger replica scaling based on CPU metrics.",
            encoding="utf-8",
        )

        note3 = notes_dir / "gardening.txt"
        note3.write_text(
            "Organic vegetable gardening notes:\n"
            "Heirloom tomatoes require deep watering and full sunlight for at least 8 hours a day.\n"
            "Companion planting with basil helps repel hornworms.",
            encoding="utf-8",
        )

        await pipeline.ingest_directory(notes_dir)

        # 3. Benchmark Queries and Ground Truth keywords
        test_cases = [
            {"query": "Where is my home located?", "target": "Zurich", "type": "fact"},
            {"query": "What kind of coffee do I like?", "target": "dark roast pour-over", "type": "fact"},
            {"query": "What is my dog's name?", "target": "Luna", "type": "fact"},
            {"query": "How does Shor's algorithm work?", "target": "factorization", "type": "doc"},
            {"query": "How do ingress controllers route traffic?", "target": "traffic", "type": "doc"},
            {"query": "Tell me about companion planting for tomatoes", "target": "basil", "type": "doc"},
        ]

        hits_top_1 = 0
        hits_top_3 = 0
        total_queries = len(test_cases)

        print("\n" + "=" * 65)
        print("  Running Retrieval Benchmark (Top-k Hit Rate)")
        print("=" * 65)

        for case in test_cases:
            q = case["query"]
            target = case["target"].lower()
            ctx = await retrieval.retrieve_context(q, top_k_facts=3, top_k_docs=3)

            # Check matches in retrieved facts or chunks depending on query type
            if case["type"] == "fact":
                retrieved_texts = [f["text"].lower() for f in ctx.retrieved_facts]
            else:
                retrieved_texts = [c["text"].lower() for c in ctx.retrieved_chunks]

            hit_1 = len(retrieved_texts) > 0 and target in retrieved_texts[0]
            hit_3 = any(target in t for t in retrieved_texts[:3])

            if hit_1:
                hits_top_1 += 1
            if hit_3:
                hits_top_3 += 1

            status_str = "HIT (Top-1)" if hit_1 else ("HIT (Top-3)" if hit_3 else "MISS")
            print(f"Query:  \"{q}\"")
            print(f"Target: \"{case['target']}\" -> Result: {status_str}\n")

        top_1_rate = (hits_top_1 / total_queries) * 100
        top_3_rate = (hits_top_3 / total_queries) * 100

        print("=" * 65)
        print(f"  Benchmark Results: {total_queries} queries evaluated")
        print(f"  Top-1 Hit Rate: {top_1_rate:.1f}% ({hits_top_1}/{total_queries})")
        print(f"  Top-3 Hit Rate: {top_3_rate:.1f}% ({hits_top_3}/{total_queries})")
        print("=" * 65)

        if top_3_rate >= 80.0:
            print(">> Evaluation Status: PASSED (Hit rate exceeds 80% threshold)")
            return 0
        else:
            print(">> Evaluation Status: FAILED (Hit rate below 80% threshold)")
            return 1


if __name__ == "__main__":
    code = asyncio.run(run_evaluation())
    sys.exit(code)
