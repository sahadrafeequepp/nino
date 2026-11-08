"""Live verification script running end-to-end against the live Nino server."""

import asyncio
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from nino.config import get_settings
import httpx

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


async def run_live_verification():
    settings = get_settings()
    host_client = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host
    base_url = f"http://{host_client}:{settings.port}"
    token = settings.auth_token
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    print("=" * 60)
    print("  Nino End-to-End Live Verification")
    print(f"  Target: {base_url}")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Health Check
        print("\n[Step 1] Checking /health...")
        h_resp = await client.get(f"{base_url}/health")
        print("  Status:", h_resp.status_code, h_resp.json())
        assert h_resp.status_code == 200

        # Step 2: Session 1 - Tell Nino a fact
        print("\n[Step 2] Telling Nino a fact in Session 1...")
        turn1_payload = {
            "session_id": "session-live-01",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello Nino, I want to let you know that I work as a quantum computing researcher in Zurich.",
                }
            ],
        }
        res1 = await client.post(f"{base_url}/chat", headers=headers, json=turn1_payload)
        print("  Status:", res1.status_code)
        data1 = res1.json()
        print("  Reply:", repr(data1.get("reply", "")[:100]))
        assert res1.status_code == 200

        # Poll up to 25s for background fact extraction to complete
        print("  Polling for background fact extraction (up to 25s)...")
        facts = []
        for attempt in range(12):
            await asyncio.sleep(2.0)
            facts_res = await client.get(f"{base_url}/memory/facts", headers=headers)
            facts = facts_res.json().get("facts", [])
            if facts:
                print(f"  Fact extraction completed on attempt {attempt + 1}.")
                break

        # Step 3: Check /memory/facts
        print("\n[Step 3] Verifying fact was stored in SQLite & vector store...")
        print(f"  Found {len(facts)} fact(s):")
        for f in facts:
            print(f"    - [{f['category']}] {f['text']}")
        assert len(facts) >= 1

        # Step 4: Session 2 (Brand New Session) - Recall the fact
        print("\n[Step 4] Asking Nino to recall fact in a brand new Session 2...")
        turn2_payload = {
            "session_id": "session-live-02",
            "messages": [
                {
                    "role": "user",
                    "content": "What is my profession and what city do I live in?",
                }
            ],
        }
        res2 = await client.post(f"{base_url}/chat", headers=headers, json=turn2_payload)
        data2 = res2.json()
        print("  Reply:", data2.get("reply"))
        assert res2.status_code == 200
        reply_lower = data2.get("reply", "").lower()
        # Verify recall
        assert "quantum" in reply_lower or "zurich" in reply_lower or "researcher" in reply_lower

        # Step 5: Test Note Ingestion (RAG)
        print("\n[Step 5] Ingesting test notes and testing document citation...")
        notes_dir = Path("./test_notes")
        notes_dir.mkdir(exist_ok=True)
        note_file = notes_dir / "project_supernova.md"
        note_file.write_text(
            "# Project Supernova\n\n"
            "Project Supernova is our secret next-generation propulsion engine designed for deep space travel.\n"
            "It operates using magnetized target fusion and achieves 10% the speed of light.\n"
            "The lead engineering division is based in Zurich.",
            encoding="utf-8",
        )

        from nino.memory.db import MemoryDB
        from nino.memory.ingest import DocumentIngestionPipeline
        from nino.memory.vector.ollama_embedder import OllamaEmbedder
        from nino.memory.vector.sqlite_store import SQLiteVectorStore

        db = MemoryDB(settings.db_path, settings.memory_encryption_key)
        vstore = SQLiteVectorStore(db)
        embedder = OllamaEmbedder(settings.ollama_base_url, settings.embed_model)
        pipeline = DocumentIngestionPipeline(db, vstore, embedder)
        stats = await pipeline.ingest_directory(notes_dir)
        print("  Ingestion stats:", stats)
        assert stats["indexed"] >= 1 or stats["skipped"] >= 1

        # Query Nino about the document
        turn3_payload = {
            "session_id": "session-live-03",
            "messages": [
                {
                    "role": "user",
                    "content": "What is Project Supernova and what propulsion mechanism does it use?",
                }
            ],
        }
        res3 = await client.post(f"{base_url}/chat", headers=headers, json=turn3_payload)
        data3 = res3.json()
        print("  Reply:", repr(data3.get("reply", "")))
        print("  Citations:", data3.get("citations"))
        assert res3.status_code == 200
        assert len(data3.get("citations", [])) >= 1

        # Step 6: Memory Management - Delete Fact
        print("\n[Step 6] Deleting fact and verifying removal...")
        fact_id_to_delete = facts[0]["id"]
        del_res = await client.delete(f"{base_url}/memory/facts/{fact_id_to_delete}", headers=headers)
        print("  Delete status:", del_res.status_code, del_res.json())
        assert del_res.status_code == 200

        # Verify fact is gone
        facts_after_res = await client.get(f"{base_url}/memory/facts", headers=headers)
        remaining_facts = facts_after_res.json().get("facts", [])
        assert not any(f["id"] == fact_id_to_delete for f in remaining_facts)
        print("  Fact successfully purged from SQLite and vector search.")

    print("\n" + "=" * 60)
    print("  ALL LIVE VERIFICATION CHECKS PASSED!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    code = asyncio.run(run_live_verification())
    sys.exit(code)
