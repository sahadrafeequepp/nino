"""Live verification script demonstrating Phase 3 tools: note saving/retrieval, task deletion confirmation, and argument tamper rejection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from nino.config import get_settings
from nino.main import app

# Ensure Windows terminal handles UTF-8 cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AUTH_HEADER = {"Authorization": f"Bearer {get_settings().auth_token}"}


def main():
    print("=" * 60)
    print("  Nino Phase 3 Tools Verification")
    print("=" * 60)

    with TestClient(app) as client:
        # 1. Health check
        print("\n[Step 1] Checking /health...")
        res = client.get("/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        print(f"  OK: {res.json()}")

        # 2. Direct tool execution through API: Create a note
        print("\n[Step 2] Testing Note Creation via Orchestrator / API...")
        payload_note = {
            "messages": [
                {
                    "role": "user",
                    "content": '```json\n{"tool": "notes_create", "arguments": {"title": "Secret Recipe", "content": "The secret ingredient is cardamom."}}\n```',
                }
            ]
        }
        res_note = client.post("/chat", headers=AUTH_HEADER, json=payload_note)
        assert res_note.status_code == 200, f"Note creation failed: {res_note.text}"
        data_note = res_note.json()
        print(f"  Note creation reply: {data_note.get('reply')[:100]}...")
        assert len(data_note.get("tool_calls", [])) > 0
        assert data_note["tool_calls"][0]["decision"] == "executed"
        print("  Note created successfully.")

        # 3. Find the note later
        print("\n[Step 3] Finding the note later...")
        payload_search = {
            "messages": [
                {
                    "role": "user",
                    "content": '```json\n{"tool": "notes_search", "arguments": {"query": "Recipe"}}\n```',
                }
            ]
        }
        res_search = client.post("/chat", headers=AUTH_HEADER, json=payload_search)
        assert res_search.status_code == 200
        data_search = res_search.json()
        assert len(data_search.get("tool_calls", [])) > 0
        search_output = str(data_search["tool_calls"][0]["output"])
        assert "Secret Recipe" in search_output
        assert 'taint="UNTRUSTED"' in search_output
        print("  Found note content correctly with untrusted taint boundary tags.")

        # 4. Create a task, then attempt to delete it (sensitive tool)
        print("\n[Step 4] Creating task and testing deletion confirmation flow...")
        payload_task = {
            "messages": [
                {
                    "role": "user",
                    "content": '```json\n{"tool": "task_create", "arguments": {"title": "Buy groceries", "priority": "high"}}\n```',
                }
            ]
        }
        res_task = client.post("/chat", headers=AUTH_HEADER, json=payload_task)
        assert res_task.status_code == 200
        task_out = res_task.json()["tool_calls"][0]["output"]
        task_id = task_out["task_id"]
        print(f"  Task created with ID #{task_id}.")

        # Attempt delete task
        payload_del = {
            "messages": [
                {
                    "role": "user",
                    "content": f'```json\n{{"tool": "task_delete", "arguments": {{"task_id": {task_id}}}}}\n```',
                }
            ]
        }
        res_del = client.post("/chat", headers=AUTH_HEADER, json=payload_del)
        assert res_del.status_code == 200
        data_del = res_del.json()

        # Verify confirmation was required and tool did NOT execute
        assert data_del["pending_confirmation"] is not None
        conf = data_del["pending_confirmation"]
        conf_id = conf["id"]
        print(f"  Confirmation required! ID: {conf_id}")
        print(f"  Summary: {conf['summary']}")
        assert conf["tool_name"] == "task_delete"
        assert conf["tier"] == "SENSITIVE"

        # Verify task still exists in database!
        res_list = client.post(
            "/chat",
            headers=AUTH_HEADER,
            json={"messages": [{"role": "user", "content": '```json\n{"tool": "task_list", "arguments": {}}\n```'}]},
        )
        assert "Buy groceries" in str(res_list.json())
        print("  Verified: Task was NOT deleted while confirmation was pending.")

        # 5. Tamper test: attempt to approve with modified arguments
        print("\n[Step 5] Testing Argument Tampering Defense...")
        res_approve = client.post(f"/confirmations/{conf_id}/approve", headers=AUTH_HEADER)
        assert res_approve.status_code == 200
        print(f"  Approved confirmation '{conf_id}'.")

        # Try to execute with tampered argument (task_id 99999 instead of task_id)
        tampered_payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'```json\n{{"tool": "task_delete", "arguments": {{"task_id": 99999}}}}\n```',
                }
            ],
            "confirmation_id": conf_id,
        }
        res_tamper = client.post("/chat", headers=AUTH_HEADER, json=tampered_payload)
        assert res_tamper.status_code == 200
        tamper_data = res_tamper.json()
        assert tamper_data["tool_calls"][0]["decision"] == "blocked"
        print("  Tampered execution successfully blocked by argument binding verification!")

        # 6. Legitimate execution with exact approved arguments
        print("\n[Step 6] Executing with exact arguments and approved token...")
        legit_payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'```json\n{{"tool": "task_delete", "arguments": {{"task_id": {task_id}}}}}\n```',
                }
            ],
            "confirmation_id": conf_id,
        }
        res_legit = client.post("/chat", headers=AUTH_HEADER, json=legit_payload)
        assert res_legit.status_code == 200
        legit_data = res_legit.json()
        assert legit_data["tool_calls"][0]["decision"] == "executed"
        print("  Task successfully deleted after approval.")

        # Verify task is now deleted
        res_list_after = client.post(
            "/chat",
            headers=AUTH_HEADER,
            json={"messages": [{"role": "user", "content": '```json\n{"tool": "task_list", "arguments": {}}\n```'}]},
        )
        assert "Buy groceries" not in str(res_list_after.json())
        print("  Verified: Task is now permanently removed.")

        print("\n" + "=" * 60)
        print("  ALL PHASE 3 TOOLS LIVE VERIFICATION STEPS PASSED!")
        print("=" * 60)


if __name__ == "__main__":
    main()
