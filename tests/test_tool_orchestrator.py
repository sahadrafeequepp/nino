"""Tests for ToolOrchestrator: multi-turn tool loops, fallbacks, retries, and caps."""

from unittest.mock import AsyncMock
import pytest
from pydantic import BaseModel, Field

from nino.memory.db import MemoryDB
from nino.models.base import LLMAdapter
from nino.security.audit import AuditLogger
from nino.tools.base import BaseTool
from nino.tools.confirmation import ConfirmationManager
from nino.tools.executor import CentralExecutor
from nino.tools.orchestrator import ToolOrchestrator
from nino.tools.permissions import PermissionTier
from nino.tools.registry import ToolRegistry


class AddNumbersInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")


class AddNumbersTool(BaseTool):
    name = "add_numbers"
    description = "Add two numbers together"
    args_schema = AddNumbersInput
    tier = PermissionTier.READ_ONLY

    async def run(self, a: int, b: int) -> int:
        return a + b


class MockAdapter(LLMAdapter):
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.call_count = 0

    async def chat(self, messages: list[dict[str, str]], **kwargs):
        self.call_count += 1
        if self.responses:
            return self.responses.pop(0)
        return "Default reply"

    async def stream_chat(self, messages, **kwargs):
        yield "chunk"


@pytest.fixture
def setup_orch(tmp_path):
    db_file = tmp_path / "test_orch.db"
    audit_file = tmp_path / "test_orch_audit.jsonl"
    memory_db = MemoryDB(str(db_file))
    conf_mgr = ConfirmationManager(memory_db)
    audit_logger = AuditLogger(str(audit_file))

    registry = ToolRegistry()
    registry.register(AddNumbersTool())

    executor = CentralExecutor(
        registry=registry,
        confirmation_manager=conf_mgr,
        audit_logger=audit_logger,
    )
    orchestrator = ToolOrchestrator(
        registry=registry,
        executor=executor,
        max_iterations=5,
        total_timeout=10.0,
    )
    return registry, conf_mgr, executor, orchestrator


@pytest.mark.asyncio
async def test_native_tool_calling_loop(setup_orch):
    _, _, _, orchestrator = setup_orch

    # Mock model: first turn requests tool call via native tool_calls format;
    # second turn receives tool result and provides final text reply.
    turn_1 = {
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "add_numbers",
                    "arguments": {"a": 10, "b": 25},
                }
            }
        ],
    }
    turn_2 = {"content": "The sum of 10 and 25 is 35.", "tool_calls": []}

    mock_llm = MockAdapter([turn_1, turn_2])
    messages = [{"role": "user", "content": "What is 10 plus 25?"}]

    res = await orchestrator.run(messages=messages, adapter=mock_llm)
    assert res.reply == "The sum of 10 and 25 is 35."
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].tool_name == "add_numbers"
    assert res.tool_calls[0].output == 35
    assert res.iterations == 2


@pytest.mark.asyncio
async def test_fallback_json_tool_calling_loop(setup_orch):
    _, _, _, orchestrator = setup_orch

    # Mock model: outputs JSON block tool call in text
    turn_1 = '```json\n{"tool": "add_numbers", "arguments": {"a": 5, "b": 7}}\n```'
    turn_2 = "The answer is 12."

    mock_llm = MockAdapter([turn_1, turn_2])
    messages = [{"role": "user", "content": "Calculate 5 + 7."}]

    res = await orchestrator.run(messages=messages, adapter=mock_llm)
    assert res.reply == "The answer is 12."
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].output == 12


@pytest.mark.asyncio
async def test_malformed_tool_call_retry_and_recovery(setup_orch):
    _, _, _, orchestrator = setup_orch

    # Mock model: turn 1 malformed JSON; turn 2 corrects itself; turn 3 answers
    turn_1 = '```json\n{"tool": "add_numbers", "arguments": MALFORMED_JSON\n```'
    turn_2 = '```json\n{"tool": "add_numbers", "arguments": {"a": 2, "b": 3}}\n```'
    turn_3 = "The result is 5."

    mock_llm = MockAdapter([turn_1, turn_2, turn_3])
    messages = [{"role": "user", "content": "Add 2 and 3."}]

    res = await orchestrator.run(messages=messages, adapter=mock_llm)
    assert res.reply == "The result is 5."
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].output == 5


@pytest.mark.asyncio
async def test_iteration_cap_enforcement(setup_orch):
    _, _, _, orchestrator = setup_orch

    # Mock model: infinite loop calling tool repeatedly
    infinite_calls = [
        '```json\n{"tool": "add_numbers", "arguments": {"a": 1, "b": 1}}\n```'
        for _ in range(10)
    ]
    mock_llm = MockAdapter(infinite_calls)
    messages = [{"role": "user", "content": "Loop forever."}]

    res = await orchestrator.run(messages=messages, adapter=mock_llm)
    assert res.iterations == 5
    assert "maximum allowed iterations" in res.reply
    assert len(res.tool_calls) == 5
