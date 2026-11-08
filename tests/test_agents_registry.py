"""Tests for Nino sub-agent registry and definition loading."""

from pathlib import Path
import pytest

from nino.agents.models import AgentBudget, AgentDefinition
from nino.agents.registry import AgentRegistry


def test_agent_registry_loads_default_yaml():
    registry = AgentRegistry("persona/agents.yaml")
    agents = registry.list_agents()

    assert len(agents) >= 3
    names = {a.name for a in agents}
    assert {"researcher", "coder", "planner"}.issubset(names)


def test_researcher_agent_definition():
    registry = AgentRegistry("persona/agents.yaml")
    researcher = registry.get("researcher")

    assert researcher is not None
    assert researcher.name == "researcher"
    assert researcher.max_tier == "READ_ONLY"
    assert "web_search" in researcher.allowed_tools
    assert "file_read" in researcher.allowed_tools
    assert "note_search" in researcher.allowed_tools
    assert "note_read" in researcher.allowed_tools
    # Disallowed tools
    assert "execute_python_code" not in researcher.allowed_tools
    assert "task_delete" not in researcher.allowed_tools
    assert "delegate_task" not in researcher.allowed_tools


def test_coder_agent_definition():
    registry = AgentRegistry("persona/agents.yaml")
    coder = registry.get("coder")

    assert coder is not None
    assert coder.name == "coder"
    assert "file_search" in coder.allowed_tools
    assert "execute_python_code" in coder.allowed_tools
    assert "delegate_task" not in coder.allowed_tools


def test_planner_agent_definition():
    registry = AgentRegistry("persona/agents.yaml")
    planner = registry.get("planner")

    assert planner is not None
    assert planner.name == "planner"
    assert "calendar_list_events" in planner.allowed_tools
    assert "task_create" in planner.allowed_tools
    assert "reminder_create" in planner.allowed_tools
    assert "focus_start" in planner.allowed_tools
    assert "delegate_task" not in planner.allowed_tools


def test_depth_limit_tool_stripping():
    # Enforce that even if someone puts 'delegate_task' into allowed_tools, the registry strips it
    registry = AgentRegistry("persona/agents.yaml")
    custom_agent = AgentDefinition(
        name="rogue_agent",
        description="Attempts to delegate",
        allowed_tools=["web_search", "delegate_task"],
        system_prompt="Delegate everything",
    )
    registry.register(custom_agent)

    retrieved = registry.get("rogue_agent")
    assert retrieved is not None
    assert "web_search" in retrieved.allowed_tools
    assert "delegate_task" not in retrieved.allowed_tools


def test_fallback_defaults_when_no_config(tmp_path):
    empty_path = tmp_path / "nonexistent.yaml"
    registry = AgentRegistry(empty_path)
    agents = registry.list_agents()

    assert len(agents) >= 3
    names = {a.name for a in agents}
    assert "researcher" in names
    assert "coder" in names
    assert "planner" in names
