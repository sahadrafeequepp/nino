"""Registry for managing and discovering Nino sub-agent definitions."""

import logging
from pathlib import Path
from typing import Any
import yaml

from nino.agents.models import AgentBudget, AgentDefinition

logger = logging.getLogger("nino.agents.registry")

DEFAULT_AGENTS_YAML = Path("persona/agents.yaml")


class AgentRegistry:
    """Registry maintaining available sub-agent definitions and their configurations."""

    def __init__(self, config_path: str | Path | None = None):
        self._agents: dict[str, AgentDefinition] = {}
        self.config_path = Path(config_path) if config_path else DEFAULT_AGENTS_YAML
        self.load()

    def register(self, agent: AgentDefinition) -> None:
        """Register a sub-agent definition."""
        # Enforce invariant: sub-agents can never include delegation tools (depth 1 limit)
        sanitized_tools = [t for t in agent.allowed_tools if t != "delegate_task"]
        if len(sanitized_tools) != len(agent.allowed_tools):
            logger.warning(
                "Stripped disallowed 'delegate_task' tool from agent '%s' (depth limit: 1).",
                agent.name,
            )
            agent.allowed_tools = sanitized_tools

        self._agents[agent.name] = agent
        logger.debug("Registered sub-agent: %s (tools: %s)", agent.name, agent.allowed_tools)

    def get(self, name: str) -> AgentDefinition | None:
        """Retrieve a sub-agent definition by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[AgentDefinition]:
        """Return all registered sub-agents."""
        return list(self._agents.values())

    def load(self) -> None:
        """Load sub-agents from YAML configuration file or populate sensible defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                raw_agents = data.get("agents", {})
                for name, cfg in raw_agents.items():
                    budget_dict = cfg.get("default_budget", {})
                    budget = AgentBudget(
                        max_steps=budget_dict.get("max_steps", 5),
                        timeout_seconds=budget_dict.get("timeout_seconds", 60.0),
                        max_tokens=budget_dict.get("max_tokens", 2048),
                    )
                    defn = AgentDefinition(
                        name=cfg.get("name", name),
                        description=cfg.get("description", ""),
                        model=cfg.get("model", "gemma2:2b"),
                        max_tier=cfg.get("max_tier", "READ_ONLY"),
                        allowed_tools=cfg.get("allowed_tools", []),
                        default_budget=budget,
                        system_prompt=cfg.get("system_prompt", ""),
                    )
                    self.register(defn)
                logger.info("Loaded %d sub-agents from %s", len(self._agents), self.config_path)
                return
            except Exception as e:
                logger.error("Failed to load agent definitions from %s: %s", self.config_path, e)

        # Fallback built-ins if config file is not found
        self._load_fallback_defaults()

    def _load_fallback_defaults(self) -> None:
        """Populate built-in default sub-agents if no config file exists."""
        self.register(
            AgentDefinition(
                name="researcher",
                description="Web search, document retrieval, and note inspection with strictly read-only tools.",
                model="gemma2:2b",
                max_tier="READ_ONLY",
                allowed_tools=["web_search", "file_read", "note_search", "note_read", "current_time"],
                default_budget=AgentBudget(max_steps=5, timeout_seconds=60.0, max_tokens=2048),
                system_prompt="You are Nino's Researcher sub-agent. Gather factual information accurately.",
            )
        )
        self.register(
            AgentDefinition(
                name="coder",
                description="Code inspection, repository file search, and sandboxed Python execution.",
                model="gemma2:2b",
                max_tier="SENSITIVE",
                allowed_tools=["file_search", "file_read", "execute_python_code", "calculator"],
                default_budget=AgentBudget(max_steps=6, timeout_seconds=90.0, max_tokens=4096),
                system_prompt="You are Nino's Coder sub-agent. Inspect files and test Python code safely.",
            )
        )
        self.register(
            AgentDefinition(
                name="planner",
                description="Calendar management, tasks, reminders, and focus session scheduling.",
                model="gemma2:2b",
                max_tier="REVERSIBLE",
                allowed_tools=[
                    "calendar_list_events",
                    "calendar_create_event",
                    "calendar_update_event",
                    "calendar_delete_event",
                    "task_list",
                    "task_create",
                    "task_complete",
                    "task_delete",
                    "reminder_create",
                    "reminder_list",
                    "reminder_cancel",
                    "reminder_snooze",
                    "focus_start",
                    "focus_status",
                    "focus_stop",
                    "current_time",
                ],
                default_budget=AgentBudget(max_steps=6, timeout_seconds=60.0, max_tokens=2048),
                system_prompt="You are Nino's Planner sub-agent. Organize schedule, tasks, and reminders.",
            )
        )
