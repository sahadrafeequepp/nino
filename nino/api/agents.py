"""API router for sub-agents inspection and delegation observability."""

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status

from nino.agents.models import AgentDefinition, AgentRunRecord
from nino.agents.registry import AgentRegistry
from nino.agents.store import DelegationRunStore, get_run_store
from nino.auth import verify_token

logger = logging.getLogger("nino.api.agents")

router = APIRouter(prefix="/agents", tags=["Sub-Agents"], dependencies=[Depends(verify_token)])

_agent_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry


def set_agent_registry(registry: AgentRegistry) -> None:
    global _agent_registry
    _agent_registry = registry


@router.get(
    "/definitions",
    response_model=list[AgentDefinition],
    summary="List registered sub-agents and their tool capabilities",
)
async def list_agent_definitions(
    registry: AgentRegistry = Depends(get_agent_registry),
) -> list[AgentDefinition]:
    """Return all configured sub-agents, allowed tools, permission tiers, and budget defaults."""
    return registry.list_agents()


@router.get(
    "/runs",
    response_model=list[AgentRunRecord],
    summary="List recent sub-agent delegation runs and outcomes",
)
async def list_agent_runs(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of runs to return"),
    store: DelegationRunStore = Depends(get_run_store),
) -> list[AgentRunRecord]:
    """Return recent sub-agent delegations, tools invoked, budget metrics, and taint outcomes."""
    return store.list_runs(limit=limit)


@router.get(
    "/runs/{run_id}",
    response_model=AgentRunRecord,
    summary="Retrieve full execution trace for a specific sub-agent run",
)
async def get_agent_run(
    run_id: str,
    store: DelegationRunStore = Depends(get_run_store),
) -> AgentRunRecord:
    """Return detailed trace and steps for a specific delegation run."""
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Delegation run '{run_id}' not found.",
        )
    return run
