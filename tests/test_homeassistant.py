"""Unit tests for Home Assistant integration: allowlist, permission tiers, dry-run, and security restrictions."""

import pytest
import httpx

from nino.config import Settings
from nino.integrations.homeassistant import (
    EntityNotAllowedError,
    HAGetStateTool,
    HAListEntitiesTool,
    HASecurityControlTool,
    HASetValueTool,
    HATurnOffTool,
    HATurnOnTool,
    HomeAssistantClient,
)
from nino.tools.permissions import PermissionTier


@pytest.fixture
def ha_settings():
    return Settings(
        ha_url="http://127.0.0.1:8123",
        ha_token="secret_ha_long_lived_token_12345",
        ha_allowed_entities="light.kitchen,switch.coffee_maker,climate.living_room,lock.front_door",
        ha_dry_run=True,
        ha_enable_restricted=False,
    )


@pytest.fixture
def mock_ha_client(ha_settings):
    # Mock transport responding to allowed entities
    def handler(request: httpx.Request):
        url = str(request.url)
        if "light.kitchen" in url:
            return httpx.Response(200, json={"entity_id": "light.kitchen", "state": "on", "attributes": {"brightness": 200}})
        elif "switch.coffee_maker" in url:
            return httpx.Response(200, json={"entity_id": "switch.coffee_maker", "state": "off", "attributes": {}})
        return httpx.Response(404, json={"message": "Not found"})

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return HomeAssistantClient(settings=ha_settings, http_client=http_client)


def test_permission_tiers(mock_ha_client):
    """Verify tool permission tiers conform to Nino security invariants."""
    assert HAListEntitiesTool(mock_ha_client).tier == PermissionTier.READ_ONLY
    assert HAGetStateTool(mock_ha_client).tier == PermissionTier.READ_ONLY
    assert HATurnOnTool(mock_ha_client).tier == PermissionTier.REVERSIBLE
    assert HATurnOffTool(mock_ha_client).tier == PermissionTier.REVERSIBLE
    assert HASetValueTool(mock_ha_client).tier == PermissionTier.REVERSIBLE
    assert HASecurityControlTool(mock_ha_client).tier == PermissionTier.RESTRICTED


@pytest.mark.asyncio
async def test_entity_allowlist_enforcement(mock_ha_client):
    """Verify non-allowlisted entities are strictly rejected."""
    # light.kitchen is allowlisted
    assert mock_ha_client.is_entity_allowed("light.kitchen") is True
    # light.bedroom is NOT allowlisted
    assert mock_ha_client.is_entity_allowed("light.bedroom") is False

    tool_on = HATurnOnTool(mock_ha_client)
    with pytest.raises(EntityNotAllowedError, match="not in the Home Assistant allowlist"):
        await tool_on.run(entity_id="light.bedroom")


@pytest.mark.asyncio
async def test_dry_run_simulation(mock_ha_client):
    """Verify dry-run mode returns formatted simulations without executing mutations."""
    tool_on = HATurnOnTool(mock_ha_client)
    res = await tool_on.run(entity_id="light.kitchen", dry_run=True)
    assert res["status"] == "dry_run_simulated"
    assert "Successfully simulated" in res["message"]
    assert res["entity_id"] == "light.kitchen"


@pytest.mark.asyncio
async def test_restricted_security_devices(ha_settings):
    """Verify security devices (locks, garage) are forbidden in general tools and restricted."""
    client = HomeAssistantClient(settings=ha_settings)

    # Calling ha_turn_on on a lock must be rejected
    tool_on = HATurnOnTool(client)
    with pytest.raises(ValueError, match="security-critical device"):
        await tool_on.run(entity_id="lock.front_door")

    # HASecurityControlTool when ha_enable_restricted is False must be rejected
    sec_tool_disabled = HASecurityControlTool(client, enabled=False)
    with pytest.raises(PermissionError, match="disabled in configuration"):
        await sec_tool_disabled.run(entity_id="lock.front_door", action="unlock")

    # When enabled, HASecurityControlTool executes (simulated in dry-run)
    sec_tool_enabled = HASecurityControlTool(client, enabled=True)
    res = await sec_tool_enabled.run(entity_id="lock.front_door", action="unlock", dry_run=True)
    assert res["status"] == "dry_run_simulated"
    assert res["service"] == "unlock"


def test_token_not_leaked_in_exceptions(ha_settings):
    """Verify the HA token never leaks into string representation or allowlist errors."""
    client = HomeAssistantClient(settings=ha_settings)
    assert ha_settings.ha_token not in str(client)
    try:
        raise EntityNotAllowedError("Unauthorized access to entity")
    except Exception as e:
        assert ha_settings.ha_token not in str(e)
