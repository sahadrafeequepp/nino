"""Home Assistant safe smart-home integration with entity allowlist and permission tiers."""

import logging
from typing import Any, Optional
import httpx
from pydantic import BaseModel, Field

from nino.config import Settings, get_settings
from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier

logger = logging.getLogger("nino.integrations.homeassistant")

# Domains considered security-critical and classified as RESTRICTED
SECURITY_DOMAINS = {"lock", "alarm_control_panel", "cover", "camera"}


class EntityNotAllowedError(ValueError):
    """Raised when an operation targets an entity not in the configured allowlist."""
    pass


class HomeAssistantClient:
    """Safely interacts with Home Assistant REST API respecting entity allowlists and dry-run."""

    def __init__(self, settings: Optional[Settings] = None, http_client: Optional[httpx.AsyncClient] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.ha_url.rstrip("/")
        self.token = self.settings.ha_token.strip()
        self.allowed_entities = self._parse_allowlist(self.settings.ha_allowed_entities)
        self.dry_run = self.settings.ha_dry_run
        self.http_client = http_client

    def _parse_allowlist(self, raw_str: str) -> set[str]:
        if not raw_str:
            return set()
        return {e.strip().lower() for e in raw_str.split(",") if e.strip()}

    def is_entity_allowed(self, entity_id: str) -> bool:
        """Check whether entity is in allowlist."""
        return entity_id.strip().lower() in self.allowed_entities

    def is_security_entity(self, entity_id: str) -> bool:
        """Check whether entity belongs to a restricted security domain (lock, alarm, cover, camera)."""
        domain = entity_id.split(".")[0].lower() if "." in entity_id else ""
        return domain in SECURITY_DOMAINS

    async def _get_client(self) -> httpx.AsyncClient:
        if self.http_client:
            return self.http_client
        return httpx.AsyncClient(timeout=8.0)

    async def get_state(self, entity_id: str) -> dict[str, Any]:
        """Fetch the current state and attributes of an entity."""
        entity_clean = entity_id.strip().lower()
        if not self.is_entity_allowed(entity_clean):
            raise EntityNotAllowedError(f"Entity '{entity_clean}' is not in the Home Assistant allowlist.")

        if self.dry_run:
            return {
                "entity_id": entity_clean,
                "state": "on",
                "attributes": {"friendly_name": entity_clean, "simulated": True},
                "dry_run": True,
            }

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        url = f"{self.base_url}/api/states/{entity_clean}"

        async with await self._get_client() as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 404:
                raise ValueError(f"Entity '{entity_clean}' not found in Home Assistant.")
            resp.raise_for_status()
            return resp.json()

    async def list_allowed_states(self) -> list[dict[str, Any]]:
        """List state for all allowlisted entities."""
        if not self.allowed_entities:
            return []

        if self.dry_run:
            return [
                {
                    "entity_id": e,
                    "state": "off",
                    "attributes": {"friendly_name": e},
                    "dry_run": True,
                }
                for e in sorted(self.allowed_entities)
            ]

        results = []
        for e in sorted(self.allowed_entities):
            try:
                st = await self.get_state(e)
                results.append(st)
            except Exception as exc:
                results.append({"entity_id": e, "error": str(exc)})
        return results

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        service_data: Optional[dict[str, Any]] = None,
        dry_run_override: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Call a Home Assistant service on an allowed entity."""
        entity_clean = entity_id.strip().lower()
        if not self.is_entity_allowed(entity_clean):
            raise EntityNotAllowedError(f"Entity '{entity_clean}' is not in the Home Assistant allowlist.")

        is_dry_run = self.dry_run if dry_run_override is None else dry_run_override

        payload = {"entity_id": entity_clean}
        if service_data:
            payload.update(service_data)

        if is_dry_run:
            logger.info("[DRY RUN] Would call HA service %s.%s on %s with %s", domain, service, entity_clean, payload)
            return {
                "status": "dry_run_simulated",
                "domain": domain,
                "service": service,
                "entity_id": entity_clean,
                "data": payload,
                "message": f"[DRY RUN] Successfully simulated {domain}.{service} for {entity_clean}.",
            }

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        url = f"{self.base_url}/api/services/{domain}/{service}"

        async with await self._get_client() as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return {
                "status": "success",
                "domain": domain,
                "service": service,
                "entity_id": entity_clean,
                "data": payload,
            }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class HAListEntitiesTool(BaseTool):
    """List all allowed Home Assistant smart-home devices and their current states."""

    name = "ha_list_entities"
    description = "List all smart-home devices in the configured Home Assistant allowlist and their states."
    tier = PermissionTier.READ_ONLY

    def __init__(self, client: Optional[HomeAssistantClient] = None):
        self.client = client or HomeAssistantClient()

    async def run(self) -> list[dict[str, Any]]:
        return await self.client.list_allowed_states()


class HAGetStateArgs(BaseModel):
    entity_id: str = Field(..., description="Home Assistant entity ID (e.g. 'light.living_room')")


class HAGetStateTool(BaseTool):
    """Get the current state and attributes of a specific allowed Home Assistant entity."""

    name = "ha_get_state"
    description = "Retrieve current state and attributes of an allowed smart home entity."
    tier = PermissionTier.READ_ONLY
    args_schema = HAGetStateArgs

    def __init__(self, client: Optional[HomeAssistantClient] = None):
        self.client = client or HomeAssistantClient()

    async def run(self, entity_id: str) -> dict[str, Any]:
        return await self.client.get_state(entity_id)


class HATurnOnArgs(BaseModel):
    entity_id: str = Field(..., description="Entity ID to turn on (e.g. 'light.kitchen')")
    dry_run: bool = Field(default=False, description="Simulate without calling Home Assistant")


class HATurnOnTool(BaseTool):
    """Turn on a light, switch, or climate entity."""

    name = "ha_turn_on"
    description = "Turn on an allowlisted smart-home entity (light, switch, etc.)."
    tier = PermissionTier.REVERSIBLE
    args_schema = HATurnOnArgs

    def __init__(self, client: Optional[HomeAssistantClient] = None):
        self.client = client or HomeAssistantClient()

    async def run(self, entity_id: str, dry_run: bool = False) -> dict[str, Any]:
        if self.client.is_security_entity(entity_id):
            raise ValueError(f"Entity '{entity_id}' is a security-critical device and cannot be controlled via ha_turn_on.")
        domain = entity_id.split(".")[0] if "." in entity_id else "homeassistant"
        return await self.client.call_service(domain, "turn_on", entity_id, dry_run_override=dry_run)


class HATurnOffArgs(BaseModel):
    entity_id: str = Field(..., description="Entity ID to turn off (e.g. 'light.kitchen')")
    dry_run: bool = Field(default=False, description="Simulate without calling Home Assistant")


class HATurnOffTool(BaseTool):
    """Turn off a light, switch, or climate entity."""

    name = "ha_turn_off"
    description = "Turn off an allowlisted smart-home entity (light, switch, etc.)."
    tier = PermissionTier.REVERSIBLE
    args_schema = HATurnOffArgs

    def __init__(self, client: Optional[HomeAssistantClient] = None):
        self.client = client or HomeAssistantClient()

    async def run(self, entity_id: str, dry_run: bool = False) -> dict[str, Any]:
        if self.client.is_security_entity(entity_id):
            raise ValueError(f"Entity '{entity_id}' is a security-critical device and cannot be controlled via ha_turn_off.")
        domain = entity_id.split(".")[0] if "." in entity_id else "homeassistant"
        return await self.client.call_service(domain, "turn_off", entity_id, dry_run_override=dry_run)


class HASetValueArgs(BaseModel):
    entity_id: str = Field(..., description="Entity ID to adjust")
    brightness: Optional[int] = Field(default=None, description="Brightness (0-255) for lights")
    temperature: Optional[float] = Field(default=None, description="Target temperature for climate")
    dry_run: bool = Field(default=False, description="Simulate without calling Home Assistant")


class HASetValueTool(BaseTool):
    """Set brightness or temperature for an allowed smart-home device."""

    name = "ha_set_value"
    description = "Set light brightness or thermostat target temperature for an allowlisted entity."
    tier = PermissionTier.REVERSIBLE
    args_schema = HASetValueArgs

    def __init__(self, client: Optional[HomeAssistantClient] = None):
        self.client = client or HomeAssistantClient()

    async def run(
        self,
        entity_id: str,
        brightness: Optional[int] = None,
        temperature: Optional[float] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if self.client.is_security_entity(entity_id):
            raise ValueError(f"Entity '{entity_id}' is security-critical and cannot be controlled via ha_set_value.")

        domain = entity_id.split(".")[0] if "." in entity_id else "homeassistant"
        service_data = {}
        if domain == "light" and brightness is not None:
            service = "turn_on"
            service_data["brightness"] = max(0, min(255, brightness))
        elif domain == "climate" and temperature is not None:
            service = "set_temperature"
            service_data["temperature"] = temperature
        else:
            service = "turn_on"

        return await self.client.call_service(domain, service, entity_id, service_data, dry_run_override=dry_run)


class HASecurityControlArgs(BaseModel):
    entity_id: str = Field(..., description="Security entity ID (e.g. 'lock.front_door', 'cover.garage')")
    action: str = Field(..., description="Action to perform: 'lock', 'unlock', 'open', 'close', 'arm', 'disarm'")
    dry_run: bool = Field(default=False, description="Simulate without calling Home Assistant")


class HASecurityControlTool(BaseTool):
    """RESTRICTED tool to control locks, garage doors, and alarm systems. Requires explicit user confirmation."""

    name = "ha_security_control"
    description = "Control locks, alarms, and garage doors. RESTRICTED: requires explicit user confirmation."
    tier = PermissionTier.RESTRICTED
    args_schema = HASecurityControlArgs

    def __init__(self, client: Optional[HomeAssistantClient] = None, enabled: bool = False):
        self.client = client or HomeAssistantClient()
        self.enabled = enabled or self.client.settings.ha_enable_restricted

    async def run(self, entity_id: str, action: str, dry_run: bool = False) -> dict[str, Any]:
        if not self.enabled:
            raise PermissionError("Control of security devices (locks, garage doors, alarms) is disabled in configuration.")

        if not self.client.is_security_entity(entity_id):
            raise ValueError(f"Entity '{entity_id}' is not classified as a security device.")

        domain = entity_id.split(".")[0]
        service_map = {
            "lock": {"lock": "lock", "unlock": "unlock"},
            "cover": {"open": "open_cover", "close": "close_cover"},
            "alarm_control_panel": {"arm": "alarm_arm_home", "disarm": "alarm_disarm"},
        }
        domain_actions = service_map.get(domain, {})
        service = domain_actions.get(action.lower())
        if not service:
            raise ValueError(f"Action '{action}' is not supported for domain '{domain}'.")

        return await self.client.call_service(domain, service, entity_id, dry_run_override=dry_run)
