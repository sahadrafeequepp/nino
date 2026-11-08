# Home Assistant Integration Guide: Nino Personal AI Assistant

## 1. Overview & Security Model

Nino integrates with **Home Assistant** to let you inspect and control smart-home devices safely through conversational commands (text, voice, and proactive automations).

In strict adherence to **Nino's Workspace Rules and Invariants**:
1. **Strict Entity Allowlist**: Nino can **only** inspect or control entities explicitly declared in your `HA_ALLOWED_ENTITIES` configuration. Any command targeting an unlisted entity is rejected immediately.
2. **Permission Tiers**:
   - `READ_ONLY`: Device inspection (`ha_list_entities`, `ha_get_state`).
   - `REVERSIBLE`: State mutations on daily appliances (`ha_turn_on`, `ha_turn_off`, `ha_set_value` for lights, switches, climate).
   - `RESTRICTED`: Security devices (`ha_security_control` for locks, garage doors, alarms, cameras). **Disabled by default** (`HA_ENABLE_RESTRICTED=false`), and if enabled, **always requires explicit human confirmation**.
3. **Dry-Run Simulation**: A built-in dry-run mode (`HA_DRY_RUN=true` or tool argument `dry_run=true`) simulates and logs state changes without sending network requests to Home Assistant.
4. **Token Quarantine**: Long-lived access tokens (`HA_TOKEN`) are never logged in audit files, console outputs, or error traces.

---

## 2. Configuration (`.env`)

Add the following environment variables to your `.env` file:

```env
# Home Assistant Gateway URL
HA_URL=http://127.0.0.1:8123

# Long-Lived Access Token (create in Home Assistant: Profile -> Long-Lived Access Tokens)
HA_TOKEN=your_long_lived_access_token_here

# Entity Allowlist (comma-separated). Unlisted devices are invisible and inaccessible to Nino.
HA_ALLOWED_ENTITIES=light.living_room,light.desk,switch.coffee_maker,climate.thermostat

# Dry-Run Mode (true: simulate only; false: send live commands)
HA_DRY_RUN=false

# Security Device Control (locks, garage doors, alarms)
# Keep false to outright refuse control over security boundaries.
HA_ENABLE_RESTRICTED=false
```

---

## 3. Available Tools & Capabilities

| Tool | Permission Tier | Allowed Devices | Description |
|---|---|---|---|
| `ha_list_entities` | `READ_ONLY` | All allowlisted entities | Lists all configured smart devices and their current states. |
| `ha_get_state` | `READ_ONLY` | Specific allowlisted entity | Retrieves state and attributes (brightness, temperature). |
| `ha_turn_on` | `REVERSIBLE` | Lights, switches, climate | Turns on a device. Security devices are blocked. |
| `ha_turn_off` | `REVERSIBLE` | Lights, switches, climate | Turns off a device. Security devices are blocked. |
| `ha_set_value` | `REVERSIBLE` | Lights, climate | Adjusts brightness (0–255) or target temperature (°C). |
| `ha_security_control` | `RESTRICTED` | Locks, covers, alarms | Locks/unlocks doors, opens/closes garage. Disabled unless `HA_ENABLE_RESTRICTED=true` and approved. |

---

## 4. Usage Examples in Chat or Voice

- *"What lights are currently on?"*
  Nino calls `ha_list_entities` and reports states for only your allowlisted devices.

- *"Turn on the desk lamp."*
  Nino calls `ha_turn_on` with `{"entity_id": "light.desk"}`.

- *"Dim the living room light to 50%."*
  Nino calls `ha_set_value` with `{"entity_id": "light.living_room", "brightness": 128}`.

- *"Unlock the front door."*
  - If `HA_ENABLE_RESTRICTED=false`: Nino refuses the request: *"Control of security devices (locks, garage doors, alarms) is disabled in configuration."*
  - If `HA_ENABLE_RESTRICTED=true`: Nino generates a cryptographic confirmation token and prompts you: *"Confirm: Unlock front door? [y/N]"*. It executes only after explicit approval.

---

## 5. Troubleshooting & Safety Verification

1. **Entity Rejected**:
   If Nino responds that an entity is not in the allowlist, verify that its exact ID (e.g. `light.study_lamp`) is included in `HA_ALLOWED_ENTITIES`.
2. **Testing in Dry-Run**:
   Set `HA_DRY_RUN=true` to test voice and chat commands safely. Nino will describe the action it would take without altering any physical hardware.
