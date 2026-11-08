"""Unit and integration tests for GetLocationTool and GetWeatherForecastTool."""

import pytest
from unittest.mock import AsyncMock, patch
import httpx

from nino.tools.weather import (
    GetLocationTool,
    GetWeatherForecastTool,
    WMO_WEATHER_CODES,
    detect_current_location,
    geocode_place_name,
)
from nino.tools.permissions import PermissionTier
from nino.config import Settings
from nino.memory.db import MemoryDB
from nino.tools import build_default_registry


def test_wmo_weather_codes():
    assert WMO_WEATHER_CODES[0] == "Clear sky"
    assert WMO_WEATHER_CODES[95] == "Thunderstorm"
    assert 61 in WMO_WEATHER_CODES


def test_tool_metadata():
    loc_tool = GetLocationTool()
    assert loc_tool.name == "get_location"
    assert loc_tool.tier == PermissionTier.READ_ONLY

    weather_tool = GetWeatherForecastTool()
    assert weather_tool.name == "get_weather_forecast"
    assert weather_tool.tier == PermissionTier.READ_ONLY


@pytest.mark.asyncio
async def test_get_location_tool_configured():
    tool = GetLocationTool(
        default_lat=37.7749,
        default_lon=-122.4194,
        default_city="San Francisco",
    )
    res = await tool.run()
    assert res["city"] == "San Francisco"
    assert res["latitude"] == 37.7749
    assert res["longitude"] == -122.4194
    assert "San Francisco" in res["summary"]


@pytest.mark.asyncio
async def test_get_weather_forecast_mocked():
    tool = GetWeatherForecastTool(
        default_lat=40.7128,
        default_lon=-74.0060,
        default_city="New York",
    )

    mock_resp_data = {
        "current": {
            "temperature_2m": 22.5,
            "apparent_temperature": 23.0,
            "relative_humidity_2m": 65,
            "wind_speed_10m": 12.0,
            "weather_code": 1,
        },
        "daily": {
            "time": ["2026-10-04", "2026-10-05"],
            "temperature_2m_max": [24.0, 23.5],
            "temperature_2m_min": [16.0, 15.0],
            "weather_code": [0, 61],
            "precipitation_probability_max": [10, 80],
        },
    }

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: mock_resp_data

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        res = await tool.run(days=2)
        assert "New York" in res["place"]
        assert res["current"]["temperature_c"] == 22.5
        assert res["current"]["condition"] == "Mainly clear"
        assert len(res["forecast"]) == 2
        assert res["forecast"][0]["condition"] == "Clear sky"
        assert res["forecast"][1]["condition"] == "Slight rain"
        assert "22.5°C" in res["summary"]


def test_tools_registered_in_default_registry(tmp_path):
    db_file = tmp_path / "test.db"
    db = MemoryDB(db_path=str(db_file))
    settings = Settings(
        default_latitude=51.5074,
        default_longitude=-0.1278,
        default_city="London",
    )
    registry = build_default_registry(db, settings)

    assert registry.get("get_location") is not None
    assert registry.get("set_location") is not None
    assert registry.get("get_weather_forecast") is not None

    loc_tool = registry.get("get_location")
    assert loc_tool.default_city == "London"
    assert loc_tool.default_lat == 51.5074


@pytest.mark.asyncio
async def test_set_location_and_local_storage(tmp_path):
    from nino.tools.weather import SetLocationTool, get_stored_location
    db_file = tmp_path / "loc_test.db"
    db = MemoryDB(db_path=str(db_file))

    set_tool = SetLocationTool(memory_db=db)
    res = await set_tool.run(location="Seattle", latitude=47.6062, longitude=-122.3321)
    assert res["success"] is True
    assert res["stored_locally"] is True
    assert "Seattle" in res["location"]

    # Verify stored in DB
    stored = get_stored_location(db)
    assert stored is not None
    assert stored["city"] == "Seattle"
    assert stored["latitude"] == 47.6062
    assert stored["longitude"] == -122.3321
    assert stored["source"] == "local_secure_storage"

    # Verify GetLocationTool uses local storage with zero network activity
    get_tool = GetLocationTool(memory_db=db)
    get_res = await get_tool.run(refresh=False)
    assert get_res["source"] == "local_secure_storage"
    assert get_res["city"] == "Seattle"
    assert "local memory" in get_res["summary"]


def test_gps_coordinates_privacy_redaction():
    from nino.security.privacy_gateway import PrivacyGateway
    gw = PrivacyGateway(allow_external=True)

    text_with_coords = "User lives at Latitude: 37.7749, Longitude: -122.4194 in California."
    sanitized = gw.redact_pii(text_with_coords)
    assert "[REDACTED_COORDINATES]" in sanitized
    assert "37.7749" not in sanitized
    assert "-122.4194" not in sanitized

