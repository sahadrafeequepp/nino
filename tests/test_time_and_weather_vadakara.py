"""Test accurate local time for India/Kerala and weather forecast for Vadakara, Calicut."""

import pytest
from nino.memory.db import MemoryDB
from nino.tools.utilities import CurrentTimeTool
from nino.tools.weather import GetLocationTool, GetWeatherForecastTool, geocode_place_name, get_stored_location
from nino.persona.manager import PersonaManager


@pytest.mark.asyncio
async def test_current_time_shows_local_time():
    tool = CurrentTimeTool(default_timezone="Asia/Kolkata", location="Vadakara, Calicut, Kerala, India")
    res = await tool.run()
    assert "Current Local Time:" in res
    assert "Vadakara, Calicut, Kerala, India" in res
    assert "24-Hour Format:" in res
    # Should not show UTC on the first line
    lines = res.strip().split("\n")
    assert lines[0].startswith("Current Local Time:")


@pytest.mark.asyncio
async def test_geocoding_vadakara_and_calicut_alias():
    # Vadakara
    geo_v = await geocode_place_name("Vadakara")
    assert geo_v is not None
    assert geo_v["admin1"] == "Kerala"
    assert geo_v["country"] == "India"
    assert round(geo_v["latitude"], 2) == 11.60 or round(geo_v["latitude"], 2) == 11.59 or round(geo_v["latitude"], 2) == 11.61

    # Calicut
    geo_c = await geocode_place_name("Calicut")
    assert geo_c is not None
    assert geo_c["admin1"] == "Kerala"
    assert geo_c["country"] == "India"


@pytest.mark.asyncio
async def test_weather_forecast_for_vadakara(tmp_path):
    db_file = tmp_path / "test_w.db"
    db = MemoryDB(str(db_file))
    from nino.tools.weather import store_location_locally
    store_location_locally(
        db,
        {
            "city": "Vadakara, Calicut",
            "region": "Kerala",
            "country": "India",
            "latitude": 11.5978,
            "longitude": 75.5814,
            "timezone": "Asia/Kolkata",
        },
    )

    weather_tool = GetWeatherForecastTool(memory_db=db)
    res = await weather_tool.run(days=3)
    assert "summary" in res
    assert "Weather for Vadakara, Calicut" in res["summary"]
    assert "Forecast:" in res["summary"]
    assert len(res["forecast"]) == 3
    assert "temperature_c" in res["current"]


def test_persona_system_prompt_includes_time_and_location():
    mgr = PersonaManager()
    prompt = mgr.build_system_prompt(location_info="Vadakara, Calicut, Kerala, India")
    assert "[Current Time & Location Context]" in prompt
    assert "Vadakara, Calicut, Kerala, India" in prompt
    assert "Current Date & Time:" in prompt
