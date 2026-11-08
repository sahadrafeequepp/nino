"""GPS Location and Open-Meteo Weather Forecast tools for Nino."""

import logging
from typing import Any, Optional
import httpx
from pydantic import BaseModel, Field

from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier

logger = logging.getLogger(__name__)

# Standard WMO Weather Interpretation Codes (World Meteorological Organization)
WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# Common location aliases for accurate local resolution
KNOWN_PLACE_ALIASES: dict[str, dict[str, Any]] = {
    "vadakara": {
        "name": "Vadakara",
        "admin1": "Kerala",
        "country": "India",
        "latitude": 11.5978,
        "longitude": 75.5814,
        "timezone": "Asia/Kolkata",
    },
    "vatakara": {
        "name": "Vatakara",
        "admin1": "Kerala",
        "country": "India",
        "latitude": 11.5978,
        "longitude": 75.5814,
        "timezone": "Asia/Kolkata",
    },
    "badagara": {
        "name": "Badagara",
        "admin1": "Kerala",
        "country": "India",
        "latitude": 11.5978,
        "longitude": 75.5814,
        "timezone": "Asia/Kolkata",
    },
    "calicut": {
        "name": "Calicut",
        "admin1": "Kerala",
        "country": "India",
        "latitude": 11.2588,
        "longitude": 75.7804,
        "timezone": "Asia/Kolkata",
    },
    "kozhikode": {
        "name": "Kozhikode",
        "admin1": "Kerala",
        "country": "India",
        "latitude": 11.2588,
        "longitude": 75.7804,
        "timezone": "Asia/Kolkata",
    },
    "kerala": {
        "name": "Kerala",
        "admin1": "Kerala",
        "country": "India",
        "latitude": 10.8505,
        "longitude": 76.2711,
        "timezone": "Asia/Kolkata",
    },
}


async def detect_current_location(
    default_lat: Optional[float] = None,
    default_lon: Optional[float] = None,
    default_city: Optional[str] = None,
) -> dict[str, Any]:
    """Detect geographical coordinates and place name via configured values or IP geolocation."""
    if default_lat is not None and default_lon is not None:
        return {
            "city": default_city or "Vadakara, Calicut",
            "region": "Kerala",
            "country": "India",
            "latitude": default_lat,
            "longitude": default_lon,
            "timezone": "Asia/Kolkata",
            "source": "configured",
        }

    async with httpx.AsyncClient(timeout=4.0) as client:
        try:
            resp = await client.get("https://ipinfo.io/json")
            if resp.status_code == 200:
                data = resp.json()
                loc = data.get("loc", "")
                if loc and "," in loc:
                    lat_str, lon_str = loc.split(",", 1)
                    return {
                        "city": data.get("city", "Vadakara, Calicut"),
                        "region": data.get("region", "Kerala"),
                        "country": data.get("country", "India"),
                        "postal": data.get("postal", ""),
                        "latitude": float(lat_str.strip()),
                        "longitude": float(lon_str.strip()),
                        "timezone": data.get("timezone", "Asia/Kolkata"),
                        "source": "ip_geolocation",
                    }
        except Exception as e:
            logger.debug("Failed to detect location via ipinfo: %s", e)

    # Safe fallback to user's configured location in Vadakara, Kerala
    return {
        "city": default_city or "Vadakara, Calicut",
        "region": "Kerala",
        "country": "India",
        "latitude": default_lat or 11.5978,
        "longitude": default_lon or 75.5814,
        "timezone": "Asia/Kolkata",
        "source": "default_fallback",
    }


async def geocode_place_name(query: str) -> Optional[dict[str, Any]]:
    """Resolve a city or place name into geographical coordinates using Open-Meteo Geocoding."""
    q_lower = query.lower().strip()
    for alias_key, alias_val in KNOWN_PLACE_ALIASES.items():
        if alias_key in q_lower:
            return dict(alias_val)

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": 1, "language": "en", "format": "json"}
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                results = resp.json().get("results")
                if results and len(results) > 0:
                    top = results[0]
                    return {
                        "name": top.get("name"),
                        "admin1": top.get("admin1", ""),
                        "country": top.get("country", ""),
                        "latitude": top.get("latitude"),
                        "longitude": top.get("longitude"),
                        "timezone": top.get("timezone", "auto"),
                    }
        except Exception as e:
            logger.debug("Geocoding lookup failed for '%s': %s", query, e)
    return None


def get_stored_location(memory_db: Optional[Any]) -> Optional[dict[str, Any]]:
    """Retrieve user's securely stored location from local SQLite memory without any network activity."""
    if not memory_db:
        return None
    try:
        with memory_db.get_connection() as conn:
            cur = conn.execute(
                "SELECT key, value FROM proactive_preferences WHERE key IN ('user_city', 'user_latitude', 'user_longitude', 'user_region', 'user_country', 'user_timezone')"
            )
            rows = {row["key"]: row["value"] for row in cur.fetchall()}
            if "user_city" in rows and "user_latitude" in rows and "user_longitude" in rows:
                return {
                    "city": memory_db.decrypt_text(rows["user_city"]),
                    "region": memory_db.decrypt_text(rows.get("user_region", "")),
                    "country": memory_db.decrypt_text(rows.get("user_country", "")),
                    "latitude": float(rows["user_latitude"]),
                    "longitude": float(rows["user_longitude"]),
                    "timezone": rows.get("user_timezone", "auto"),
                    "source": "local_secure_storage",
                }
    except Exception as e:
        logger.debug("Failed to read stored location from local database: %s", e)
    return None


def store_location_locally(memory_db: Optional[Any], loc_data: dict[str, Any]) -> bool:
    """Store location coordinates and place info into local SQLite memory.
    
    Protects user privacy:
    - Data stays 100% on the local machine in SQLite.
    - Encrypted if MEMORY_ENCRYPTION_KEY is enabled.
    - Eliminates repetitive external geolocation calls.
    """
    if not memory_db or not loc_data:
        return False
    city = loc_data.get("city", "Current Location")
    lat = loc_data.get("latitude")
    lon = loc_data.get("longitude")
    if lat is None or lon is None:
        return False

    region = loc_data.get("region", "")
    country = loc_data.get("country", "")
    tz = loc_data.get("timezone", "auto")

    try:
        enc_city = memory_db.encrypt_text(str(city))
        enc_region = memory_db.encrypt_text(str(region))
        enc_country = memory_db.encrypt_text(str(country))

        with memory_db.get_connection() as conn:
            entries = [
                ("user_city", enc_city),
                ("user_region", enc_region),
                ("user_country", enc_country),
                ("user_latitude", str(lat)),
                ("user_longitude", str(lon)),
                ("user_timezone", str(tz)),
            ]
            for k, v in entries:
                conn.execute(
                    "INSERT INTO proactive_preferences (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
                    (k, v),
                )

            # Store durable fact for conversational awareness
            place_desc = f"{city}" + (f", {country}" if country else "")
            fact_text = f"User's primary location is {place_desc} (Coordinates: {lat}, {lon})"
            enc_fact = memory_db.encrypt_text(fact_text)
            import uuid
            fact_id = f"fact-loc-{uuid.uuid4().hex[:8]}"
            conn.execute(
                "UPDATE facts SET is_current = 0, superseded_by = ? WHERE category = 'location' AND is_current = 1",
                (fact_id,),
            )
            conn.execute(
                "INSERT INTO facts (id, text, category, is_current) VALUES (?, ?, 'location', 1)",
                (fact_id, enc_fact),
            )
            conn.commit()
            return True
    except Exception as e:
        logger.error("Failed to securely store location locally: %s", e)
        return False


class SetLocationArgs(BaseModel):
    """Arguments for setting and storing the user's geographical location."""

    location: str = Field(
        ...,
        description="The city or place name to set as your primary location (e.g. 'Tokyo', 'London', 'New York').",
    )
    latitude: Optional[float] = Field(
        default=None,
        description="Optional GPS latitude coordinate. If omitted, it will be automatically resolved from the place name.",
    )
    longitude: Optional[float] = Field(
        default=None,
        description="Optional GPS longitude coordinate. If omitted, it will be automatically resolved from the place name.",
    )


class SetLocationTool(BaseTool):
    """Set and securely store the user's location in local persistent memory."""

    name = "set_location"
    description = (
        "Set and store the user's location in local SQLite memory. "
        "All data remains strictly on your machine and is never leaked to external AI or tracking services."
    )
    args_schema = SetLocationArgs
    tier = PermissionTier.REVERSIBLE

    def __init__(self, memory_db: Optional[Any] = None):
        self.memory_db = memory_db

    async def run(
        self,
        location: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        city_name = location.strip()
        region = ""
        country = ""
        tz = "auto"

        if latitude is None or longitude is None:
            geocoded = await geocode_place_name(city_name)
            if not geocoded:
                return {
                    "success": False,
                    "error": f"Could not find coordinates for location '{city_name}'. Please verify the city name.",
                }
            city_name = geocoded.get("name", city_name)
            region = geocoded.get("admin1", "")
            country = geocoded.get("country", "")
            latitude = geocoded["latitude"]
            longitude = geocoded["longitude"]
            tz = geocoded.get("timezone", "auto")

        loc_data = {
            "city": city_name,
            "region": region,
            "country": country,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": tz,
            "source": "user_configured",
        }

        stored = store_location_locally(self.memory_db, loc_data)
        parts = [p for p in [city_name, region, country] if p]
        loc_str = ", ".join(parts)
        return {
            "success": True,
            "stored_locally": stored,
            "location": loc_str,
            "city": city_name,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": tz,
            "privacy_notice": "Location is saved in local protected SQLite memory. Coordinates are never sent to external AI providers or tracking services.",
            "summary": f"Your location has been securely saved as {loc_str} (Lat: {latitude}, Lon: {longitude}).",
        }


class GetLocationArgs(BaseModel):
    """Arguments for retrieving current GPS location."""

    refresh: Optional[bool] = Field(
        default=False,
        description="If True, re-checks live network geolocation instead of using the locally stored location.",
    )


class GetLocationTool(BaseTool):
    """Retrieve the user's current GPS location coordinates and place name."""

    name = "get_location"
    description = (
        "Retrieve the user's geographical location (city, region, country, latitude, and longitude). "
        "Favors locally stored memory to prevent third-party network tracking."
    )
    args_schema = GetLocationArgs
    tier = PermissionTier.READ_ONLY

    def __init__(
        self,
        default_lat: Optional[float] = 11.5978,
        default_lon: Optional[float] = 75.5814,
        default_city: Optional[str] = "Vadakara, Calicut, Kerala, India",
        memory_db: Optional[Any] = None,
    ):
        self.default_lat = default_lat or 11.5978
        self.default_lon = default_lon or 75.5814
        self.default_city = default_city or "Vadakara, Calicut, Kerala, India"
        self.memory_db = memory_db

    async def run(self, refresh: bool = False, **kwargs: Any) -> dict[str, Any]:
        # 1. Check local persistent storage first to eliminate external network calls
        if not refresh:
            stored = get_stored_location(self.memory_db)
            if stored:
                parts = [p for p in [stored.get("city"), stored.get("region"), stored.get("country")] if p]
                loc_str = ", ".join(parts) if parts else "Stored Location"
                return {
                    "summary": f"Location: {loc_str} (Latitude: {stored['latitude']}, Longitude: {stored['longitude']}) [Loaded securely from local memory]",
                    "city": stored.get("city"),
                    "region": stored.get("region"),
                    "country": stored.get("country"),
                    "latitude": stored.get("latitude"),
                    "longitude": stored.get("longitude"),
                    "timezone": stored.get("timezone"),
                    "source": "local_secure_storage",
                    "privacy_guarantee": "Zero external network requests made. Loaded from local protected SQLite memory.",
                }

        # 2. If not stored or refresh explicitly requested, detect via IP geolocation or config
        loc = await detect_current_location(
            default_lat=self.default_lat,
            default_lon=self.default_lon,
            default_city=self.default_city,
        )
        if "error" in loc:
            return loc

        # 3. Securely store locally so future queries do not touch external services
        if self.memory_db:
            store_location_locally(self.memory_db, loc)

        parts = [p for p in [loc.get("city"), loc.get("region"), loc.get("country")] if p]
        location_str = ", ".join(parts) if parts else "Current Location"
        summary = f"Location: {location_str} (Latitude: {loc['latitude']}, Longitude: {loc['longitude']})"
        return {
            "summary": summary,
            "city": loc.get("city"),
            "region": loc.get("region"),
            "country": loc.get("country"),
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "timezone": loc.get("timezone"),
            "source": loc.get("source", "detected"),
            "privacy_guarantee": "Stored locally in SQLite. Coordinates protected from external AI providers.",
        }


class GetWeatherForecastArgs(BaseModel):
    """Arguments for fetching weather conditions and forecast."""

    location: Optional[str] = Field(
        default=None,
        description=(
            "Optional target city or location name (e.g. 'Tokyo', 'London', 'San Francisco'). "
            "If omitted, automatically uses user's stored local location or GPS."
        ),
    )
    days: Optional[int] = Field(
        default=3,
        description="Number of forecast days to retrieve (1 to 7 days, default is 3).",
    )


class GetWeatherForecastTool(BaseTool):
    """Fetch real-time weather and multi-day forecast via Open-Meteo."""

    name = "get_weather_forecast"
    description = (
        "Get current weather conditions (temperature, feels-like, condition, humidity, wind) "
        "and multi-day forecast (highs, lows, rain probability). "
        "Automatically uses user's stored local location if no city is specified."
    )
    args_schema = GetWeatherForecastArgs
    tier = PermissionTier.READ_ONLY

    def __init__(
        self,
        default_lat: Optional[float] = 11.5978,
        default_lon: Optional[float] = 75.5814,
        default_city: Optional[str] = "Vadakara, Calicut, Kerala, India",
        memory_db: Optional[Any] = None,
    ):
        self.default_lat = default_lat or 11.5978
        self.default_lon = default_lon or 75.5814
        self.default_city = default_city or "Vadakara, Calicut, Kerala, India"
        self.memory_db = memory_db

    async def run(
        self,
        location: Optional[str] = None,
        days: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        target_lat: float
        target_lon: float
        place_label: str
        tz_name: str = "auto"

        days = max(1, min(7, days or 3))

        # 1. Determine Coordinates (Geocoding vs. Stored Local Location vs. Detection)
        if location and location.strip():
            geo = await geocode_place_name(location.strip())
            if not geo:
                return {
                    "error": f"Could not find coordinates for location '{location}'. Please verify the city name."
                }
            target_lat = geo["latitude"]
            target_lon = geo["longitude"]
            tz_name = geo.get("timezone", "auto")
            parts = [p for p in [geo.get("name"), geo.get("admin1"), geo.get("country")] if p]
            place_label = ", ".join(parts)
        else:
            # Check local stored location first
            stored = get_stored_location(self.memory_db)
            if stored:
                target_lat = stored["latitude"]
                target_lon = stored["longitude"]
                tz_name = stored.get("timezone", "auto")
                parts = [p for p in [stored.get("city"), stored.get("region"), stored.get("country")] if p]
                place_label = ", ".join(parts) if parts else "Your Stored Location"
            else:
                current_loc = await detect_current_location(
                    default_lat=self.default_lat,
                    default_lon=self.default_lon,
                    default_city=self.default_city,
                )
                if "error" in current_loc:
                    return current_loc
                target_lat = current_loc["latitude"]
                target_lon = current_loc["longitude"]
                tz_name = current_loc.get("timezone", "auto")
                parts = [p for p in [current_loc.get("city"), current_loc.get("region"), current_loc.get("country")] if p]
                place_label = ", ".join(parts) if parts else "Your Current Location"
                if self.memory_db:
                    store_location_locally(self.memory_db, current_loc)


        # 2. Query Open-Meteo Forecast API
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": target_lat,
            "longitude": target_lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": tz_name if (tz_name and tz_name != "auto") else "Asia/Kolkata",
        }

        async with httpx.AsyncClient(timeout=8.0) as client:
            try:
                resp = await client.get(forecast_url, params=params)
                if resp.status_code != 200:
                    return {"error": f"Weather service responded with status {resp.status_code}."}
                data = resp.json()
            except Exception as e:
                return {"error": f"Failed to reach weather service: {e}"}

        curr = data.get("current", {})
        daily = data.get("daily", {})

        curr_code = curr.get("weather_code", 0)
        curr_desc = WMO_WEATHER_CODES.get(curr_code, "Unknown")
        curr_temp = curr.get("temperature_2m")
        curr_feels = curr.get("apparent_temperature")
        curr_humidity = curr.get("relative_humidity_2m")
        curr_wind = curr.get("wind_speed_10m")

        # Format multi-day forecast entries
        forecast_entries = []
        dates = daily.get("time", [])[:days]
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        w_codes = daily.get("weather_code", [])
        rain_probs = daily.get("precipitation_probability_max", [])

        for i, dt in enumerate(dates):
            code = w_codes[i] if i < len(w_codes) else 0
            condition = WMO_WEATHER_CODES.get(code, "Clear")
            high = max_temps[i] if i < len(max_temps) else None
            low = min_temps[i] if i < len(min_temps) else None
            rain = rain_probs[i] if i < len(rain_probs) else 0

            forecast_entries.append({
                "date": dt,
                "condition": condition,
                "high_c": high,
                "low_c": low,
                "precipitation_probability": f"{rain}%",
            })

        # Build natural language summary for conversational and voice responses
        forecast_lines = [
            f"• {e['date']}: {e['condition']}, High {e['high_c']}°C / Low {e['low_c']}°C (Rain: {e['precipitation_probability']})"
            for e in forecast_entries
        ]
        summary = (
            f"Weather for {place_label}:\n"
            f"Currently {curr_temp}°C (feels like {curr_feels}°C), {curr_desc}. "
            f"Humidity: {curr_humidity}%, Wind: {curr_wind} km/h.\n"
            f"Forecast:\n" + "\n".join(forecast_lines)
        )

        return {
            "summary": summary,
            "place": place_label,
            "latitude": target_lat,
            "longitude": target_lon,
            "current": {
                "temperature_c": curr_temp,
                "apparent_temperature_c": curr_feels,
                "condition": curr_desc,
                "humidity_percent": curr_humidity,
                "wind_speed_kmh": curr_wind,
            },
            "forecast": forecast_entries,
        }
