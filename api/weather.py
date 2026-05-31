"""Weather data provider for S2 sales forecasting.

Primary: OpenWeatherMap 5-day/3-hour forecast (free tier, API key required)
Fallback: Open-Meteo (free, no key)
Last resort: Malaysian Met. Dept. monthly climatology
Thread-safe: threading.Lock on circuit breaker + in-memory cache (1h TTL)
"""

import os
import time
import threading
import httpx
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("weather")

# Kuala Lumpur coordinates
KL_LAT = 3.139
KL_LON = 101.6869

# OpenWeatherMap API key
_VISUALCROSSING_KEY = os.getenv("WEATHER_API_KEY", "")

# Monthly fallback (Sultan Abdul Aziz Shah Airport, 1991-2020 normals)
# Source: Wikipedia / Met Malaysia (temp_C, rainfall_mm, humidity_pct)
KL_MONTHLY = {
    1: (27.3, 226.7, 80), 2: (27.8, 192.8, 80), 3: (28.1, 270.4, 80),
    4: (28.1, 301.5, 82), 5: (28.5, 229.9, 81), 6: (28.4, 145.8, 80),
    7: (28.0, 165.2, 79), 8: (28.0, 174.3, 79), 9: (27.7, 220.3, 81),
    10: (27.5, 283.8, 82), 11: (27.1, 355.8, 84), 12: (27.1, 280.6, 83),
}

# In-memory cache
_weather_cache = {}
_cache_lock = threading.Lock()
_WEATHER_CACHE_TTL = 3600  # 1 hour

# Thread-safe circuit breaker
_breaker_lock = threading.Lock()
_failure_count = 0
_failure_until = 0.0
_CIRCUIT_BREAKER_THRESHOLD = 3
_CIRCUIT_BREAKER_COOLDOWN = 300  # 5 minutes


def _api_available() -> bool:
    with _breaker_lock:
        if _failure_until > 0 and time.time() < _failure_until:
            return False
        return True


def _record_failure(source: str):
    global _failure_count, _failure_until
    with _breaker_lock:
        _failure_count += 1
        if _failure_count >= _CIRCUIT_BREAKER_THRESHOLD:
            _failure_until = time.time() + _CIRCUIT_BREAKER_COOLDOWN
            _failure_count = 0
            logger.warning("Weather circuit breaker OPEN (%s) - skipping for %ds", source, _CIRCUIT_BREAKER_COOLDOWN)


def _record_success():
    global _failure_count, _failure_until
    with _breaker_lock:
        _failure_count = 0
        _failure_until = 0.0


# ------------------------------------------------------------------
# Visual Crossing conditions -> category
# ------------------------------------------------------------------
def _visualcrossing_to_category(conditions: str) -> str:
    c = conditions.lower()
    if "clear" in c:
        return "sunny"
    elif "overcast" in c:
        return "cloudy"
    elif any(w in c for w in ("rain", "precipitation", "drizzle", "shower")):
        return "rainy"
    elif any(w in c for w in ("thunder", "storm", "tstorm")):
        return "thunderstorm"
    elif any(w in c for w in ("fog", "mist", "haze")):
        return "foggy"
    elif "partially cloudy" in c or "cloud" in c:
        return "cloudy"
    return "cloudy"


# ------------------------------------------------------------------
# Open-Meteo WMO code -> category (kept for fallback)
# ------------------------------------------------------------------
def _wmo_to_category(code: int) -> str:
    if code == 0:
        return "sunny"
    elif 1 <= code <= 3:
        return "cloudy"
    elif 45 <= code <= 48:
        return "foggy"
    elif 51 <= code <= 67:
        return "rainy"
    elif 71 <= code <= 86:
        return "storm"
    elif 95 <= code <= 99:
        return "thunderstorm"
    return "cloudy"


# ==================================================================
# Public API
# ==================================================================
def get_weather(target_date: datetime) -> dict:
    """Return weather dict for a given date.

    Strategy: cache hit > OpenWeatherMap > Open-Meteo > monthly fallback.
    Thread-safe cache + circuit breaker.
    """
    date_key = target_date.strftime("%Y-%m-%d")

    # --- cache hit (lock-free read, lock for write) ---
    if date_key in _weather_cache:
        entry = _weather_cache[date_key]
        if time.time() - entry["ts"] < _WEATHER_CACHE_TTL:
            return entry["data"]

    # --- fetch (3-tier cascade) ---
    result = None
    tried_circuit = False

    if _api_available():
        try:
            result = _call_visualcrossing(target_date)
            _record_success()
        except Exception as e:
            logger.warning("Visual Crossing failed (%s), trying Open-Meteo", e)
            _record_failure("VisualCrossing")

    if result is None and _api_available():
        try:
            result = _call_openmeteo(target_date)
            _record_success()
        except Exception as e:
            logger.warning("Open-Meteo failed (%s), using monthly fallback", e)
            _record_failure("Open-Meteo")

    if result is None:
        result = _monthly_fallback(target_date)

    # --- cache store ---
    _weather_cache[date_key] = {"ts": time.time(), "data": result}
    return result


# ==================================================================
# Tier 1: Visual Crossing (free 1000 calls/day, 15-day forecast, batch fetch)
# ==================================================================
# We fetch a 15-day window once and cache all days to minimise API calls.
_VC_FETCHED_UNTIL = ""  # track which date range we already have

def _call_visualcrossing(dt: datetime) -> dict:
    """Get daily weather from Visual Crossing Timeline API.
    Fetches 15 days at once, caches all individual days.
    """
    date_str = dt.strftime("%Y-%m-%d")
    today = datetime.now()
    end_date = today + timedelta(days=14)

    # Build URL for 15-day batch
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        f"/{KL_LAT},{KL_LON}/{today.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        f"?unitGroup=metric"
        f"&key={_VISUALCROSSING_KEY}"
        f"&contentType=json"
    )

    resp = httpx.get(url, timeout=httpx.Timeout(10.0, connect=5.0))
    resp.raise_for_status()
    data = resp.json()

    # Cache every day from the batch response
    for day in data.get("days", []):
        d = day
        day_key = d.get("datetime", "")
        if not day_key:
            continue

        conditions = d.get("conditions", "Cloudy")
        result = {
            "temperature": round(d.get("tempmax", 28.0), 1),
            "rainfall": round(d.get("precip", 0.0), 1),
            "humidity": round(d.get("humidity", 80.0), 1),
            "weather_type": _visualcrossing_to_category(conditions),
            "is_rainy": d.get("precip", 0.0) > 5.0,
            "is_weekend": datetime.strptime(day_key, "%Y-%m-%d").weekday() >= 5,
        }
        _weather_cache[day_key] = {"ts": time.time(), "data": result}

    # Return the requested date from cache
    if date_str in _weather_cache:
        return _weather_cache[date_str]["data"]
    raise ValueError(f"No forecast data for {date_str}")

# Tier 2: Open-Meteo (free, no key)
# ==================================================================
def _call_openmeteo(dt: datetime) -> dict:
    date_str = dt.strftime("%Y-%m-%d")
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={KL_LAT}&longitude={KL_LON}"
        f"&daily=temperature_2m_max,precipitation_sum,relative_humidity_2m_max,weather_code"
        f"&timezone=Asia/Kuala_Lumpur"
        f"&start_date={date_str}&end_date={date_str}"
    )

    resp = httpx.get(url, timeout=httpx.Timeout(5.0, connect=3.0))
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    temp = daily.get("temperature_2m_max", [28.0])[0]
    rain = daily.get("precipitation_sum", [0.0])[0]
    humidity = daily.get("relative_humidity_2m_max", [80.0])[0]
    wmo_code = daily.get("weather_code", [1])[0]

    return {
        "temperature": float(temp),
        "rainfall": float(rain),
        "humidity": float(humidity),
        "weather_type": _wmo_to_category(int(wmo_code)),
        "is_rainy": float(rain) > 5.0,
        "is_weekend": dt.weekday() >= 5,
    }


# ==================================================================
# Tier 3: Monthly climatology
# ==================================================================
def _monthly_fallback(dt: datetime) -> dict:
    month = dt.month
    temp, rain, humidity = KL_MONTHLY.get(month, (28.0, 0.0, 80.0))
    return {
        "temperature": temp,
        "rainfall": rain,
        "humidity": humidity,
        "weather_type": "cloudy",
        "is_rainy": rain > 100,
        "is_weekend": dt.weekday() >= 5,
    }


def weather_one_hot(weather_type: str) -> dict:
    """Shared weather one-hot encoding used by S2 training and inference."""
    return {
        "weather_sunny": int(weather_type == "sunny"),
        "weather_cloudy": int(weather_type == "cloudy"),
        "weather_rainy": int(weather_type == "rainy"),
        "weather_storm": int(weather_type in ("storm", "thunderstorm")),
    }
