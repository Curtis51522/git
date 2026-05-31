import os, sys, asyncio, time
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import OrderedDict
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import APIRouter, Query
from datetime import datetime, timedelta, date
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PRODUCT_TYPES, FRESHNESS_STATES
from api.weather import get_weather
import holidays
from hijri_converter import convert
from db.mysql_client import get_db, q
from models.schemas import SalesForecast

logger = logging.getLogger("s2.forecast")

_my_holidays = holidays.MY()

def _is_ramadan_date(dt):
    """Check if a Gregorian date falls within Ramadan (Hijri month 9)."""
    try:
        h = convert.Gregorian(dt.year, dt.month, dt.day).to_hijri()
        return h.month == 9
    except Exception:
        return False

router = APIRouter(prefix="/s2", tags=["Module 2 - Sales Forecast"])

MODEL_DIR = "models/xgboost"

_model_cache: Dict[str, xgb.XGBRegressor] = {}
_executor = ThreadPoolExecutor(max_workers=2)

# Forecast cache (keyed by "product:days", TTL 1 hour)
_forecast_cache: OrderedDict = OrderedDict()
_MAX_CACHE_SIZE = 100
_cache_lock = threading.Lock()
_FORECAST_CACHE_TTL = 3600

def _cache_get(key: str):
    with _cache_lock:
        if key in _forecast_cache:
            entry = _forecast_cache.pop(key)
            if time.time() - entry["ts"] < _FORECAST_CACHE_TTL:
                _forecast_cache[key] = entry
                return dict(entry["data"])
    return None

def _cache_set(key: str, data: dict):
    with _cache_lock:
        _forecast_cache[key] = {"ts": time.time(), "data": data}
        if len(_forecast_cache) > _MAX_CACHE_SIZE:
            _forecast_cache.popitem(last=False)

def _model_path(product_name: str) -> str:
    safe = product_name.replace(" ", "_").lower()
    return os.path.join(MODEL_DIR, f"{safe}_model.json")

def load_product_model(product_name: str) -> xgb.XGBRegressor:
    if product_name in _model_cache:
        return _model_cache[product_name]
    path = _model_path(product_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model for '{product_name}' at {path}.")
    model = xgb.XGBRegressor()
    model.load_model(path)
    _model_cache[product_name] = model
    logger.info("Loaded model for %s from %s", product_name, path)
    return model

def get_available_products() -> list:
    return [p for p in PRODUCT_TYPES if os.path.exists(_model_path(p))]


# --- Lag feature helpers (DB-backed) ---
_lag_cache = {}
_lag_cache_ts = {}

def _get_product_daily_sales(product_name: str) -> dict:
    """Query DB for daily sales totals per product. Cached for 5 minutes."""
    global _lag_cache, _lag_cache_ts
    now = time.time()
    if product_name in _lag_cache and product_name in _lag_cache_ts and (now - _lag_cache_ts[product_name]) < 300:
        return _lag_cache[product_name]
    try:
        db = get_db()
        c = db.cursor(dictionary=True)
        c.execute(
            "SELECT DATE(transaction_time) as dt, SUM(quantity) as qty "
            "FROM inventory_transactions "
            "WHERE transaction_type='outflow' AND product_name=%s "
            "GROUP BY DATE(transaction_time) ORDER BY dt",
            (product_name,)
        )
        sales = {row['dt'].strftime('%Y-%m-%d') if hasattr(row['dt'], 'strftime') else str(row['dt']): row['qty'] for row in c.fetchall()}
    except Exception as e:
        logger.warning("Lag features DB query failed for %s: %s", product_name, e)
        sales = {}
    _lag_cache[product_name] = sales
    _lag_cache_ts[product_name] = now
    # Cleanup stale entries (older than 10 min)
    stale = [k for k, ts in list(_lag_cache_ts.items()) if now - ts > 600]
    for k in stale:
        _lag_cache.pop(k, None)
        _lag_cache_ts.pop(k, None)
    return sales

def _get_lag(product_name: str, forecast_date, days_back: int) -> float:
    """Get sales from 'days_back' days before forecast_date, skipping closed days."""
    if not product_name:
        return 0.0
    sales = _get_product_daily_sales(product_name)
    if not sales:
        return 0.0
    from datetime import timedelta
    fd = forecast_date if hasattr(forecast_date, 'date') else forecast_date
    target = fd - timedelta(days=days_back)
    # Try the exact date first, then back up to find the nearest day with data
    for _ in range(4):
        key = target.strftime('%Y-%m-%d')
        if key in sales:
            return float(sales[key])
        target -= timedelta(days=1)
    return 0.0

def _get_rolling_7d_mean(product_name: str, forecast_date) -> float:
    """Average daily sales over the 7 days before forecast_date."""
    if not product_name:
        return 0.0
    sales = _get_product_daily_sales(product_name)
    if not sales:
        return 0.0
    from datetime import timedelta
    fd = forecast_date if hasattr(forecast_date, 'date') else forecast_date
    values = []
    for d in range(1, 8):
        target = fd - timedelta(days=d)
        key = target.strftime('%Y-%m-%d')
        if key in sales:
            values.append(sales[key])
    if not values:
        return 0.0
    return float(sum(values) / len(values))

def build_forecast_features(forecast_date: datetime, freshness: str = "", product: str = "") -> dict:
    weather = get_weather(forecast_date)
    dow = forecast_date.weekday()
    wt = weather.get("weather_type", "cloudy")
    dt_date = forecast_date.date() if hasattr(forecast_date, 'date') else date(forecast_date.year, forecast_date.month, forecast_date.day)
    from config.settings import FORECAST_FEATURE_COLS

    features_dict = {
        "day_of_week": dow,
        "is_weekend": 1 if dow >= 5 else 0,
        "day_of_month": forecast_date.day,
        "month": forecast_date.month,

        "is_public_holiday": 1 if dt_date in _my_holidays else 0,
        "is_ramadan": 1 if _is_ramadan_date(dt_date) else 0,
        "temperature": weather.get("temperature", 28.0),
        "rainfall": weather.get("rainfall", 0.0),
        "humidity": weather.get("humidity", 80.0),
        "is_rainy": 1 if weather.get("is_rainy") else 0,
        "weather_sunny": 1 if wt == "sunny" else 0,
        "weather_cloudy": 1 if wt == "cloudy" else 0,
        "weather_rainy": 1 if wt == "rainy" else 0,
        "weather_storm": 1 if wt in ("storm", "thunderstorm") else 0,
        "lag_1": _get_lag(product, forecast_date, 1),
        "lag_7": _get_lag(product, forecast_date, 7),
        "rolling_7d_mean": _get_rolling_7d_mean(product, forecast_date),
    }
    # Ensure all expected columns present
    for col in FORECAST_FEATURE_COLS:
        if col not in features_dict:
            features_dict[col] = 0
    return features_dict

def _do_forecast(product: Optional[str], days: int, use_cache: bool = True, start_date: Optional[str] = None) -> dict:
    logger.info("Forecast request: product=%s, days=%d, start=%s", product or "all", days, start_date or "today")
    # --- cache check ---
    cache_key = f"{product or 'all'}:{days}:{start_date or 'today'}"
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            cached["cached"] = True
            return cached
    # --- end cache check ---

    products_to_forecast = [product] if product else get_available_products()
    if not products_to_forecast:
        return {"status": "no_models", "message": "No trained models found."}

    if start_date:
        try:
            today = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            today = datetime.now()
    else:
        today = datetime.now()
    forecasts = []
    model_errors = []

    for prod in products_to_forecast:
        try:
            model = load_product_model(prod)
        except FileNotFoundError as e:
            model_errors.append(str(e))
            continue
        for d in range(0, days):
            forecast_date = today + timedelta(days=d)
            # Monday = shop closed, output zero-demand entry
            if forecast_date.weekday() == 0:
                forecasts.append(SalesForecast(
                    forecast_date=forecast_date.strftime("%Y-%m-%d"),
                    product_name=prod,
                    freshness_status="Total",
                    predicted_demand=0,
                    confidence="closed",
                ))
                continue
            features = build_forecast_features(forecast_date, "", prod)
            X = pd.DataFrame([features]).fillna(0)
            try:
                pred = float(model.predict(X)[0])
                pred = max(0.0, pred)
            except Exception as e:
                logger.warning("Prediction failed for %s on %s: %s", prod, forecast_date.strftime("%%Y-%%m-%%d"), e)
                pred = 0.0

            forecasts.append(SalesForecast(
                forecast_date=forecast_date.strftime("%Y-%m-%d"),
                product_name=prod,
                freshness_status="Total",
                predicted_demand=round(pred),
                
                confidence="today" if d == 0 else ("high" if d <= 2 else ("medium" if d <= 5 else "low")),
            ))

    response = {
        "status": "ok",
        "products_forecasted": len(products_to_forecast) - len(model_errors),
        "forecasts": [f.model_dump() for f in forecasts],
        "cached": False,
    }
    if model_errors:
        response["model_errors"] = model_errors

    # --- cache store ---
    _cache_set(cache_key, response)
    logger.info("Forecast complete: %d products, %d forecasts", len(products_to_forecast), len(forecasts))
    return response

@router.get("/forecast")
async def get_forecast(
    product: Optional[str] = Query(None, description="Product name or empty for all"),
    days: int = Query(7, ge=1, le=7),
    date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _do_forecast, product, days, True, date)

@router.get("/forecast/refresh")
async def refresh_forecast(
    product: Optional[str] = Query(None),
    days: int = Query(7, ge=1, le=7),
):
    """Force-refresh forecast, bypassing cache."""

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _do_forecast, product, days, False)

@router.get("/sales_history")
async def get_sales_history(days: int = Query(30, ge=1, le=90)):
    db = get_db()
    c = db.cursor(dictionary=True)
    c.execute(
        "SELECT * FROM inventory_transactions WHERE transaction_type=%s ORDER BY transaction_time DESC LIMIT 200",
        ("outflow",)
    )
    rows = c.fetchall()
    for row in rows:
        for k, v in row.items():
            if hasattr(v, 'isoformat'):
                row[k] = v.isoformat()
    return {"status": "ok", "count": len(rows), "transactions": rows}


@router.get("/accuracy")
async def get_accuracy():
    """Return per-product test MAE for prediction intervals."""
    path = os.path.join(MODEL_DIR, "test_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return {"status": "ok", "metrics": json.load(f)}
    return {"status": "no_data", "message": "test_metrics.json not found"}
