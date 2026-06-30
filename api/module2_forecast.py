import os, sys, asyncio, time, math
import json
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import OrderedDict
import pandas as pd
import xgboost as xgb
from fastapi import APIRouter, Query
from datetime import datetime, timedelta
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PRODUCT_TYPES, FORECAST_FEATURE_COLS
from db.mysql_client import get_db, q
from models.schemas import SalesForecast

logger = logging.getLogger("s2.forecast")

router = APIRouter(prefix="/s2", tags=["Module 2 - Sales Forecast"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "s2_forecasting", "outputs")

# Frozen training data metadata (lag features + weather monthly averages)
# Used when live DB has insufficient recent sales data.
_frozen_meta = None

def _init_frozen_meta():
    global _frozen_meta
    if _frozen_meta is not None:
        return
    train_path = os.path.join(ROOT, "data", "xgboost_train.csv")
    if not os.path.exists(train_path):
        logger.warning("xgboost_train.csv not found at %s", train_path)
        _frozen_meta = {}
        return
    train = pd.read_csv(train_path)
    train["date"] = pd.to_datetime(train["date"])
    # Use June monthly averages for frozen meta (seasonal match)
    jun_train = train[train["date"].dt.month == 6]
    if len(jun_train) == 0:
        jun_train = train  # fallback
    jun_avg = jun_train.groupby("product_id").agg(
        lag_1=("lag_1", "mean"),
        lag_7_avg=("lag_7_avg", "mean"),
        lag_30_avg=("lag_30_avg", "mean"),
        roll_std_7=("roll_std_7", "mean"),
        roll_std_14=("roll_std_14", "mean"),
        trend_7=("trend_7", "mean"),
    )
    _frozen_meta = {
        "last_lag": {pid: (float(row.lag_1), float(row.lag_7_avg), float(row.lag_30_avg),
                        float(row.roll_std_7), float(row.roll_std_14), float(row.trend_7))
                    for pid, row in jun_avg.iterrows()},
        "last_daily_tickets": float(jun_train.groupby("date")["daily_tickets"].first().mean()),
        "holiday_dates": sorted(train[train["is_holiday"]==1]["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "top3_products": train.groupby("product_id")["quantity"].mean().nlargest(3).index.tolist(),
        "rainy_dates": set(train[train["is_rainy"]==1]["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
    }
    logger.info("Frozen meta loaded: %d products, top3=%s", len(_frozen_meta["last_lag"]), _frozen_meta["top3_products"])

_weather_data = None

def _init_weather():
    global _weather_data
    if _weather_data is not None:
        return
    weather_path = os.path.join(ROOT, "data", "guangzhou_weather.csv")
    if os.path.exists(weather_path):
        w = pd.read_csv(weather_path)
        w["date"] = pd.to_datetime(w["date"])
        _weather_data = w.set_index("date")
        logger.info("Weather data loaded: %d days", len(_weather_data))
    else:
        logger.warning("guangzhou_weather.csv not found, weather features will be default")

def _get_weather(dt_date):
    _init_weather()
    if _weather_data is None:
        return 20.0, 6.0, 0, 0  # defaults: mean=20, range=6, cold=0, hot=0
    ds = dt_date.strftime("%Y-%m-%d")
    try:
        d = pd.Timestamp(ds)
        if d in _weather_data.index:
            row = _weather_data.loc[d]
            temp_mean = float(row["temp_mean"])
            temp_range = float(row["temp_max"] - row["temp_min"])
            is_cold = 1 if temp_mean < 15 else 0
            is_hot = 1 if temp_mean > 25 else 0
            return temp_mean, temp_range, is_cold, is_hot
    except Exception:
        pass
    # Fallback: use monthly average from historical data
    m = dt_date.month
    month_data = _weather_data[_weather_data.index.month == m]
    if len(month_data) > 0:
        temp_mean = float(month_data["temp_mean"].mean())
        temp_range = float((month_data["temp_max"] - month_data["temp_min"]).mean())
        is_cold = 1 if temp_mean < 15 else 0
        is_hot = 1 if temp_mean > 25 else 0
        return temp_mean, temp_range, is_cold, is_hot
    return 20.0, 6.0, 0, 0

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

# Unified model (single XGBoost model for all 45 products with product_id feature)
_unified_model = None
_unified_quantile = {}
_product_id_map = None
_product_bounds = None

def _get_unified_model():
    global _unified_model
    if _unified_model is None:
        path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Unified model not found at {path}")
        import joblib
        _unified_model = joblib.load(path)
        logger.info("Loaded unified model from %s", path)
    return _unified_model

def _get_unified_quantile(q: str):
    global _unified_quantile
    if q not in _unified_quantile:
        path = os.path.join(MODEL_DIR, f"quantile_model_{q}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Quantile model {q} not found at {path}")
        import joblib as _jl
        model = _jl.load(path)
        _unified_quantile[q] = model
        logger.info("Loaded quantile %s from %s", q, path)
    return _unified_quantile[q]

def _get_product_id_map():
    global _product_id_map
    if _product_id_map is None:
        path = os.path.join(MODEL_DIR, "product_id_map.json")
        if os.path.exists(path):
            with open(path) as f:
                _product_id_map = json.load(f)
        else:
            _product_id_map = {p: i for i, p in enumerate(sorted(PRODUCT_TYPES))}
    return _product_id_map

_conformal_calibration = None

def _get_conformal_half(product_name: str) -> float:
    global _conformal_calibration
    if _conformal_calibration is None:
        cal_path = os.path.join(MODEL_DIR, "conformal_calibration.json")
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                _conformal_calibration = json.load(f)
            logger.info("Conformal calibration loaded")
        else:
            _conformal_calibration = {"per_product": {}, "global": 1.0}
    pid_map = _get_product_id_map()
    pid = pid_map.get(product_name, -1)
    return _conformal_calibration.get("per_product", {}).get(str(pid), _conformal_calibration.get("global", 1.0))

def _get_product_bounds():
    global _product_bounds
    if _product_bounds is None:
        bounds_path = os.path.join(MODEL_DIR, "product_bounds.json")
        if os.path.exists(bounds_path):
            with open(bounds_path) as f:
                _product_bounds = json.load(f)
            logger.info("Loaded product bounds from %s", bounds_path)
        else:
            _product_bounds = {}
            logger.warning("product_bounds.json not found")
    return _product_bounds

def get_available_products() -> list:
    path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    if os.path.exists(path):
        return list(_get_product_id_map().keys())
    return []

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

def _get_rolling_avg(product_name: str, forecast_date, window: int) -> float:
    if not product_name or window < 1:
        return 0.0
    sales = _get_product_daily_sales(product_name)
    if not sales:
        return 0.0
    fd = forecast_date if hasattr(forecast_date, "date") else forecast_date
    vals = []
    for d in range(1, window + 1):
        target = fd - timedelta(days=d)
        for _ in range(4):
            key = target.strftime("%Y-%m-%d")
            if key in sales:
                vals.append(float(sales[key]))
                break
            target -= timedelta(days=1)
    return float(np.mean(vals)) if vals else 0.0

def _get_daily_tickets(forecast_date) -> float:
    try:
        db = get_db()
        c = db.cursor(dictionary=True)
        fd = forecast_date if hasattr(forecast_date, "date") else forecast_date
        c.execute(
            "SELECT COUNT(DISTINCT receipt_id) as cnt "
            "FROM inventory_transactions "
            "WHERE transaction_type='outflow' "
            "AND DATE(transaction_time) = ("
            "  SELECT MAX(DATE(transaction_time)) FROM inventory_transactions "
            "  WHERE transaction_type='outflow' AND DATE(transaction_time) < DATE(%s)"
            ")",
            (fd.strftime("%Y-%m-%d"),)
        )
        row = c.fetchone()
        if row and row["cnt"]:
            return float(row["cnt"])
    except Exception:
        pass
    return 0.0

def _get_is_holiday(dt_date) -> int:
    try:
        from chinese_calendar import is_holiday as chinese_holiday
        # Pure public holiday on weekday only (not weekend)
        is_hol = chinese_holiday(dt_date)
        is_weekend = dt_date.weekday() >= 5
        return int(is_hol and not is_weekend)
    except Exception:
        ds = dt_date.strftime("%Y-%m-%d")
        return 1 if ds in _frozen_meta.get("holiday_dates", []) else 0

def _get_is_day1(product_name: str) -> int:
    try:
        db = get_db()
        c = db.cursor()
        c.execute(
            "SELECT stock_day1 FROM products WHERE product_name=%s",
            (product_name,)
        )
        row = c.fetchone()
        if row and row[0] and row[0] > 0:
            return 1
    except Exception:
        pass
    return 0



# Feature order matching train_quantile.py (17 features)
FORECAST_FEATURE_ORDER = [
    "product_id", "category", "daily_tickets", "day_of_week", "month",
    "is_weekend", "is_holiday",
    "lag_1", "lag_7_avg", "lag_30_avg", "roll_std_7", "roll_std_14", "trend_7",
    "is_day1", "is_top3", "discount_pct",
    "is_member_day", "is_rainy",
    "temp_mean", "temp_range", "is_cold_day", "is_hot_day",
    "large_ratio", "cold_ratio", "sweetness_avg", "ice_avg", "temp_hot_ratio",
]

def build_forecast_features(forecast_date: datetime, product: str = "") -> dict:
    _init_frozen_meta()
    dow = forecast_date.weekday()
    dt_date = forecast_date.date() if hasattr(forecast_date, "date") else datetime(forecast_date.year, forecast_date.month, forecast_date.day).date()
    m = forecast_date.month

    # Product ID and category (0=bread, 1=beverage)
    pid_map = _get_product_id_map()
    pid = pid_map.get(product, -1)
    category = 1 if pid >= 30 else 0

    # Frozen fallback values: (lag_1, lag_7_avg, lag_30_avg, roll_std_7, roll_std_14, trend_7)
    frozen = _frozen_meta.get("last_lag", {}).get(pid, (0, 0, 0, 0, 0, 0))

    # Lag features: live DB first, fall back to frozen
    lag_1 = _get_lag(product, forecast_date, 1)
    if lag_1 == 0 and len(frozen) >= 1:
        lag_1 = frozen[0]
    lag_7 = _get_rolling_avg(product, forecast_date, 7)
    if lag_7 == 0 and len(frozen) >= 2:
        lag_7 = frozen[1]
    lag_30 = _get_rolling_avg(product, forecast_date, 30)
    if lag_30 == 0 and len(frozen) >= 3:
        lag_30 = frozen[2]

    # roll_std_7/14 and trend_7: frozen fallback only (no live compute needed)
    roll_std_7 = frozen[3] if len(frozen) >= 4 else 0.0
    roll_std_14 = frozen[4] if len(frozen) >= 5 else 0.0
    trend_7 = lag_1 - lag_7 if (lag_1 > 0 or lag_7 > 0) else (frozen[5] if len(frozen) >= 6 else 0.0)

    # daily_tickets: live DB estimate, fallback to frozen
    daily_tickets = _get_daily_tickets(forecast_date)
    if daily_tickets == 0:
        daily_tickets = _frozen_meta.get("last_daily_tickets", 0)

    # Promo / event features (mostly 0 for forecast)
    is_day1 = _get_is_day1(product)
    is_top3 = 1 if pid in _frozen_meta.get("top3_products", []) else 0
    discount_pct = 0.0
    is_member_day = 0
    is_new_product = 0
    is_competitor = 0

    # is_rainy: check weather data from frozen training mapping
    ds = dt_date.strftime("%Y-%m-%d")
    is_rainy = 1 if ds in _frozen_meta.get("rainy_dates", set()) else 0

    # Weather features (always available from historical data)
    temp_mean, temp_range, is_cold_day, is_hot_day = _get_weather(dt_date)

    # Beverage aggregate features (frozen fallback from training data)
    large_ratio = 0.25
    cold_ratio = 0.27
    sweetness_avg = 1.8
    ice_avg = 1.6
    temp_hot_ratio = 0.15 + 0.70 / (1 + np.exp((temp_mean - 22) / 4)) if temp_mean else 0.5

    return {
        "product_id": pid,
        "category": category,
        "daily_tickets": daily_tickets,
        "day_of_week": dow,
        "month": m,
        "is_weekend": 1 if dow >= 5 else 0,
        "is_holiday": _get_is_holiday(dt_date),
        "lag_1": lag_1,
        "lag_7_avg": lag_7,
        "lag_30_avg": lag_30,
        "roll_std_7": roll_std_7,
        "roll_std_14": roll_std_14,
        "trend_7": trend_7,
        "is_day1": is_day1,
        "is_top3": is_top3,
        "discount_pct": discount_pct,
        "is_member_day": is_member_day,
        "is_rainy": is_rainy,
        "temp_mean": temp_mean,
        "temp_range": temp_range,
        "is_cold_day": is_cold_day,
        "is_hot_day": is_hot_day,
        "large_ratio": large_ratio,
        "cold_ratio": cold_ratio,
        "sweetness_avg": sweetness_avg,
        "ice_avg": ice_avg,
        "temp_hot_ratio": temp_hot_ratio,
    }


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

    q50_model = _get_unified_quantile("q50")
    pid_map = _get_product_id_map()

    for prod in products_to_forecast:
        if prod not in pid_map:
            model_errors.append(f"Product {prod} not in product_id_map")
            continue
        half_width = _get_conformal_half(prod)
        pid = pid_map[prod]
        for d in range(0, days):
            forecast_date = today + timedelta(days=d)
            features = build_forecast_features(forecast_date, prod)
            X = pd.DataFrame([features])[FORECAST_FEATURE_ORDER].fillna(0).values
            try:
                q50_pred = float(q50_model.predict(X)[0])
                pred = max(0.0, q50_pred)
                lower_bound = max(0, round(pred - half_width))
                upper_bound = max(lower_bound, round(pred + half_width))
            except Exception as e:
                err_msg = f"Prediction failed for {prod} on {forecast_date.strftime('%Y-%m-%d')}: {e}"
                logger.warning(err_msg)
                model_errors.append(err_msg)
                pred = 0.0
                lower_bound = 0
                upper_bound = 0

            forecasts.append(SalesForecast(
                forecast_date=forecast_date.strftime("%Y-%m-%d"),
                product_name=prod,
                freshness_status="Total",
                predicted_demand=round(pred),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence="Conformal 80%",
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
            metrics = json.load(f)
        # Sanitize non-JSON-compliant float values (inf, -inf, nan -> null)
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, float) and not math.isfinite(obj):
                return None
            return obj
        return {"status": "ok", "metrics": _sanitize(metrics)}
    return {"status": "no_data", "message": "test_metrics.json not found"}



# Training feature list matching train_xgboost.py (16 features)
_TRAINING_FEATURES = [
    "product_id", "daily_tickets", "day_of_week", "month",
    "is_weekend", "is_holiday",
    "lag_1", "lag_7_avg", "lag_30_avg",
    "is_day1", "is_top3", "discount_pct",
    "is_member_day", "is_new_product", "is_competitor", "is_rainy",
]

@router.get("/features/importance")
async def get_feature_importance():
    """Return XGBoost feature importance scores for S2-S5 cross-analysis."""
    try:
        model = _get_unified_model()
        importances = model.feature_importances_
        features = _TRAINING_FEATURES
        if len(importances) != len(features):
            return {"status": "mismatch", "message": f"Model has {len(importances)} features, expected {len(features)}"}

        # Pair feature name with importance, sorted by importance desc
        pairs = sorted(
            [{"feature": f, "importance": round(float(v), 5)} for f, v in zip(features, importances)],
            key=lambda x: x["importance"], reverse=True
        )
        # Group by category for easier consumption
        categories = {
            "temporal": ["day_of_week", "month", "is_weekend", "is_holiday"],
            "lag": ["lag_1", "lag_7_avg", "lag_30_avg"],
            "product": ["product_id", "is_day1", "is_top3", "discount_pct"],
            "event": ["is_member_day", "is_new_product", "is_competitor", "is_rainy"],
            "demand": ["daily_tickets"],
        }
        grouped = {}
        for cat, cols in categories.items():
            grouped[cat] = [p for p in pairs if p["feature"] in cols]

        return {
            "status": "ok",
            "total_features": len(pairs),
            "ranked": pairs,
            "grouped": grouped,
        }
    except Exception as e:
        logger.warning("Feature importance failed: %s", e)
        return {"status": "error", "message": str(e)}


@router.get("/features/today")
async def get_today_features(date: str = ""):
    """Return today's actual feature values for S5 cross-referencing."""
    import datetime as _dt
    try:
        if date:
            forecast_date = _dt.datetime.strptime(date, "%Y-%m-%d")
        else:
            forecast_date = _dt.datetime.now()

        # Build full features then extract training subset + weather context
        feats = build_forecast_features(forecast_date, "")
        readable = {k: feats.get(k, 0) for k in _TRAINING_FEATURES}
        # Weather features not in training model but provide context
        weather_context = {
            "temp_mean": feats.get("temp_mean", 0),
            "temp_range": feats.get("temp_range", 0),
            "is_cold_day": feats.get("is_cold_day", 0),
            "is_hot_day": feats.get("is_hot_day", 0),
        }
        # Add human-readable interpretations (training features + weather)
        interpretations = {
            "is_weekend": "weekend" if readable.get("is_weekend") else "weekday",
            "is_holiday": "holiday" if readable.get("is_holiday") else "non-holiday",
            "is_rainy": "rainy" if readable.get("is_rainy") else "dry",
            "is_member_day": "member_day" if readable.get("is_member_day") else "non_member_day",
            "is_day1": "day1_promo" if readable.get("is_day1") else "non_promo",
            "is_top3": "top3_product" if readable.get("is_top3") else "non_top3",
            "is_new_product": "new_product" if readable.get("is_new_product") else "existing_product",
            "is_competitor": "competitor_active" if readable.get("is_competitor") else "no_competitor_event",
        }
        if weather_context.get("is_cold_day"):
            interpretations["weather"] = "cold_day"
        elif weather_context.get("is_hot_day"):
            interpretations["weather"] = "hot_day"
        else:
            interpretations["weather"] = "mild"
        return {
            "status": "ok",
            "date": forecast_date.strftime("%Y-%m-%d"),
            "features": readable,
            "weather_context": weather_context,
            "interpretations": {k: v for k, v in interpretations.items()},
        }
    except Exception as e:
        logger.warning("Today features failed: %s", e)
        return {"status": "error", "message": str(e)}
