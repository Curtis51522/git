import os, sys, asyncio, time, math
import json
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import OrderedDict
import pandas as pd
import xgboost as xgb
from fastapi import APIRouter, Depends, Query
from datetime import datetime, timedelta
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BEVERAGE_PRODUCT_TYPES, PRODUCT_TYPES, FORECAST_FEATURE_COLS
from db.mysql_client import get_db, q
from api.auth import require_manager
from api.operation_clock import operation_now
from models.schemas import SalesForecast
from s2_forecasting.feature_contract import (
    FEATURE_GROUPS,
    FEATURE_METADATA,
    FORECAST_FEATURES,
    RESERVED_SCENARIO_FEATURES,
)
from s2_forecasting.quantile_utils import enforce_quantile_monotonicity

logger = logging.getLogger("s2.forecast")

router = APIRouter(prefix="/s2", tags=["Module 2 - Sales Forecast"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "s2_forecasting", "outputs")

BUSINESS_EVENT_TYPES = {
    "new_product_launch": {
        "label": "New product launch",
        "reserved_feature": "is_new_product",
    },
    "competitor_activity": {
        "label": "Competitor activity",
        "reserved_feature": "is_competitor",
    },
}

# Frozen training data metadata used when live DB has insufficient recent sales data.
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
    calendar_day = _weather_data[
        (_weather_data.index.month == dt_date.month)
        & (_weather_data.index.day == dt_date.day)
    ]
    if len(calendar_day) > 0:
        temp_mean = float(calendar_day["temp_mean"].mean())
        temp_range = float(
            (calendar_day["temp_max"] - calendar_day["temp_min"]).mean()
        )
        is_cold = 1 if temp_mean < 15 else 0
        is_hot = 1 if temp_mean > 25 else 0
        return temp_mean, temp_range, is_cold, is_hot

    # Last fallback: use the broader monthly climatology.
    m = dt_date.month
    month_data = _weather_data[_weather_data.index.month == m]
    if len(month_data) > 0:
        temp_mean = float(month_data["temp_mean"].mean())
        temp_range = float((month_data["temp_max"] - month_data["temp_min"]).mean())
        is_cold = 1 if temp_mean < 15 else 0
        is_hot = 1 if temp_mean > 25 else 0
        return temp_mean, temp_range, is_cold, is_hot
    return 20.0, 6.0, 0, 0


def _get_is_rainy(dt_date):
    _init_weather()
    if _weather_data is None or "precipitation" not in _weather_data.columns:
        return 0
    selected = pd.Timestamp(dt_date.strftime("%Y-%m-%d"))
    if selected in _weather_data.index:
        return int(float(_weather_data.loc[selected]["precipitation"]) >= 1.0)

    calendar_day = _weather_data[
        (_weather_data.index.month == dt_date.month)
        & (_weather_data.index.day == dt_date.day)
    ]
    if len(calendar_day) > 0:
        rainy_share = float((calendar_day["precipitation"] >= 1.0).mean())
        return int(rainy_share >= 0.5)

    month_data = _weather_data[_weather_data.index.month == dt_date.month]
    if len(month_data) > 0:
        rainy_share = float((month_data["precipitation"] >= 1.0).mean())
        return int(rainy_share >= 0.5)
    return 0

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


def _get_category_id(product_name: str) -> int:
    return int(product_name in BEVERAGE_PRODUCT_TYPES)


def _compute_bias_factor(
    actual_total: float,
    predicted_total: float,
    completed_days: int,
) -> float:
    if completed_days < 3 or predicted_total <= 0:
        return 1.0
    return min(max(actual_total / predicted_total, 0.75), 1.5)

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


def _build_interval_context(prediction: float, lower_bound: int, upper_bound: int) -> dict:
    interval_width = max(upper_bound - lower_bound, 0)
    rounded_prediction = max(float(round(prediction)), 1.0)
    relative_width = round(interval_width / rounded_prediction, 3)
    if interval_width <= 1 or relative_width <= 0.5 or (rounded_prediction <= 3 and interval_width <= 2):
        uncertainty_level = "low"
    else:
        uncertainty_level = "high"
    return {
        "q50": round(float(prediction), 3),
        "interval_width": interval_width,
        "relative_width": relative_width,
        "uncertainty_level": uncertainty_level,
        "interval_method": "Conformal 80%",
    }


def get_available_products() -> list:
    path = os.path.join(MODEL_DIR, "quantile_model_q50.pkl")
    if os.path.exists(path):
        return list(_get_product_id_map().keys())
    return []


def _parse_products(value):
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _format_date_value(value):
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _serialize_business_event(row):
    event_type = row.get("event_type", "")
    config = BUSINESS_EVENT_TYPES.get(event_type, {})
    return {
        "id": row.get("id"),
        "event_type": event_type,
        "label": config.get("label", event_type),
        "start_date": _format_date_value(row.get("start_date")),
        "end_date": _format_date_value(row.get("end_date")),
        "products": _parse_products(row.get("products")),
        "discount_pct": row.get("discount_pct"),
        "note": row.get("note") or "",
        "active": bool(row.get("active")),
    }


def _build_reserved_scenario_summary(events):
    summary = {
        name: {
            **meta,
            "active": False,
            "value": 0,
            "model_input": False,
        }
        for name, meta in RESERVED_SCENARIO_FEATURES.items()
    }
    for event in events:
        if not event.get("active", True):
            continue
        event_config = BUSINESS_EVENT_TYPES.get(event.get("event_type"))
        if not event_config:
            continue
        feature = event_config["reserved_feature"]
        if feature not in summary:
            continue
        summary[feature]["active"] = True
        summary[feature]["value"] = 1
        summary[feature]["events"] = [
            *summary[feature].get("events", []),
            event,
        ]
    return summary


def _list_business_events(date: str = ""):
    selected_date = date or operation_now(datetime.now).strftime("%Y-%m-%d")
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT id, event_type, start_date, end_date, products, discount_pct, note, active
        FROM business_events
        WHERE active = 1 AND start_date <= %s AND end_date >= %s
        ORDER BY start_date ASC, id ASC
        """,
        (selected_date, selected_date),
    )
    return [_serialize_business_event(row) for row in cursor.fetchall()]


def _validate_business_event_payload(payload):
    event_type = payload.get("event_type")
    if event_type not in BUSINESS_EVENT_TYPES:
        raise ValueError("Unsupported business event type")
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    if not start_date or not end_date:
        raise ValueError("start_date and end_date are required")
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("start_date and end_date must use YYYY-MM-DD") from exc
    if end_dt < start_dt:
        raise ValueError("end_date must be on or after start_date")
    products = payload.get("products") or []
    if not isinstance(products, list):
        raise ValueError("products must be a list")
    discount_pct = payload.get("discount_pct")
    if discount_pct in ("", None):
        discount_pct = None
    else:
        discount_pct = float(discount_pct)
        if discount_pct < 0 or discount_pct > 100:
            raise ValueError("discount_pct must be between 0 and 100")
    return {
        "event_type": event_type,
        "start_date": start_date,
        "end_date": end_date,
        "products": json.dumps(products),
        "discount_pct": discount_pct,
        "note": payload.get("note") or "",
        "active": 1 if payload.get("active", True) else 0,
    }


# --- Lag feature helpers (DB-backed) ---
_lag_cache = {}
_lag_cache_ts = {}

def _get_product_daily_sales(product_name: str) -> dict:
    """Query DB for daily sales totals per product. Cached for 5 minutes."""
    global _lag_cache, _lag_cache_ts
    now = time.time()
    if product_name in _lag_cache and product_name in _lag_cache_ts and (now - _lag_cache_ts[product_name]) < 300:
        return _lag_cache[product_name]
    db = None
    c = None
    try:
        db = get_db()
        c = db.cursor(dictionary=True)
        c.execute(
            "SELECT DATE(transaction_time) as dt, SUM(quantity) as qty "
            "FROM inventory_transactions "
            "WHERE transaction_type='outflow' AND receipt_id IS NOT NULL "
            "AND product_name=%s "
            "GROUP BY DATE(transaction_time) ORDER BY dt",
            (product_name,)
        )
        sales = {row['dt'].strftime('%Y-%m-%d') if hasattr(row['dt'], 'strftime') else str(row['dt']): row['qty'] for row in c.fetchall()}
    except Exception as e:
        logger.warning("Lag features DB query failed for %s: %s", product_name, e)
        sales = {}
    finally:
        if c is not None:
            c.close()
        if db is not None:
            db.close()
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
    return _get_lag_from_history(
        _get_product_daily_sales(product_name),
        forecast_date,
        days_back,
    )


def _history_rows_before(sales_history: dict, forecast_date):
    fd = forecast_date.date() if hasattr(forecast_date, "date") else forecast_date
    rows = []
    for key, value in sales_history.items():
        try:
            row_date = datetime.strptime(str(key), "%Y-%m-%d").date()
        except ValueError:
            continue
        if row_date < fd:
            rows.append((row_date, float(value)))
    return sorted(rows, key=lambda item: item[0])


def _get_lag_from_history(sales_history: dict, forecast_date, days_back: int) -> float:
    if not sales_history or days_back < 1:
        return 0.0
    fd = forecast_date.date() if hasattr(forecast_date, "date") else forecast_date
    target = fd - timedelta(days=days_back)
    candidates = [
        (row_date, value)
        for row_date, value in _history_rows_before(sales_history, forecast_date)
        if row_date <= target
    ]
    return candidates[-1][1] if candidates else 0.0

def _get_rolling_avg(product_name: str, forecast_date, window: int) -> float:
    if not product_name or window < 1:
        return 0.0
    return _get_rolling_avg_from_history(
        _get_product_daily_sales(product_name),
        forecast_date,
        window,
    )


def _get_rolling_avg_from_history(
    sales_history: dict,
    forecast_date,
    window: int,
) -> float:
    if not sales_history or window < 1:
        return 0.0
    rows = _history_rows_before(sales_history, forecast_date)
    values = [value for _, value in rows[-window:]]
    return float(np.mean(values)) if values else 0.0

def _get_daily_tickets(forecast_date) -> float:
    db = None
    c = None
    ticket_count = 0.0
    try:
        db = get_db()
        c = db.cursor(dictionary=True)
        fd = forecast_date if hasattr(forecast_date, "date") else forecast_date
        c.execute(
            "SELECT COUNT(DISTINCT receipt_id) as cnt "
            "FROM inventory_transactions "
            "WHERE transaction_type='outflow' AND receipt_id IS NOT NULL "
            "AND DATE(transaction_time) = ("
            "  SELECT MAX(DATE(transaction_time)) FROM inventory_transactions "
            "  WHERE transaction_type='outflow' AND receipt_id IS NOT NULL "
            "  AND DATE(transaction_time) < DATE(%s)"
            ")",
            (fd.strftime("%Y-%m-%d"),)
        )
        row = c.fetchone()
        if row and row["cnt"]:
            ticket_count = float(row["cnt"])
    except Exception:
        pass
    finally:
        if c is not None:
            c.close()
        if db is not None:
            db.close()
    return ticket_count

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

def _get_is_day1(product_name: str, forecast_date) -> int:
    db = None
    cursor = None
    try:
        db = get_db()
        cursor = db.cursor()
        date_str = (
            forecast_date.strftime("%Y-%m-%d")
            if hasattr(forecast_date, "strftime")
            else str(forecast_date)
        )
        cursor.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM batch_inventory bi
                JOIN products p ON p.product_name = bi.product_name
                WHERE bi.product_name = %s
                  AND p.category = 'bakery'
                  AND COALESCE(bi.quantity_remaining, bi.quantity) > 0
                  AND DATE(bi.production_time) = DATE(%s) - INTERVAL 1 DAY
            )
            """,
            (product_name, date_str),
        )
        row = cursor.fetchone()
        if row and row[0] and row[0] > 0:
            return 1
    except Exception:
        pass
    finally:
        if cursor is not None:
            cursor.close()
        if db is not None:
            db.close()
    return 0



# Feature order matching the deployed S2 Q50 forecast model.
FORECAST_FEATURE_ORDER = FORECAST_FEATURE_COLS

def build_forecast_features(
    forecast_date: datetime,
    product: str = "",
    sales_history: dict | None = None,
    daily_tickets: float | None = None,
) -> dict:
    _init_frozen_meta()
    dow = forecast_date.weekday()
    dt_date = forecast_date.date() if hasattr(forecast_date, "date") else datetime(forecast_date.year, forecast_date.month, forecast_date.day).date()
    m = forecast_date.month

    # Product ID and category (0=bread, 1=beverage)
    pid_map = _get_product_id_map()
    pid = pid_map.get(product, -1)
    category = _get_category_id(product)

    # Frozen fallback values: (lag_1, lag_7_avg, lag_30_avg, roll_std_7, roll_std_14, trend_7)
    frozen = _frozen_meta.get("last_lag", {}).get(pid, (0, 0, 0, 0, 0, 0))

    # Lag features: live DB first, fall back to frozen
    history = sales_history
    lag_1 = (
        _get_lag_from_history(history, forecast_date, 1)
        if history is not None
        else _get_lag(product, forecast_date, 1)
    )
    if lag_1 == 0 and len(frozen) >= 1:
        lag_1 = frozen[0]
    lag_7 = (
        _get_rolling_avg_from_history(history, forecast_date, 7)
        if history is not None
        else _get_rolling_avg(product, forecast_date, 7)
    )
    if lag_7 == 0 and len(frozen) >= 2:
        lag_7 = frozen[1]
    lag_30 = (
        _get_rolling_avg_from_history(history, forecast_date, 30)
        if history is not None
        else _get_rolling_avg(product, forecast_date, 30)
    )
    if lag_30 == 0 and len(frozen) >= 3:
        lag_30 = frozen[2]

    # roll_std_7/14 and trend_7: frozen fallback only (no live compute needed)
    roll_std_7 = frozen[3] if len(frozen) >= 4 else 0.0
    roll_std_14 = frozen[4] if len(frozen) >= 5 else 0.0
    trend_7 = lag_1 - lag_7 if (lag_1 > 0 or lag_7 > 0) else (frozen[5] if len(frozen) >= 6 else 0.0)

    # daily_tickets: live DB estimate, fallback to frozen
    ticket_count = (
        float(daily_tickets)
        if daily_tickets is not None
        else _get_daily_tickets(forecast_date)
    )
    if ticket_count == 0:
        ticket_count = _frozen_meta.get("last_daily_tickets", 0)

    # Promo / event features (mostly 0 for forecast)
    is_day1 = _get_is_day1(product, dt_date)
    is_top3 = 1 if pid in _frozen_meta.get("top3_products", []) else 0
    discount_pct = 0.0
    is_member_day = 0

    is_rainy = _get_is_rainy(dt_date)

    # Weather features (always available from historical data)
    temp_mean, temp_range, is_cold_day, is_hot_day = _get_weather(dt_date)

    # Beverage aggregate features (frozen fallback from training data)
    large_ratio = 0.25
    cold_ratio = 0.27
    sweetness_avg = 1.8
    ice_avg = 1.6
    temp_hot_ratio = 0.15 + 0.70 / (1 + np.exp((temp_mean - 22) / 4)) if temp_mean else 0.5

    features = {
        "product_id": pid,
        "category": category,
        "daily_tickets": ticket_count,
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
    return {name: features.get(name, 0) for name in FORECAST_FEATURE_ORDER}


def _get_recent_category_bias_factors(
    start_date: datetime,
    products: list[str],
    quantile_models: dict,
    lookback_days: int = 7,
) -> dict[str, float]:
    totals = {
        "bakery": {"actual": 0.0, "predicted": 0.0, "days": 0},
        "beverage": {"actual": 0.0, "predicted": 0.0, "days": 0},
    }
    histories = {
        product: {
            str(key): float(value)
            for key, value in _get_product_daily_sales(product).items()
        }
        for product in products
    }

    for offset in range(lookback_days, 0, -1):
        backcast_date = start_date - timedelta(days=offset)
        date_key = backcast_date.strftime("%Y-%m-%d")
        ticket_count = _get_daily_tickets(backcast_date)
        feature_rows = []
        categories = []
        day_actual = {"bakery": 0.0, "beverage": 0.0}

        for product in products:
            category = "beverage" if _get_category_id(product) else "bakery"
            history = {
                key: value
                for key, value in histories[product].items()
                if datetime.strptime(key, "%Y-%m-%d").date()
                < backcast_date.date()
            }
            feature_rows.append(
                build_forecast_features(
                    backcast_date,
                    product,
                    sales_history=history,
                    daily_tickets=ticket_count,
                )
            )
            categories.append(category)
            day_actual[category] += histories[product].get(date_key, 0.0)

        if not feature_rows:
            continue
        X = pd.DataFrame(feature_rows)[FORECAST_FEATURE_ORDER].fillna(0).values
        raw_quantiles = {
            name: np.maximum(model.predict(X), 0)
            for name, model in quantile_models.items()
        }
        corrected = enforce_quantile_monotonicity(
            raw_quantiles["q10"],
            raw_quantiles["q50"],
            raw_quantiles["q90"],
        )
        day_predicted = {"bakery": 0.0, "beverage": 0.0}
        for category, prediction in zip(categories, corrected["q50"]):
            day_predicted[category] += float(prediction)

        for category in totals:
            if day_actual[category] <= 0:
                continue
            totals[category]["actual"] += day_actual[category]
            totals[category]["predicted"] += day_predicted[category]
            totals[category]["days"] += 1

    return {
        category: _compute_bias_factor(
            values["actual"],
            values["predicted"],
            values["days"],
        )
        for category, values in totals.items()
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
            today = operation_now(datetime.now)
    else:
        today = operation_now(datetime.now)
    forecasts = []
    model_errors = []

    quantile_models = {
        "q10": _get_unified_quantile("q10"),
        "q50": _get_unified_quantile("q50"),
        "q90": _get_unified_quantile("q90"),
    }
    pid_map = _get_product_id_map()
    forecast_ticket_count = _get_daily_tickets(today)
    bias_factors = _get_recent_category_bias_factors(
        today,
        list(pid_map),
        quantile_models,
    )
    unit_prices = {}
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT product_name, unit_price FROM products")
        unit_prices = {str(row[0]): float(row[1] or 0.0) for row in cur.fetchall()}
    except Exception as e:
        logger.warning("Product price lookup failed for forecast response: %s", e)

    for prod in products_to_forecast:
        if prod not in pid_map:
            model_errors.append(f"Product {prod} not in product_id_map")
            continue
        half_width = _get_conformal_half(prod)
        category = "beverage" if _get_category_id(prod) else "bakery"
        bias_factor = bias_factors[category]
        cutoff = today.date()
        sales_history = {
            str(key): float(value)
            for key, value in _get_product_daily_sales(prod).items()
            if datetime.strptime(str(key), "%Y-%m-%d").date() < cutoff
        }
        for d in range(0, days):
            forecast_date = today + timedelta(days=d)
            features = build_forecast_features(
                forecast_date,
                prod,
                sales_history=sales_history,
                daily_tickets=forecast_ticket_count,
            )
            X = pd.DataFrame([features])[FORECAST_FEATURE_ORDER].fillna(0).values
            try:
                raw_quantiles = {
                    name: np.maximum(model.predict(X), 0) * bias_factor
                    for name, model in quantile_models.items()
                }
                corrected = enforce_quantile_monotonicity(
                    raw_quantiles["q10"],
                    raw_quantiles["q50"],
                    raw_quantiles["q90"],
                )
                pred = float(corrected["q50"][0])
                adjusted_half_width = half_width * bias_factor
                lower_bound = max(0, round(pred - adjusted_half_width))
                upper_bound = max(lower_bound, round(pred + adjusted_half_width))
                interval_context = _build_interval_context(pred, lower_bound, upper_bound)
            except Exception as e:
                err_msg = f"Prediction failed for {prod} on {forecast_date.strftime('%Y-%m-%d')}: {e}"
                logger.warning(err_msg)
                model_errors.append(err_msg)
                pred = 0.0
                lower_bound = 0
                upper_bound = 0
                interval_context = _build_interval_context(pred, lower_bound, upper_bound)

            sales_history[forecast_date.strftime("%Y-%m-%d")] = pred
            forecasts.append(SalesForecast(
                forecast_date=forecast_date.strftime("%Y-%m-%d"),
                product_name=prod,
                freshness_status="Total",
                predicted_demand=round(pred),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                **interval_context,
                confidence="Conformal 80%",
                unit_price=unit_prices.get(prod, 0.0),
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

@router.get("/forecast", dependencies=[Depends(require_manager)])
async def get_forecast(
    product: Optional[str] = Query(None, description="Product name or empty for all"),
    days: int = Query(7, ge=1, le=7),
    date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _do_forecast, product, days, True, date)

@router.get("/forecast/refresh", dependencies=[Depends(require_manager)])
async def refresh_forecast(
    product: Optional[str] = Query(None),
    days: int = Query(7, ge=1, le=7),
    date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
):
    """Force-refresh forecast, bypassing cache."""

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _do_forecast, product, days, False, date)

@router.get("/sales_history", dependencies=[Depends(require_manager)])
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


@router.get("/accuracy", dependencies=[Depends(require_manager)])
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

@router.get("/features/importance", dependencies=[Depends(require_manager)])
async def get_feature_importance():
    """Return deployed S2 Q50 feature importance scores for S2-S5 analysis."""
    try:
        model = _get_unified_quantile("q50")
        importances = model.feature_importances_
        features = FORECAST_FEATURES
        if len(importances) != len(features):
            return {"status": "mismatch", "message": f"Model has {len(importances)} features, expected {len(features)}"}

        pairs = sorted(
            [
                {
                    "feature": f,
                    "importance": round(float(v), 5),
                    "group": FEATURE_METADATA[f]["group"],
                    "availability": FEATURE_METADATA[f]["availability"],
                }
                for f, v in zip(features, importances)
            ],
            key=lambda x: x["importance"], reverse=True
        )
        grouped = {}
        for group, cols in FEATURE_GROUPS.items():
            grouped[group] = [p for p in pairs if p["feature"] in cols]

        return {
            "status": "ok",
            "model": "quantile_model_q50",
            "total_features": len(pairs),
            "ranked": pairs,
            "grouped": grouped,
        }
    except Exception as e:
        logger.warning("Feature importance failed: %s", e)
        return {"status": "error", "message": str(e)}


@router.get("/business-events", dependencies=[Depends(require_manager)])
async def get_business_events(date: str = ""):
    selected_date = date or operation_now(datetime.now).strftime("%Y-%m-%d")
    try:
        events = _list_business_events(selected_date)
        return {
            "status": "ok",
            "date": selected_date,
            "events": events,
            "reserved_scenario_features": _build_reserved_scenario_summary(events),
        }
    except Exception as e:
        logger.warning("Business events query failed: %s", e)
        return {"status": "error", "message": str(e)}


@router.post("/business-events", dependencies=[Depends(require_manager)])
async def create_business_event(payload: dict):
    try:
        data = _validate_business_event_payload(payload)
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            """
            INSERT INTO business_events
            (event_type, start_date, end_date, products, discount_pct, note, active)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                data["event_type"],
                data["start_date"],
                data["end_date"],
                data["products"],
                data["discount_pct"],
                data["note"],
                data["active"],
            ),
        )
        return {"status": "ok", "id": cursor.lastrowid}
    except Exception as e:
        logger.warning("Business event create failed: %s", e)
        return {"status": "error", "message": str(e)}


@router.put(
    "/business-events/{event_id}",
    dependencies=[Depends(require_manager)],
)
async def update_business_event(event_id: int, payload: dict):
    try:
        data = _validate_business_event_payload(payload)
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            """
            UPDATE business_events
            SET event_type = %s, start_date = %s, end_date = %s,
                products = %s, discount_pct = %s, note = %s, active = %s
            WHERE id = %s
            """,
            (
                data["event_type"],
                data["start_date"],
                data["end_date"],
                data["products"],
                data["discount_pct"],
                data["note"],
                data["active"],
                event_id,
            ),
        )
        return {"status": "ok", "id": event_id}
    except Exception as e:
        logger.warning("Business event update failed: %s", e)
        return {"status": "error", "message": str(e)}


@router.delete(
    "/business-events/{event_id}",
    dependencies=[Depends(require_manager)],
)
async def delete_business_event(event_id: int):
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("UPDATE business_events SET active = 0 WHERE id = %s", (event_id,))
        return {"status": "ok", "id": event_id}
    except Exception as e:
        logger.warning("Business event delete failed: %s", e)
        return {"status": "error", "message": str(e)}


@router.get("/features/today", dependencies=[Depends(require_manager)])
async def get_today_features(date: str = ""):
    """Return today's actual feature values for S5 cross-referencing."""
    import datetime as _dt
    try:
        if date:
            forecast_date = _dt.datetime.strptime(date, "%Y-%m-%d")
        else:
            forecast_date = operation_now(_dt.datetime.now)

        feats = build_forecast_features(forecast_date, "")
        readable = {k: feats.get(k, 0) for k in FORECAST_FEATURE_ORDER}
        weather_context = {
            "temp_mean": feats.get("temp_mean", 0),
            "temp_range": feats.get("temp_range", 0),
            "is_cold_day": feats.get("is_cold_day", 0),
            "is_hot_day": feats.get("is_hot_day", 0),
        }
        interpretations = {
            "is_weekend": "weekend" if readable.get("is_weekend") else "weekday",
            "is_holiday": "holiday" if readable.get("is_holiday") else "non-holiday",
            "is_rainy": "rainy" if readable.get("is_rainy") else "dry",
            "is_member_day": "member_day" if readable.get("is_member_day") else "non_member_day",
            "is_day1": "day1_promo" if readable.get("is_day1") else "non_promo",
            "is_top3": "top3_product" if readable.get("is_top3") else "non_top3",
        }
        if weather_context.get("is_cold_day"):
            interpretations["weather"] = "cold_day"
        elif weather_context.get("is_hot_day"):
            interpretations["weather"] = "hot_day"
        else:
            interpretations["weather"] = "mild"
        active_events = []
        selected_date = forecast_date.strftime("%Y-%m-%d")
        try:
            active_events = _list_business_events(selected_date)
        except Exception as event_error:
            logger.warning("Business event context unavailable for %s: %s", selected_date, event_error)
        reserved_scenario_features = _build_reserved_scenario_summary(active_events)
        return {
            "status": "ok",
            "date": selected_date,
            "features": readable,
            "feature_contract": {
                "source": "s2_forecasting.feature_contract",
                "total_features": len(FORECAST_FEATURE_ORDER),
                "groups": FEATURE_GROUPS,
            },
            "business_events": active_events,
            "reserved_scenario_features": reserved_scenario_features,
            "weather_context": weather_context,
            "interpretations": {k: v for k, v in interpretations.items()},
        }
    except Exception as e:
        logger.warning("Today features failed: %s", e)
        return {"status": "error", "message": str(e)}
