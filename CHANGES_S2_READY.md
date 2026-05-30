# S2 - Gap Analysis and Changes

## Requirements vs Current Implementation

| # | Requirement | Current | Status |
|---|------------|---------|:------:|
| 1 | D+1 to D+7 multi-horizon | Daily loop (functional equivalent) | OK |
| 2 | Freshness granularity | Loops FRESHNESS_STATES | OK |
| 3 | Multi-model comparison (LR/RF/XGBoost) | XGBoost only | MISSING |
| 4 | Lag features (lag_1, lag_7, rolling_7d) | None | MISSING |
| 5 | TimeSeriesSplit | Random data, no temporal split | MISSING |
| 6 | Empirical prediction intervals | Fixed 15% std_dev | PARTIAL |
| 7 | WAPE metric | MSE/MAE/R2 only | MISSING |
| 8 | Production/discount/waste recommendations | In S5 Fusion (correct architecture) | OK |
| 9 | SHAP explainability | In S5 causal_reasoning.py (correct architecture) | OK |
| 10 | Real weather + holidays + Ramadan | Implemented via weather.py | OK |

---

## Two-Perspective Priority Matrix

### System Integrity (what breaks if missing)

| Priority | Item | Real-data needed? |
|:--------:|------|:-----------------:|
| P0 | Lag features (lag_1, lag_7, rolling_7d) | Yes (cold-start returns 0) |
| P1 | Freshness as training feature | No |
| P2 | Empirical prediction intervals | No |

NOT needed for system: multi-model comparison, WAPE, TS split (synthetic data has no temporal signal).

### Thesis Defense (what fails defense if missing)

| Priority | Item | Why |
|:--------:|------|-----|
| P0 | Multi-model comparison (LR/RF/XGBoost) | "Why XGBoost?" without comparison = weak defense |
| P0 | TimeSeriesSplit | Random split on time series = data leakage = instant fail |
| P0 | Empirical prediction intervals | "Where does 15% come from?" has no answer |
| P1 | Error analysis by product/freshness/weather | Shows depth; "model struggles on rainy Day-2 items" = strong |
| P2 | Lag features + ablation | Shows temporal feature contribution |

---

## Changes

### 1. Multi-Model Comparison (Thesis P0)

Add to training script. Train LR, RF, XGBoost on same data, output comparison table.

New function in train_xgboost_full.py:

`
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def model_comparison(X_train, y_train, X_val, y_val, product_name):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                     random_state=42, n_jobs=-1),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results.append({
            "model": name,
            "MAE": round(mean_absolute_error(y_val, y_pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_val, y_pred)), 2),
            "R2": round(r2_score(y_val, y_pred), 4),
        })
    results.sort(key=lambda x: x["R2"], reverse=True)
    return results
`

Save to models/xgboost/{product}_comparison.json.

---

### 2. TimeSeriesSplit (Thesis P0)

Replace random split with time-aware split.

`
def split_by_date(df, val_days=90):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = len(df) - val_days
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]
    return train, val
`

Use TimeSeriesSplit for hyperparameter search:

`
tscv = TimeSeriesSplit(n_splits=5)
`

---

### 3. Empirical Prediction Intervals (Thesis P0 + System P1)

Replace fixed 15% std_dev with validation error distribution.

During training, save per-freshness/per-horizon error stats:

`
def save_error_stats(y_true, y_pred, df_val, product_name):
    errors = {}
    for freshness in FRESHNESS_STATES:
        for h in range(1, 8):
            mask = (df_val["freshness_status"] == freshness) & (df_val["horizon"] == h)
            if mask.sum() > 0:
                residuals = y_true[mask] - y_pred[mask]
                key = f"{freshness}:{h}"
                errors[key] = {
                    "std_dev": float(np.std(residuals)),
                    "p5": float(np.percentile(residuals, 5)),
                    "p95": float(np.percentile(residuals, 95)),
                    "n_samples": int(mask.sum()),
                }
    path = os.path.join(MODEL_DIR, f"{product_name}_errors.json")
    with open(path, "w") as f:
        json.dump(errors, f, indent=2)
`

In module2_forecast.py, load error stats for prediction intervals:

`
def _get_empirical_std(product, freshness, horizon):
    error_path = os.path.join(MODEL_DIR, f"{product}_errors.json")
    try:
        with open(error_path) as f:
            errors = json.load(f)
        key = f"{freshness}:{horizon}"
        return errors.get(key, {}).get("std_dev", 0.15)
    except (FileNotFoundError, KeyError):
        return 0.15  # cold start fallback
`

Replace in _do_forecast():

`
# Before:
std_dev = pred * 0.15

# After:
error_ratio = _get_empirical_std(product, freshness, d)
std_dev = pred * error_ratio if error_ratio > 0 else pred * 0.15
`

---

### 4. Lag Features (System P0 + Thesis P2)

Add to build_forecast_features(). Returns 0 on cold start.

`
def _get_lag_sales(product, date, lag_days):
    try:
        db = get_db()
        target = date - timedelta(days=lag_days)
        r = q(db, "inventory_transactions").select("quantity")
            .eq("product_name", product)
            .eq("transaction_type", "outflow")
            .gte("transaction_time", target.strftime("%Y-%m-%d"))
            .lt("transaction_time", (target + timedelta(days=1)).strftime("%Y-%m-%d"))
            .execute()
        if r.data:
            return sum(row["quantity"] for row in r.data)
        return 0
    except Exception:
        return 0

def _get_rolling_avg(product, date, window):
    try:
        db = get_db()
        end = date
        start = date - timedelta(days=window)
        r = q(db, "inventory_transactions").select("quantity")
            .eq("product_name", product)
            .eq("transaction_type", "outflow")
            .gte("transaction_time", start.strftime("%Y-%m-%d"))
            .lt("transaction_time", end.strftime("%Y-%m-%d"))
            .execute()
        if r.data:
            total = sum(row["quantity"] for row in r.data)
            return round(total / window, 1)
        return 0
    except Exception:
        return 0
`

Add to build_forecast_features() return dict:

`
"lag_1": _get_lag_sales(product, forecast_date, 1),
"lag_7": _get_lag_sales(product, forecast_date, 7),
"rolling_7d": _get_rolling_avg(product, forecast_date, 7),
`

Update FEATURE_COLS in training script:

`
FEATURE_COLS = [
    # ... existing ...
    "lag_1", "lag_7", "rolling_7d",
]
`

In generate_sales_data(), populate lag features from synthetic history:

`
row["lag_1"] = rows[-1]["predicted_demand"] if len(rows) > 0 else 0
row["lag_7"] = rows[-7]["predicted_demand"] if len(rows) >= 7 else 0
row["rolling_7d"] = np.mean([r["predicted_demand"] for r in rows[-7:]]) if len(rows) >= 7 else 0
`

---

### 5. Freshness as Training Feature (System P1)

Add one-hot freshness encodings to features:

`
FEATURE_COLS = [
    # ... existing ...
    "freshness_fresh", "freshness_day1", "freshness_day2", "freshness_discount",
]
`

In build_forecast_features():

`
"freshness_fresh": 1 if freshness == "Fresh" else 0,
"freshness_day1": 1 if freshness == "Day-1" else 0,
"freshness_day2": 1 if freshness == "Day-2" else 0,
"freshness_discount": 1 if freshness == "Discount" else 0,
`

In generate_sales_data(), generate rows per freshness:

`
for freshness in ["Fresh", "Day-1", "Day-2", "Discount"]:
    row = {
        # ... existing features ...
        "freshness_status": freshness,
        "freshness_fresh": 1 if freshness == "Fresh" else 0,
        "freshness_day1": 1 if freshness == "Day-1" else 0,
        "freshness_day2": 1 if freshness == "Day-2" else 0,
        "freshness_discount": 1 if freshness == "Discount" else 0,
        "predicted_demand": base_demand * FRESHNESS_UPLIFT[freshness] * random.uniform(0.7, 1.3),
    }
    rows.append(row)
`

---

### 6. Error Analysis by Segment (Thesis P1)

Per-segment error breakdown for thesis.

`
def error_analysis(y_true, y_pred, df_val, product_name):
    analysis = {"by_freshness": {}, "by_weekday": {}, "by_weather": {}}
    df = df_val.copy()
    df["error"] = y_true - y_pred
    df["abs_error"] = np.abs(df["error"])
    
    for freshness in df["freshness_status"].unique():
        mask = df["freshness_status"] == freshness
        analysis["by_freshness"][freshness] = {
            "MAE": round(df.loc[mask, "abs_error"].mean(), 2),
            "RMSE": round(np.sqrt((df.loc[mask, "error"]**2).mean()), 2),
            "n_samples": int(mask.sum()),
        }
    for dow in range(1, 7):  # Skip Monday
        mask = df["day_of_week"] == dow
        if mask.sum() > 0:
            analysis["by_weekday"][dow] = {
                "MAE": round(df.loc[mask, "abs_error"].mean(), 2),
                "n_samples": int(mask.sum()),
            }
    for weather in ["sunny", "cloudy", "rainy", "storm"]:
        col = f"weather_{weather}"
        if col in df.columns:
            mask = df[col] == 1
            if mask.sum() > 0:
                analysis["by_weather"][weather] = {
                    "MAE": round(df.loc[mask, "abs_error"].mean(), 2),
                    "n_samples": int(mask.sum()),
                }
    path = os.path.join(MODEL_DIR, f"{product_name}_error_analysis.json")
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    return analysis
`

---

## Implementation Order

1. train_xgboost_full.py - TimeSeriesSplit
2. train_xgboost_full.py - Freshness as feature dimension
3. train_xgboost_full.py - Lag features in data generation
4. train_xgboost_full.py - Multi-model comparison
5. train_xgboost_full.py - Error statistics + error analysis
6. api/module2_forecast.py - Lag feature slots
7. api/module2_forecast.py - Empirical prediction intervals
8. api/module2_forecast.py - Freshness one-hot features

## Notes

- All changes work with synthetic data AND real data
- Lag features return 0 on cold start, auto-populate when real data arrives
- Multi-model comparison runs during training only, zero runtime cost
- Prediction intervals gracefully fallback to 15% when stats unavailable
- Monday-closed logic preserved
﻿
---

# S2 - Code Review: Engineering Quality

## Pipeline Order: CORRECT

Training: Generate -> Clean -> Validate -> Split -> Tune -> Evaluate -> Save
Inference: Check cache -> Load model -> Build features -> Predict -> Return

Both are correct and complete.

## Issues Found

### train_xgboost_full.py (Training)

| # | Issue | Severity | Fix |
|---|-------|:--------:|------|
| 1 | warnings.filterwarnings("ignore") hides ALL warnings including convergence failures | Medium | Change to specific: warnings.filterwarnings("ignore", category=FutureWarning) |
| 2 | split_data uses string date comparison instead of datetime | Low | Wrap in pd.to_datetime before comparison |
| 3 | clean_data clips outliers to boundaries causing value pile-up | Low | Use np.where to NaN then drop, or Winsorize SciPy |
| 4 | Weather one-hot logic duplicated between training and inference | Medium | Extract to shared module (see fix below) |
| 5 | confidence field purely heuristic (d=0=high, d<=5=medium) | Low | Replace with empirical from error stats (already in gap section) |

### api/module2_forecast.py (Inference)

| # | Issue | Severity | Fix |
|---|-------|:--------:|------|
| 1 | Feature columns defined separately from training - drift risk | HIGH | Single FEATURE_COLS in config/settings.py shared by both |
| 2 | Zero logging - API calls and failures are silent | HIGH | Add logging module with structured output |
| 3 | Forecast cache has no LRU eviction - unbounded growth | Medium | Add max size + LRU via collections.OrderedDict or functools.lru_cache |
| 4 | Cache write has no thread lock - race condition on concurrent requests | Medium | Add threading.Lock around cache mutation |
| 5 | Fixed 15% std_dev prediction intervals (already addressed in gap section) | Medium | See empirical prediction intervals section |

## Fixes

### Fix 1: Shared Feature Columns (HIGH - both files)

Move FEATURE_COLS to config/settings.py:

`
# config/settings.py
FORECAST_FEATURE_COLS = [
    ""day_of_week"", ""is_weekend"", ""day_of_month"", ""month"",
    ""discount_rate"", ""is_public_holiday"", ""is_ramadan"",
    ""temperature"", ""rainfall"", ""humidity"", ""is_rainy"",
    ""weather_sunny"", ""weather_cloudy"", ""weather_rainy"", ""weather_storm"",
]
`

In train_xgboost_full.py:

`
from config.settings import FORECAST_FEATURE_COLS as FEATURE_COLS
`

In module2_forecast.py, validate feature dict against the config:

`
from config.settings import FORECAST_FEATURE_COLS

def build_forecast_features(forecast_date, freshness, product=""):
    features = { ... }
    # Validate all expected columns present
    for col in FORECAST_FEATURE_COLS:
        if col not in features:
            features[col] = 0  # safe default
    return features
`

### Fix 2: Add Logging (HIGH - module2_forecast.py)

`
import logging
logger = logging.getLogger("s2.forecast")

def load_product_model(product_name):
    logger.info("Loading model for %s", product_name)
    ...

def _do_forecast(product, days, ...):
    logger.info("Forecast request: product=%s, days=%d", product or "all", days)
    ...
    logger.info("Forecast complete: %d products, %d forecasts",
                 len(products_to_forecast), len(forecasts))
`

### Fix 3: Warnings Filter (MEDIUM - train_xgboost_full.py)

`
# Before:
warnings.filterwarnings("ignore")

# After:
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
`

### Fix 4: Cache Thread Safety + LRU (MEDIUM - module2_forecast.py)

`
import threading
from collections import OrderedDict

_forecast_cache: OrderedDict = OrderedDict()
_MAX_CACHE_SIZE = 100
_cache_lock = threading.Lock()

def _cache_get(key):
    with _cache_lock:
        if key in _forecast_cache:
            entry = _forecast_cache.pop(key)  # Move to end (LRU)
            if time.time() - entry["ts"] < _FORECAST_CACHE_TTL:
                _forecast_cache[key] = entry
                return dict(entry["data"])
    return None

def _cache_set(key, data):
    with _cache_lock:
        _forecast_cache[key] = {"ts": time.time(), "data": data}
        if len(_forecast_cache) > _MAX_CACHE_SIZE:
            _forecast_cache.popitem(last=False)  # Evict oldest
`

### Fix 5: Weather Feature Sharing (MEDIUM - both files)

Extract weather one-hot to shared location:

`
# api/weather.py - add:
def weather_one_hot(weather_type):
    return {
        ""weather_sunny"": int(weather_type == ""sunny""),
        ""weather_cloudy"": int(weather_type == ""cloudy""),
        ""weather_rainy"": int(weather_type == ""rainy""),
        ""weather_storm"": int(weather_type in (""storm"", ""thunderstorm"")),
    }
`

Both train_xgboost_full.py and module2_forecast.py import from api.weather.

---

## Updated S2 Implementation Order

1. config/settings.py - Add FORECAST_FEATURE_COLS
2. api/weather.py - Add weather_one_hot()
3. train_xgboost_full.py - Import shared FEATURE_COLS + weather_one_hot + fix warnings
4. train_xgboost_full.py - TimeSeriesSplit + Freshness features + Lag features
5. train_xgboost_full.py - Multi-model comparison + Error analysis
6. api/module2_forecast.py - Import shared FEATURE_COLS + weather_one_hot
7. api/module2_forecast.py - Add logging
8. api/module2_forecast.py - Cache thread safety + LRU
9. api/module2_forecast.py - Lag feature slots + Empirical intervals + Freshness features
