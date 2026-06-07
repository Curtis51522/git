# Toolbox - SHAP explainer and trend detector for S5 agents
# Provides tool-augmented reasoning capabilities for LLM synthesis.
import os, json, logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("s5.toolbox")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "xgboost")
PRODUCT_NAMES = ["croissant", "donut", "chiffon", "bread_roll", "bread_coconut", "croissant_chocolate"]

_shap_cache: Dict[str, Dict] = {}
_cache_ts: Dict[str, float] = {}
CACHE_TTL_SEC = 3600


def _load_model(product: str):
    """Load XGBoost model + feature columns. Returns (model, feature_cols) or None."""
    try:
        import xgboost as xgb
        model_path = os.path.join(MODEL_DIR, f"{product}_model.json")
        feat_path = os.path.join(MODEL_DIR, "feature_columns.json")
        if not os.path.exists(model_path) or not os.path.exists(feat_path):
            return None
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        with open(feat_path, "r") as f:
            feature_cols = json.load(f)
        return model, feature_cols
    except Exception as e:
        logger.warning("Failed to load XGBoost model for %s: %s", product, e)
        return None


def explain_forecast(product: str) -> Optional[Dict]:
    """Return SHAP feature contributions for a product's latest forecast."""
    import time
    now = time.time()
    if product in _shap_cache and (now - _cache_ts.get(product, 0)) < CACHE_TTL_SEC:
        return _shap_cache[product]

    try:
        loaded = _load_model(product)
        if not loaded:
            return None
        model, feature_cols = loaded
        raw = model.get_booster().get_score(importance_type="gain")
        total = sum(raw.values())
        if total == 0:
            return {"product": product, "top_features": [], "error": "No feature importance available"}
        top = sorted(raw.items(), key=lambda x: -x[1])[:5]
        features = []
        for k, v in top:
            fname = k  # feature names are already human-readable
            features.append({"feature": fname, "contribution": round(v / total, 3)})
        result = {"product": product, "top_features": features}
        _shap_cache[product] = result
        _cache_ts[product] = now
        return result
    except Exception as e:
        logger.warning("SHAP explain failed for %s: %s", e)
        return {"product": product, "top_features": [], "error": str(e)}


def detect_trend(product: str, metric: str = "waste", lookback_days: int = 14) -> Optional[Dict]:
    """Detect linear trend in a metric from snapshots."""
    try:
        from memory_store import _get_db
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT snapshot_date, data FROM s5_daily_snapshot "
            "WHERE snapshot_date >= DATE_SUB(CURRENT_DATE, INTERVAL %s DAY) "
            "ORDER BY snapshot_date ASC",
            (lookback_days,))
        rows = cur.fetchall()
        cur.close()
        if len(rows) < 3:
            return None
        dates = []
        values = []
        for r in rows:
            data = json.loads(r[1])
            val = data.get(metric, {}).get(product, 0)
            if val is not None:
                dates.append(r[0])
                values.append(float(val))
        if len(values) < 3:
            return None
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        avg = np.mean(y)
        if avg > 0:
            direction = "rising" if slope > avg * 0.05 else "declining" if slope < -avg * 0.05 else "stable"
        else:
            direction = "stable" if abs(slope) < 0.5 else ("rising" if slope > 0 else "declining")
        return {
            "product": product,
            "metric": metric,
            "direction": direction,
            "slope_per_day": round(float(slope), 2),
            "days_analyzed": len(values),
            "avg_value": round(float(avg), 1)
        }
    except Exception as e:
        logger.warning("detect_trend failed: %s", e)
        return None


def compare_products(product_a: str, product_b: str) -> Optional[Dict]:
    """Head-to-head comparison using latest snapshot."""
    try:
        from memory_store import _get_db
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT data FROM s5_daily_snapshot ORDER BY snapshot_date DESC LIMIT 1")
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        data = json.loads(row[0])
        result = {}
        for pname in [product_a, product_b]:
            result[pname] = {
                "inventory": data.get("inventory", {}).get(pname, 0),
                "forecast": data.get("forecast", {}).get(pname, 0),
                "waste": data.get("waste", {}).get(pname, 0),
            }
        return result
    except Exception as e:
        logger.warning("compare_products failed: %s", e)
        return None
