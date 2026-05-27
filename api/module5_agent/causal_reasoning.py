"""
Causal Reasoning Engine for S5 Multi-Agent Bakery System.

Provides per-prediction SHAP-based attribution to explain WHY a forecast
differs from baseline, connecting XGBoost feature contributions to
external factors (weather, holidays, Ramadan, promotions).

Architecture:
  XGBoost model -> SHAP TreeExplainer -> per-feature SHAP values
  -> ranked attributions -> structured causal chain -> LLM report

Reference: Lundberg & Lee (2017) "A Unified Approach to Interpreting
Model Predictions" (NeurIPS). SHAP values satisfy Shapley axioms for
fair credit allocation among features.
"""
import os, json, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ---------------------------------------------------------------------------
# Feature -> Human Label mapping with causal direction
# ---------------------------------------------------------------------------
FEATURE_CAUSAL_LABELS: Dict[str, Dict[str, str]] = {
    "weather_sunny":       {"label": "Sunny weather",       "positive": "boosts walk-in traffic",    "negative": "absence reduces demand"},
    "weather_cloudy":      {"label": "Cloudy weather",      "positive": "slightly boosts demand",    "negative": "absence reduces baseline"},
    "weather_rainy":       {"label": "Rainy weather",       "positive": "dry weather lifts demand",  "negative": "rain suppresses footfall"},
    "weather_storm":       {"label": "Storm weather",       "positive": "calm weather helps",        "negative": "storm severely cuts traffic"},
    "is_weekend":          {"label": "Weekend effect",      "positive": "leisure demand surge",      "negative": "weekday baseline"},
    "is_rainy":            {"label": "Rain indicator",      "positive": "dry conditions help",       "negative": "rain suppresses footfall"},
    "rainfall":            {"label": "Rainfall amount",     "positive": "low rain helps demand",     "negative": "heavy rain reduces sales"},
    "temperature":         {"label": "Temperature",         "positive": "warmer day boosts demand",  "negative": "cooler day moderates demand"},
    "humidity":            {"label": "Humidity",            "positive": "comfortable levels help",   "negative": "high humidity reduces appetite"},
    "day_of_week":         {"label": "Day of week",         "positive": "peak day pattern",          "negative": "off-peak day pattern"},
    "day_of_month":        {"label": "Day of month",        "positive": "payday proximity boost",    "negative": "mid-month lull"},
    "month":               {"label": "Seasonal month",      "positive": "peak season effect",        "negative": "off-season effect"},
    "discount_rate":       {"label": "Discount/promo",      "positive": "discounted pricing lifts demand", "negative": "Fresh product at full price (no discount applied)"},
    "is_public_holiday":   {"label": "Public holiday",      "positive": "holiday demand surge",      "negative": "regular working day"},
    "is_ramadan":          {"label": "Ramadan season",      "positive": "Ramadan boosts iftar demand","negative": "Ramadan suppresses daytime demand"},
}


# ---------------------------------------------------------------------------
# Explainer cache
# ---------------------------------------------------------------------------
_explainer_cache: dict = {}

def _load_explainer(product_name: str):
    """Load XGBoost model + SHAP TreeExplainer (cached)."""
    import xgboost as xgb
    import shap

    if product_name in _explainer_cache:
        return _explainer_cache[product_name]

    _proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(_proj, "models", "xgboost", f"{product_name}_model.json")

    if not os.path.exists(model_path):
        return None

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    explainer = shap.TreeExplainer(model)
    _explainer_cache[product_name] = (model, explainer)
    return (model, explainer)


def _build_feature_vector(forecast_date: datetime, freshness: str = "Fresh",
                          product: str = "") -> pd.DataFrame:
    """Build the same feature vector S2 uses for prediction."""
    # Reuse S2"s feature builder
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from api.module2_forecast import build_forecast_features
    features = build_forecast_features(forecast_date, freshness, product)
    X = pd.DataFrame([features])
    # Ensure column order matches training
    _proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    feat_path = os.path.join(_proj, "models", "xgboost", "feature_columns.json")
    if os.path.exists(feat_path):
        with open(feat_path, "r") as f:
            cols = json.load(f)
        for c in cols:
            if c not in X.columns:
                X[c] = 0
        X = X[cols]
    return X.fillna(0)


def _load_feature_order() -> List[str]:
    _proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    feat_path = os.path.join(_proj, "models", "xgboost", "feature_columns.json")
    if os.path.exists(feat_path):
        with open(feat_path, "r") as f:
            return json.load(f)
    return []


def compute_shap_attribution(
    product_name: str,
    forecast_date_str: str,
    freshness: str = "Fresh",
) -> Dict:
    """
    Compute SHAP attribution for a single forecast.

    Returns:
        {
            "product": str,
            "date": str,
            "predicted_demand": float,
            "base_value": float (expected model output),
            "shap_contributions": [
                {"feature": str, "label": str, "shap_value": float,
                 "direction": "+" or "-", "explanation": str},
                ... (sorted by abs SHAP, top 5)
            ],
            "top_driver": str,
            "external_context": {...},
        }
    """
    result = {
        "product": product_name,
        "date": forecast_date_str,
        "predicted_demand": 0.0,
        "base_value": 0.0,
        "shap_contributions": [],
        "top_driver": "",
        "external_context": {},
        "error": None,
    }

    loaded = _load_explainer(product_name)
    if loaded is None:
        result["error"] = f"No model found for {product_name}"
        return result

    model, explainer = loaded

    try:
        dt = datetime.strptime(forecast_date_str, "%Y-%m-%d")
    except ValueError:
        result["error"] = f"Invalid date: {forecast_date_str}"
        return result

    X = _build_feature_vector(dt, freshness, product_name)
    feature_order = _load_feature_order()

    # Predict
    pred = float(model.predict(X)[0])
    result["predicted_demand"] = round(pred, 1)

    # SHAP values
    shap_values = explainer.shap_values(X)
    base_value = float(explainer.expected_value)
    result["base_value"] = round(base_value, 1)

    if shap_values.ndim == 1:
        shap_vals = shap_values
    else:
        shap_vals = shap_values[0]

    # Build ranked contributions with contextual filtering
    contributions = []
    feature_values = {feat: float(X.iloc[0][feat]) for feat in feature_order if feat in X.columns}
    
    # Features that should only be highlighted when their VALUE changes from baseline
    # (i.e., don't cite "no discount" for Fresh products ? that's the default)
    BASELINE_ZERO_FEATURES = {
        "discount_rate": "Fresh products don't have discounts ? this is the normal baseline, not a demand driver",
        "is_public_holiday": "Not a public holiday ? normal baseline",
        "is_ramadan": "Not Ramadan ? normal baseline",
        "weather_storm": "No storm ? normal baseline",
    }
    
    for i, (feat, sv) in enumerate(zip(feature_order, shap_vals)):
        if abs(sv) < 0.01:
            continue
        
        # Skip features that are at baseline (value=0) when they'd mislead
        feat_val = feature_values.get(feat, -1)
        if feat in BASELINE_ZERO_FEATURES and feat_val == 0:
            continue
        
        info = FEATURE_CAUSAL_LABELS.get(feat, {"label": feat, "positive": "increases demand", "negative": "decreases demand"})
        label = info["label"]
        # SHAP sign already encodes direction: + raises forecast, - lowers it
        effective_sign = "+" if sv > 0 else "-"
        explanation = info["positive"] if effective_sign == "+" else info["negative"]
        contributions.append({
            "feature": feat,
            "label": label,
            "shap_value": round(float(sv), 2),
            "effective_sign": effective_sign,
            "explanation": explanation,
            "abs_impact": abs(float(sv)),
        })

    contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
    result["shap_contributions"] = contributions[:5]
    result["top_driver"] = contributions[0]["label"] if contributions else "no clear driver"

    # External context
    result["external_context"] = _fetch_external_context(dt)

    return result


def _fetch_external_context(dt: datetime) -> Dict:
    """Fetch external factors that may influence demand."""
    from api.module2_forecast import _my_holidays, _is_ramadan_date
    dt_date = dt.date()

    ctx = {}
    ctx["day_of_week"] = dt.strftime("%A")
    ctx["is_weekend"] = dt.weekday() >= 5
    ctx["is_ramadan"] = _is_ramadan_date(dt_date)
    ctx["is_public_holiday"] = dt_date in _my_holidays

    if ctx["is_public_holiday"]:
        ctx["holiday_name"] = _my_holidays.get(dt_date, "Unknown")
    if ctx["is_ramadan"]:
        ctx["ramadan_note"] = "Ramadan period — altered consumption patterns (pre-dawn/iftar demand)"

    return ctx


def generate_causal_chain(attribution: Dict) -> List[str]:
    """
    Convert SHAP attribution into structured causal chain strings.

    Example output:
    [
        "Weekend effect (+3.2): leisure demand surge drives higher forecast",
        "Sunny weather (+1.8): boosts walk-in traffic",
        "No promotion (-0.5): absence of discount moderates demand",
    ]
    """
    if attribution.get("error"):
        return [f"Cannot generate causal chain: {attribution['error']}"]

    base = attribution["base_value"]
    pred = attribution["predicted_demand"]
    delta = pred - base
    delta_dir = "higher" if delta > 0 else "lower"

    chains = []
    chains.append(
        f"Baseline (expected): {base:.0f} units. "
        f"Forecast: {pred:.0f} units ({delta_dir} by {abs(delta):.0f} units)."
    )

    for c in attribution.get("shap_contributions", [])[:5]:
        direction = "↑" if c["effective_sign"] == "+" else "↓"
        chains.append(
            f"{c['label']} ({direction}{c['abs_impact']:.1f}): {c['explanation']}"
        )

    # Add external context
    ctx = attribution.get("external_context", {})
    if ctx.get("is_ramadan"):
        chains.append(ctx.get("ramadan_note", ""))
    if ctx.get("is_public_holiday"):
        chains.append(f"Public holiday: {ctx.get('holiday_name', '')} — expect demand shift")

    return chains


def build_llm_causal_prompt(attribution: Dict, product: str, date_str: str) -> str:
    """Build a prompt for DeepSeek LLM to generate a human-readable causal report."""
    chains = generate_causal_chain(attribution)
    chain_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(chains))

    prompt = f"""You are a bakery operations analyst. Given the following SHAP-based causal attribution for a demand forecast, write a concise 2-3 sentence explanation a bakery manager can act on.

Product: {product}
Date: {date_str}
Predicted demand: {attribution.get('predicted_demand', '?')} units
Baseline (expected): {attribution.get('base_value', '?')} units

Feature contributions (SHAP):
{chain_text}

Rules:
- Be specific: name the 1-2 biggest drivers
- If weekend/holiday/Ramadan is the driver, mention it
- Suggest one actionable takeaway
- Keep it to 2-3 sentences
- Use plain English (no technical terms like SHAP/feature)
"""
    return prompt


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test with tomorrow"s croissant forecast
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    attr = compute_shap_attribution("croissant", tomorrow)
    print(json.dumps(attr, indent=2, default=str))
    print("\n=== Causal Chain ===")
    for line in generate_causal_chain(attr):
        print(line)
