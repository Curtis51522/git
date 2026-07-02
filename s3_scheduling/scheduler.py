#!/usr/bin/env python
"""
S3 CP-SAT Production Scheduler
================================
Daily production optimization: decides bake quantities for all 30 bread products
given S2 demand forecasts, carryover stock, freshness tiers, and capacity constraints.

Features:
  - Single-day optimization with Day-1 carryover
  - Freshness: sell Day-1 first (50% price), then Fresh (full price)
  - Multi-scenario: Q10 / Q50 / Q90 demand levels
  - Raw material estimation from product recipes
  - 7-day rolling plan (placeholder for multi-period extension)
  - Output: JSON bake plan + dashboard-ready summary

Formulation:
  maximize  sum( price_i * fresh_sold_i + discount_price_i * day1_sold_i
                - cost_i * bake_i - waste_cost_i * waste_i
                - stockout_cost_i * shortage_i )
  s.t.      day1_sold_i <= day1_stock_i
            fresh_sold_i <= bake_i
            day1_sold_i + fresh_sold_i <= demand_i
            sum(bake_i) <= capacity
            all vars >= 0, integer

Usage:
  python s3_scheduling/scheduler.py                    # demo with sample data
  python s3_scheduling/scheduler.py --full             # full 30-product solve
  python s3_scheduling/scheduler.py --save plan.json   # save to file
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ortools.sat.python import cp_model

# ============================================================
# CONFIG
# ============================================================
import os as _os
_BASE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.path.join(_BASE_DIR, "data")
OUT_DIR = _os.path.join(_BASE_DIR, "s3_scheduling", "outputs")
_os.makedirs(OUT_DIR, exist_ok=True)

RAW_CSV = _os.path.join(DATA_DIR, "bakery_sales_raw.csv")

BREAD_CAPACITY = 800
DRINK_CAPACITY = 300
DAY1_DISCOUNT = 0.50          # Day-1 sold at 50% of fresh price
PRODUCTION_COST_RATIO = 0.30
WASTE_COST_RATIO = 0.50       # % of production cost
STOCKOUT_COST_RATIO = 0.25    # % of selling price
DEMAND_BUFFER = 1.05          # 5% buffer on demand (dashboard-designs Panel 2)
SCENARIO_LABELS = ["q10", "q50", "q90"]

DRINK_NAMES = {
    "latte", "americano", "cappuccino", "mocha", "espresso",
    "flat_white", "caramel_macchiato", "cold_brew",
    "hot_chocolate", "matcha_latte", "milk_tea", "chai_latte",
    "earl_grey", "english_breakfast", "lemonade",
}

# ============================================================
# PRODUCT DATA
# ============================================================
def load_products():
    """Return {product_name: {price, is_drink}} for all 45 products from MySQL."""
    from db.mysql_client import get_db
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT product_name, category, unit_price FROM products")
    result = {}
    for row in cur.fetchall():
        name = str(row[0])
        category = str(row[1]) if row[1] else ""
        price = float(row[2]) if row[2] else 0.0
        result[name] = {
            "price": round(price, 2),
            "is_drink": category == "beverage",
        }
    # Fill any missing breads from DRINK_NAMES
    for drink in DRINK_NAMES:
        if drink not in result:
            result[drink] = {"price": 12.0, "is_drink": True}
    return result


def load_s2_predictions(date_str=None):
    """
    Load S2 XGBoost quantile predictions for a given date.

    Uses the trained quantile models from s2_forecasting/outputs/.
    Falls back to sampling from training data if models unavailable.

    Returns: {product_name: {"q10": int, "q50": int, "q90": int}}
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Try real S2 models first
    preds = _predict_from_api(date_str)
    if preds:
        return preds

    # Fallback: sample from training data
    return _predict_from_training(date_str)


def generate_7day_s2_forecast(start_date):
    """
    Generate 7 days of S2 predictions via forecast API.

    Args:
        start_date: str "YYYY-MM-DD" (Monday)

    Returns:
        dict {date: {product_name: {"q50": int, "lower": int, "upper": int}}}
    """
    d0 = datetime.strptime(start_date, "%Y-%m-%d")
    forecast = {}
    for i in range(7):
        ds = (d0 + timedelta(days=i)).strftime("%Y-%m-%d")
        forecast[ds] = load_s2_predictions(ds)
    return forecast


# ---- Internal helpers ----

_S2_MODELS = None
_S2_META = None


def _init_s2_models():
    """Lazy-load S2 quantile models and metadata once."""
    global _S2_MODELS, _S2_META
    if _S2_MODELS is not None:
        return True

    import pickle as _pk
    _S2_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "s2_forecasting", "outputs")
    _MODEL_Q10 = _os.path.join(_S2_DIR, "quantile_model_q10.pkl")
    _MODEL_Q50 = _os.path.join(_S2_DIR, "quantile_model_q50.pkl")
    _MODEL_Q90 = _os.path.join(_S2_DIR, "quantile_model_q90.pkl")

    if not all(_os.path.exists(p) for p in [_MODEL_Q10, _MODEL_Q50, _MODEL_Q90]):
        return False

    _S2_MODELS = {
        "q10": _pk.load(open(_MODEL_Q10, "rb")),
        "q50": _pk.load(open(_MODEL_Q50, "rb")),
        "q90": _pk.load(open(_MODEL_Q90, "rb")),
    }

    # Build product mapping + weather means from training data
    _TRAIN_CSV = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "xgboost_train.csv")
    train = pd.read_csv(_TRAIN_CSV)
    train["date"] = pd.to_datetime(train["date"])

    # Product name mapping
    _RAW = pd.read_csv(RAW_CSV)
    all_products = sorted(_RAW["product_name"].unique())
    _S2_META = {
        "pid_to_name": {i: p for i, p in enumerate(all_products)},
        "name_to_pid": {p: i for i, p in enumerate(all_products)},
        "weather_monthly": train.groupby("month")[["temp_mean", "temp_max", "temp_min", "humidity", "precipitation"]].mean().to_dict("index"),
        "last_day": train[train["date"] == train["date"].max()][["product_id", "lag_1", "lag_7_avg", "lag_30_avg", "quantity"]].copy(),
        "feature_cols": ["product_id", "temp_mean", "temp_max", "temp_min", "humidity", "precipitation", "day_of_week", "month", "is_weekend", "is_holiday", "lag_1", "lag_7_avg", "lag_30_avg"],
    }
    return True


def _predict_from_models(date_str):
    """Generate predictions using trained XGBoost quantile models."""
    if not _init_s2_models():
        return None

    dt = pd.Timestamp(date_str)
    m = dt.month
    w = dt.weekday()
    wm = _S2_META["weather_monthly"].get(m)
    if wm is None:
        return None

    # Build feature rows for all product_ids
    rows = []
    product_ids = sorted(_S2_META["last_day"]["product_id"].unique())
    for pid in product_ids:
        ld = _S2_META["last_day"][_S2_META["last_day"]["product_id"] == pid]
        if len(ld) > 0:
            l1 = ld["quantity"].values[0]
            l7 = ld["lag_7_avg"].values[0]
            l30 = ld["lag_30_avg"].values[0]
        else:
            l1 = l7 = l30 = 0

        rows.append({
            "product_id": pid,
            "temp_mean": wm.get("temp_mean", 20),
            "temp_max": wm.get("temp_max", 25),
            "temp_min": wm.get("temp_min", 15),
            "humidity": wm.get("humidity", 70),
            "precipitation": wm.get("precipitation", 3),
            "day_of_week": w,
            "month": m,
            "is_weekend": 1 if w >= 5 else 0,
            "is_holiday": 0,
            "lag_1": l1,
            "lag_7_avg": l7,
            "lag_30_avg": l30,
        })

    X = pd.DataFrame(rows)[_S2_META["feature_cols"]]

    predictions = {}
    for pid in product_ids:
        name = _S2_META["pid_to_name"].get(pid, str(pid))
        row = X[X["product_id"] == pid]
        q10_val = max(0, int(round(_S2_MODELS["q10"].predict(row)[0])))
        q50_val = max(0, int(round(_S2_MODELS["q50"].predict(row)[0])))
        q90_val = max(0, int(round(_S2_MODELS["q90"].predict(row)[0])))
        predictions[name] = {"q10": q10_val, "q50": q50_val, "q90": q90_val}

    return predictions


def _predict_from_api(date_str):
    """Get S2 predictions by importing the forecast module directly."""
    try:
        import sys as _sys
        _base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _base not in _sys.path:
            _sys.path.insert(0, _base)
        from api.module2_forecast import _do_forecast
        from datetime import datetime as _dt
        data = _do_forecast(None, 1, False, date_str)
        if data.get("status") != "ok":
            return None
        result = {}
        for f in data.get("forecasts", []):
            result[f["product_name"]] = {
                "q50": f["predicted_demand"],
                "q10": f["lower_bound"],
                "q90": f["upper_bound"],
            }
        return result if result else None
    except Exception:
        return None

def _predict_from_training(date_str):
    """Fallback: sample predictions from historical training data."""
    train = pd.read_csv(_os.path.join(DATA_DIR, "xgboost_train.csv"))
    train["date"] = pd.to_datetime(train["date"])
    target_date = pd.Timestamp(date_str)

    mask = (train["date"].dt.month.isin([
        max(1, target_date.month - 1),
        target_date.month,
        min(12, target_date.month + 1)
    ]))
    subset = train[mask]

    # Build product name mapping
    try:
        _RAW2 = pd.read_csv(RAW_CSV)
        all_products2 = sorted(_RAW2["product_name"].unique())
        pid2name = {i: p for i, p in enumerate(all_products2)}
    except Exception:
        pid2name = {}

    predictions = {}
    for pid in sorted(subset["product_id"].unique()):
        sub = subset[subset["product_id"] == pid]
        if len(sub) < 5:
            continue
        vals = sub["quantity"].values
        name = pid2name.get(pid, str(pid))
        predictions[name] = {
            "q10": int(np.percentile(vals, 10)),
            "q50": int(np.percentile(vals, 50)),
            "q90": int(np.percentile(vals, 90)),
        }

    return predictions


# ============================================================
# RAW MATERIAL ESTIMATION
# ============================================================
# Per-product recipes: grams/ml per unit for each bread product
# Bakers percentages from verified professional sources, converted to per-unit.
#
# Sources:
#   Croissant:         BITA (Baking Industry Training Australia) - bita.org.au
#   Brioche:           King Arthur Baking Professional Formulas - kingarthurbaking.com/pro/formulas/brioche
#   Donut:             Seasoned Advice StackExchange (tested formula) - cooking.stackexchange.com/q/49438
#   Brownie:           CIA Baking and Pastry cookbook - thebutterchronicles.wordpress.com
#   Egg Tart / Pastry: The Flavor Bender (Pate Sucree standard) - theflavorbender.com
#   Baguette:          French lean bread standard (68% hydration) - sourdoughhydration.com
#   Sourdough:         Standard artisan formula - theperfectloaf.com
#   Chiffon Cake:      Baking Forums standard ratio - baking-forums.com
#   Cookie/Muffin:     Standard shortbread/quickbread ratios
#   Macaron:           Standard French macaron formula
#   Lean breads:       Baguette/sourdough ratios adapted per variant
#   Enriched breads:   Brioche/donut ratios adapted per variant
PRODUCT_RECIPES = {
    # egg_whole_g = whole egg (yolk+white together)
    # egg_yolk_g  = pure yolk only (white is byproduct)
    # egg_white_g = pure white only (yolk is byproduct)
    # 1 egg approx 50g (20g yolk + 30g white)
    #
    # Verified sources per product listed inline.

    # ==== LEAN BREADS: no egg ====
    # Source: French lean bread standard (68-75% hydration, flour+water+salt+yeast only)
    "baguette":     {"flour_g":60, "butter_g":0,  "sugar_g":0,  "egg_whole_g":0,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":0,   "chocolate_g":0},
    # Source: The Perfect Loaf, SourdoughHydration (flour+water+salt+starter only)
    "sourdough":    {"flour_g":65, "butter_g":0,  "sugar_g":0,  "egg_whole_g":0,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":0,   "chocolate_g":0},
    # Source: standard flatbread (no enrichment)
    "flatbread":    {"flour_g":55, "butter_g":3,  "sugar_g":0,  "egg_whole_g":0,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":0,   "chocolate_g":0},
    # Source: King Arthur Grissini formula (flour+water+oil+butter+salt+yeast, NO egg) ? kingarthurbaking.com/pro/formulas/grissini
    "stickbread":   {"flour_g":45, "butter_g":6,  "sugar_g":5,  "egg_whole_g":0,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":10,  "chocolate_g":0},
    # Source: standard white sandwich bread (enriched, whole egg)
    "pullman":      {"flour_g":55, "butter_g":5,  "sugar_g":5,  "egg_whole_g":0,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":10,  "chocolate_g":0},
    # Source: standard bagel (boiled, no egg in dough)
    "bagel":        {"flour_g":65, "butter_g":2,  "sugar_g":3,  "egg_whole_g":0,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":0,   "chocolate_g":0},

    # ==== ENRICHED BREADS: whole egg ====
    # Source: ChainBaker ? "enriched dough...they all contain the whole egg" ? chainbaker.com/eggs
    "bread_roll":   {"flour_g":40, "butter_g":8,  "sugar_g":6,  "egg_whole_g":8,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":15,  "chocolate_g":0},
    # Source: Oh My Food Recipes ? "enriched with butter, eggs, and milk" ? ohmyfoodrecipes.com
    "bread_coconut":{"flour_g":40, "butter_g":10, "sugar_g":10, "egg_whole_g":10, "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":20,  "chocolate_g":0},
    # Source: Serious Eats ? "milk, eggs, and butter" ? seriouseats.com/pandesal
    "pandesal":     {"flour_g":40, "butter_g":6,  "sugar_g":8,  "egg_whole_g":8,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":15,  "chocolate_g":0},
    # Source: King Arthur Baking Professional ? kingarthurbaking.com/pro/formulas/brioche
    "brioche":      {"flour_g":38, "butter_g":19, "sugar_g":5,  "egg_whole_g":22, "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":5,   "chocolate_g":0},
    # Source: Feast and Farm ? "2 eggs" in standard cornbread ? feastandfarm.com
    "cornbread":    {"flour_g":45, "butter_g":10, "sugar_g":8,  "egg_whole_g":10, "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":20,  "chocolate_g":0},
    # Source: Instagram reel C7R4TBjpDrO ? "175 grams Eggs" in pan de mantequilla
    "mantequilla":  {"flour_g":40, "butter_g":18, "sugar_g":8,  "egg_whole_g":10, "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":10,  "chocolate_g":0},
    # Source: pizza dough standard (flour+water+yeast+oil, no egg)
    "pizza_bread":  {"flour_g":50, "butter_g":8,  "sugar_g":3,  "egg_whole_g":0,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":10,  "chocolate_g":0},

    # ==== SWEET BREADS: whole egg in enriched dough ====
    # Source: Just One Cookbook ? enriched bread dough with egg ? justonecookbook.com/melon-pan
    "melon_bread":  {"flour_g":42, "butter_g":12, "sugar_g":14, "egg_whole_g":10, "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":15,  "chocolate_g":0},
    # Source: Jessica's Dinner Party ? "whisk in the egg" ? jessicasdinnerparty.com
    "soboru_bread": {"flour_g":42, "butter_g":14, "sugar_g":12, "egg_whole_g":10, "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":12,  "chocolate_g":0},

    # ==== LAMINATED / PASTRY ====
    # Source: BITA (Baking Industry Training Australia) ? bita.org.au ? 10% eggs in dough
    "croissant":         {"flour_g":48, "butter_g":24, "sugar_g":4,  "egg_whole_g":8,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":5,   "chocolate_g":0},
    "croissant_chocolate":{"flour_g":48, "butter_g":22, "sugar_g":4,  "egg_whole_g":8,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":5,   "chocolate_g":15},
    # Source: America's Test Kitchen, NYT ? custard filling uses EGG YOLK, crust uses whole egg
    "eggtart":     {"flour_g":28, "butter_g":14, "sugar_g":12, "egg_whole_g":5,  "egg_yolk_g":15, "egg_white_g":0,  "milk_ml":15,  "chocolate_g":0},
    # Source: La Cucina Italiana ? custard uses 4 EGG YOLKS, puff pastry wrapper NO egg ? lacucinaitaliana.com
    "cream_horn":  {"flour_g":38, "butter_g":20, "sugar_g":8,  "egg_whole_g":0,  "egg_yolk_g":12, "egg_white_g":0,  "milk_ml":10,  "chocolate_g":0},
    # Source: King Arthur Baking apple pie (egg in crust + egg wash) ? kingarthurbaking.com
    "apple_pie":   {"flour_g":45, "butter_g":20, "sugar_g":14, "egg_whole_g":8,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":8,   "chocolate_g":0},
    # Source: standard tostada (lean, toasted, no enrichment)
    "tostada":     {"flour_g":50, "butter_g":8,  "sugar_g":3,  "egg_whole_g":0,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":5,   "chocolate_g":0},

    # ==== DONUTS: whole egg ====
    # Source: StackExchange (tested formula) ? cooking.stackexchange.com/q/49438 ? 25% egg
    "donut":       {"flour_g":35, "butter_g":9,  "sugar_g":18, "egg_whole_g":10, "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":15,  "chocolate_g":0},
    # Source: standard pancake (whole egg)
    "pancake":     {"flour_g":40, "butter_g":8,  "sugar_g":8,  "egg_whole_g":10, "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":30,  "chocolate_g":0},

    # ==== CAKES: separated eggs ====
    # Source: ASBE (American Society of Baking), Serious Eats ? yolks in batter + whites whipped to meringue
    "chiffon":      {"flour_g":25, "butter_g":0,  "sugar_g":30, "egg_whole_g":0,  "egg_yolk_g":18, "egg_white_g":27, "milk_ml":0,   "chocolate_g":0},
    # Source: Christine's Recipes, Professional Baking textbook ? separated-egg sponge method
    "chocolate_cake":{"flour_g":30, "butter_g":15, "sugar_g":25, "egg_whole_g":0,  "egg_yolk_g":12, "egg_white_g":18, "milk_ml":15,  "chocolate_g":25},

    # ==== COOKIES / BARS: whole egg ====
    # Source: standard shortbread/sugar cookie (whole egg in dough)
    "cookie":      {"flour_g":25, "butter_g":15, "sugar_g":18, "egg_whole_g":1,  "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":3,   "chocolate_g":0},
    # Source: Elle & Vire Professionnel ? EGG WHITE ONLY for meringue, no yolks ? elle-et-vire.com
    "macaron":     {"flour_g":8,  "butter_g":0,  "sugar_g":20, "egg_whole_g":0,  "egg_yolk_g":0,  "egg_white_g":10, "milk_ml":0,   "chocolate_g":0},
    # Source: standard muffin quickbread (whole egg)
    "muffin":      {"flour_g":35, "butter_g":14, "sugar_g":18, "egg_whole_g":12, "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":25,  "chocolate_g":0},

    # ==== CHOCOLATE-RICH: whole egg ====
    # Source: CIA Baking and Pastry textbook ? thebutterchronicles.wordpress.com ? 138% eggs (whole)
    "brownie":     {"flour_g":28, "butter_g":38, "sugar_g":60, "egg_whole_g":35, "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":0,   "chocolate_g":50},
    # Source: FoodCraft Korean Choco Pie ? "incorporate the eggs one by one" (whole) ? foodcraft.app
    "chocopie":    {"flour_g":30, "butter_g":12, "sugar_g":22, "egg_whole_g":12, "egg_yolk_g":0,  "egg_white_g":0,  "milk_ml":10,  "chocolate_g":20},
}
_DEFAULT_RECIPE = {"flour_g":45, "butter_g":10, "sugar_g":8, "egg_whole_g":10, "egg_yolk_g":0, "egg_white_g":0, "milk_ml":15, "chocolate_g":0}

def estimate_raw_materials(bake_plan, products):
    """Estimate raw material requirements from bake plan using per-product recipes."""
    materials = {"flour_g": 0, "butter_g": 0, "sugar_g": 0,
                 "egg_whole_g": 0, "egg_yolk_g": 0, "egg_white_g": 0,
                 "milk_ml": 0, "chocolate_g": 0}
    cake_products = {"chiffon", "chocolate_cake"}
    for product, quantity in bake_plan.items():
        if quantity == 0 or product in DRINK_NAMES:
            continue
        recipe = PRODUCT_RECIPES.get(product, _DEFAULT_RECIPE)
        for key in materials:
            materials[key] += quantity * recipe[key]
    # Derive secondary materials from primary ones
    total_flour = materials["flour_g"]
    derived = {
        "cake_flour_g": 0,
        "baking_powder_g": round(total_flour * 0.02, 1),  # 2% of flour
        "salt_g": round(total_flour * 0.015, 1),            # 1.5% of flour
        "yeast_g": round(total_flour * 0.01, 1),            # 1% of flour
        "coffee_beans_g": 0,   # estimated separately from beverages
        "tea_leaves_g": 0,     # estimated separately from beverages
        "cups_pcs": 0,         # per beverage unit
        "cup_large_pcs": 0,
        "cup_regular_pcs": 0,
        "lids_pcs": 0,
        "box_pcs": 0,
        "packaging_bag_pcs": 0,
        "packaging_box_pcs": 0,
    }
    # Reassign cake flour for cake products
    for product, quantity in bake_plan.items():
        if product in cake_products and quantity > 0:
            recipe = PRODUCT_RECIPES.get(product, _DEFAULT_RECIPE)
            cake_flour = quantity * recipe.get("flour_g", 0)
            derived["cake_flour_g"] += cake_flour
    derived["cake_flour_g"] = round(derived["cake_flour_g"], 1)
    # Packaging: 1 packaging bag per 3 bread items
    total_bread_units = sum(q for p, q in bake_plan.items() if p not in DRINK_NAMES)
    derived["packaging_bag_pcs"] = round(total_bread_units / 3, 1)
    derived["packaging_box_pcs"] = round(total_bread_units / 6, 1)  # boxes for larger orders
    materials.update(derived)
    return {k: round(v, 1) for k, v in materials.items()}
_DEFAULT_RECIPE = {"flour_g":45, "butter_g":10, "sugar_g":8, "egg_whole_g":10, "egg_yolk_g":0, "egg_white_g":0, "milk_ml":15, "chocolate_g":0}

def estimate_raw_materials(bake_plan, products):
    """Estimate raw material requirements from bake plan using per-product recipes."""
    materials = {"flour_g": 0, "butter_g": 0, "sugar_g": 0,
                 "egg_whole_g": 0, "egg_yolk_g": 0, "egg_white_g": 0,
                 "milk_ml": 0, "chocolate_g": 0}
    cake_products = {"chiffon", "chocolate_cake"}
    for product, quantity in bake_plan.items():
        if quantity == 0 or product in DRINK_NAMES:
            continue
        recipe = PRODUCT_RECIPES.get(product, _DEFAULT_RECIPE)
        for key in materials:
            materials[key] += quantity * recipe[key]
    # Derive secondary materials from primary ones
    total_flour = materials["flour_g"]
    derived = {
        "cake_flour_g": 0,
        "baking_powder_g": round(total_flour * 0.02, 1),  # 2% of flour
        "salt_g": round(total_flour * 0.015, 1),            # 1.5% of flour
        "yeast_g": round(total_flour * 0.01, 1),            # 1% of flour
        "coffee_beans_g": 0,   # estimated separately from beverages
        "tea_leaves_g": 0,     # estimated separately from beverages
        "cups_pcs": 0,         # per beverage unit
        "cup_large_pcs": 0,
        "cup_regular_pcs": 0,
        "lids_pcs": 0,
        "box_pcs": 0,
        "packaging_bag_pcs": 0,
        "packaging_box_pcs": 0,
    }
    # Reassign cake flour for cake products
    for product, quantity in bake_plan.items():
        if product in cake_products and quantity > 0:
            recipe = PRODUCT_RECIPES.get(product, _DEFAULT_RECIPE)
            cake_flour = quantity * recipe.get("flour_g", 0)
            derived["cake_flour_g"] += cake_flour
    derived["cake_flour_g"] = round(derived["cake_flour_g"], 1)
    # Packaging: 1 packaging bag per 3 bread items
    total_bread_units = sum(q for p, q in bake_plan.items() if p not in DRINK_NAMES)
    derived["packaging_bag_pcs"] = round(total_bread_units / 3, 1)
    derived["packaging_box_pcs"] = round(total_bread_units / 6, 1)  # boxes for larger orders
    materials.update(derived)
    return {k: round(v, 1) for k, v in materials.items()}
# ============================================================
# CP-SAT SOLVER
# ============================================================
class Scheduler:
    def __init__(self):
        self.products = load_products()
        self.breads = {p: v for p, v in self.products.items() if not v["is_drink"]}
        self.drinks = {p: v for p, v in self.products.items() if v["is_drink"]}

    def solve(self, day1_stock, demand, capacity=None, verbose=False):
        """Single-day production optimization with freshness tiers."""
        if capacity is None:
            capacity = BREAD_CAPACITY
        active = [p for p in self.breads
                  if day1_stock.get(p, 0) > 0 or demand.get(p, 0) > 0]
        if not active:
            return self._empty_result()
        model = cp_model.CpModel()
        bake = {}
        fresh_sold = {}
        day1_sold = {}
        for p in active:
            s = day1_stock.get(p, 0)
            d = demand.get(p, 0)
            max_sell = s + capacity
            bake[p] = model.NewIntVar(0, capacity, f"bake_{p}")
            fresh_sold[p] = model.NewIntVar(0, max_sell, f"fresh_sold_{p}")
            day1_sold[p] = model.NewIntVar(0, min(s, d), f"day1_sold_{p}")
            model.Add(day1_sold[p] <= s)
            model.Add(fresh_sold[p] <= bake[p])
            model.Add(day1_sold[p] + fresh_sold[p] <= d)
        model.Add(sum(bake.values()) <= capacity)
        objective_terms = []
        for p in active:
            price = self.products[p]["price"]
            day1_price = price * DAY1_DISCOUNT
            prod_cost = price * PRODUCTION_COST_RATIO
            waste_cost = prod_cost * WASTE_COST_RATIO
            stockout_cost = price * STOCKOUT_COST_RATIO
            s = day1_stock.get(p, 0)
            d = demand.get(p, 0)
            revenue = int(day1_price * 100) * day1_sold[p] + int(price * 100) * fresh_sold[p]
            cost_bake = int(prod_cost * 100) * bake[p]
            cost_waste = int(waste_cost * 100) * (s + bake[p] - day1_sold[p] - fresh_sold[p])
            cost_shortage = int(stockout_cost * 100) * (d - day1_sold[p] - fresh_sold[p])
            objective_terms.append(revenue - cost_bake - cost_waste - cost_shortage)
        model.Maximize(sum(objective_terms))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        solver.parameters.num_search_workers = 8
        if not verbose:
            solver.parameters.log_search_progress = False
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._empty_result()
        return self._collect_results(active, day1_stock, demand, bake, fresh_sold, day1_sold, solver, capacity)

    def _empty_result(self):
        return {
            "bake_plan": {}, "fresh_sold": {}, "day1_sold": {},
            "waste": {}, "shortage": {}, "profit": 0.0,
            "revenue": 0.0, "total_bake": 0, "capacity_used_pct": 0,
            "status": "INFEASIBLE", "materials": {},
        }

    def _collect_results(self, active, day1_stock, demand, bake, fresh_sold, day1_sold, solver, capacity):
        result = {
            "bake_plan": {}, "fresh_sold": {}, "day1_sold": {},
            "waste": {}, "shortage": {}, "profit": 0.0, "revenue": 0.0,
            "total_bake": 0, "status": "OPTIMAL",
        }
        for p in active:
            b = solver.Value(bake[p])
            fs = solver.Value(fresh_sold[p])
            ds = solver.Value(day1_sold[p])
            s = day1_stock.get(p, 0)
            d = demand.get(p, 0)
            price = self.products[p]["price"]
            w = s + b - ds - fs
            sh = d - ds - fs
            result["bake_plan"][p] = b
            result["fresh_sold"][p] = fs
            result["day1_sold"][p] = ds
            result["waste"][p] = w
            result["shortage"][p] = sh
            profit_p = (price * fs + price * DAY1_DISCOUNT * ds
                       - price * PRODUCTION_COST_RATIO * b
                       - price * PRODUCTION_COST_RATIO * WASTE_COST_RATIO * w
                       - price * STOCKOUT_COST_RATIO * sh)
            result["profit"] += profit_p
            result["revenue"] += price * fs + price * DAY1_DISCOUNT * ds
            result["total_bake"] += b
        result["profit"] = round(result["profit"], 2)
        result["revenue"] = round(result["revenue"], 2)
        result["capacity_used_pct"] = round(result["total_bake"] / capacity * 100, 1)
        result["materials"] = estimate_raw_materials(result["bake_plan"], self.products)
        return result

    def solve_scenarios(self, day1_stock, q10, q50, q90, capacity=None):
        """Solve with Q50 and evaluate the plan against Q10 and Q90."""
        base = self.solve(day1_stock, q50, capacity)
        if base["status"] == "INFEASIBLE":
            return base
        base["scenario_q10"] = self._eval(base["bake_plan"], day1_stock, q10)
        base["scenario_q50"] = self._eval(base["bake_plan"], day1_stock, q50)
        base["scenario_q90"] = self._eval(base["bake_plan"], day1_stock, q90)
        base["demand_q50"] = q50
        return base

    def _eval(self, bake_plan, day1_stock, demand):
        """Evaluate a fixed bake plan under a demand scenario."""
        profit = 0.0
        waste_total = 0
        shortage_total = 0
        sales_total = 0
        for p, b in bake_plan.items():
            s = day1_stock.get(p, 0)
            d = demand.get(p, 0)
            price = self.products[p]["price"]
            ds = min(s, d)
            remaining_demand = d - ds
            fs = min(b, remaining_demand)
            w = s + b - ds - fs
            sh = d - ds - fs
            profit += (price * fs + price * DAY1_DISCOUNT * ds
                      - price * PRODUCTION_COST_RATIO * b
                      - price * PRODUCTION_COST_RATIO * WASTE_COST_RATIO * w
                      - price * STOCKOUT_COST_RATIO * sh)
            waste_total += w
            shortage_total += sh
            sales_total += ds + fs
        return {
            "profit": round(profit, 2),
            "waste_units": waste_total,
            "shortage_units": shortage_total,
            "sales_units": sales_total,
        }



    def generate_7day_plan(self, start_date, day1_stock, demand_forecast_7day):
        """
        Generate a 7-day rolling production plan with scenario support.
        Each day solves with Q50 demand x DEMAND_BUFFER, then evaluates
        against Q10/Q50/Q90 for scenario analysis.
        """
        plans = []
        stock = dict(day1_stock)
        date = datetime.strptime(start_date, "%Y-%m-%d")
        for day_offset in range(7):
            date_str = date.strftime("%Y-%m-%d")
            q50_demand = {}
            q10_demand = {}
            q90_demand = {}
            if date_str in demand_forecast_7day:
                for p, pred in demand_forecast_7day[date_str].items():
                    raw_q50 = pred.get("q50", 0)
                    q50_demand[p] = max(0, int(raw_q50 * DEMAND_BUFFER))
                    q10_demand[p] = max(0, int(pred.get("q10", raw_q50) * DEMAND_BUFFER))
                    q90_demand[p] = max(0, int(pred.get("q90", raw_q50) * DEMAND_BUFFER))
            else:
                for p in self.breads:
                    q50_demand[p] = 10
                    q10_demand[p] = 5
                    q90_demand[p] = 16
            result = self.solve_scenarios(stock, q10_demand, q50_demand, q90_demand)
            plans.append({"date": date_str, **result})
            next_stock = {}
            for p in self.breads:
                b = result["bake_plan"].get(p, 0)
                fs = result["fresh_sold"].get(p, 0)
                unsold_fresh = max(0, b - fs)
                prev_d1 = stock.get(p, 0)
                ds = result["day1_sold"].get(p, 0)
                unsold_d1 = max(0, prev_d1 - ds)
                next_stock[p] = unsold_fresh + unsold_d1
            stock = next_stock
            date += timedelta(days=1)
        weekly = self._aggregate_weekly(plans)
        return {
            "plans": plans,
            "weekly_summary": weekly,
            "dashboard_7day": self.dashboard_format_7day(plans, start_date),
            "dashboard_materials": self.dashboard_format_materials(weekly, plans, demand_forecast_7day),
        }

    def _aggregate_weekly(self, plans):
        """Aggregate 7 daily plans into a weekly summary."""
        agg = {
            "total_bake": 0, "total_profit": 0.0, "total_revenue": 0.0,
            "total_waste": 0, "total_shortage": 0, "total_sales": 0,
            "daily_profits": [],
            "scenarios": {"q10": {"profit": 0, "waste": 0, "shortage": 0},
                          "q50": {"profit": 0, "waste": 0, "shortage": 0},
                          "q90": {"profit": 0, "waste": 0, "shortage": 0}},
            "materials": {}, "top_products": [],
        }
        product_bakes = {}
        for plan in plans:
            agg["total_bake"] += plan.get("total_bake", 0)
            agg["total_profit"] += plan.get("profit", 0)
            agg["total_revenue"] += plan.get("revenue", 0)
            agg["total_waste"] += sum(plan.get("waste", {}).values())
            agg["total_shortage"] += sum(plan.get("shortage", {}).values())
            agg["total_sales"] += sum(plan.get("fresh_sold", {}).values()) + sum(plan.get("day1_sold", {}).values())
            agg["daily_profits"].append(round(plan.get("profit", 0), 2))
            for p, b in plan.get("bake_plan", {}).items():
                product_bakes[p] = product_bakes.get(p, 0) + b
        for scen in SCENARIO_LABELS:
            key = f"scenario_{scen}"
            for plan in plans:
                sc = plan.get(key, {})
                agg["scenarios"][scen]["profit"] += sc.get("profit", 0)
                agg["scenarios"][scen]["waste"] += sc.get("waste_units", 0)
                agg["scenarios"][scen]["shortage"] += sc.get("shortage_units", 0)
            agg["scenarios"][scen]["profit"] = round(agg["scenarios"][scen]["profit"], 2)
        mats = {}
        for plan in plans:
            for m, v in plan.get("materials", {}).items():
                mats[m] = mats.get(m, 0) + v
        agg["materials"] = {k: round(v, 1) for k, v in mats.items()}
        agg["top_products"] = sorted(product_bakes.items(), key=lambda x: -x[1])[:10]
        return agg

    def dashboard_format_7day(self, plans, start_date):
        """Format for Forecasting Dashboard Panel 2: 7-day Production Plan."""
        dates = [(datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        grid = []
        for i, plan in enumerate(plans):
            grid.append({
                "date": dates[i],
                "bake_total": plan.get("total_bake", 0),
                "capacity_pct": plan.get("capacity_used_pct", 0),
                "profit": plan.get("profit", 0),
                "revenue": plan.get("revenue", 0),
                "waste_units": sum(plan.get("waste", {}).values()),
                "shortage_units": sum(plan.get("shortage", {}).values()),
                "bake_plan": plan.get("bake_plan", {}), "top_5_bakes": sorted(plan.get("bake_plan", {}).items(), key=lambda x: -x[1])[:5],
                "materials": plan.get("materials", {}),
            })
        return {
            "week_start": dates[0], "week_end": dates[-1],
            "dates": dates, "buffer_applied": DEMAND_BUFFER, "grid": grid,
        }

    def dashboard_format_materials(self, weekly_summary, plans=None, forecast=None):
        """Format for Forecasting Dashboard Panel 3: Raw Material Procurement."""
        # Material key to DB mapping
        _MATERIAL_DB_MAP = {
            "flour_g": ("Bread Flour", 1000),
            "butter_g": ("Butter", 1000),
            "sugar_g": ("Sugar", 1000),
            "egg_whole_g": ("Eggs", 1000),
            "egg_yolk_g": ("Eggs", 1000),
            "egg_white_g": ("Eggs", 1000),
            "milk_ml": ("Milk", 1000),
            "chocolate_g": ("Chocolate", 1000),
            "cake_flour_g": ("Cake Flour", 1000),
            "baking_powder_g": ("Baking Powder", 1000),
            "salt_g": ("Salt", 1000),
            "yeast_g": ("Yeast", 1000),
            "coffee_beans_g": ("Coffee Beans", 1000),
            "tea_leaves_g": ("Tea Leaves", 1000),
            "cups_pcs": ("Cups", 1),
            "cup_large_pcs": ("Cup Large", 1),
            "cup_regular_pcs": ("Cup Regular", 1),
            "lids_pcs": ("Lids", 1),
            "box_pcs": ("Box", 1),
            "packaging_bag_pcs": ("Packaging Bag", 1),
            "packaging_box_pcs": ("Packaging Box", 1),
        }
        # Fetch real stock from database
        db_stock = {}
        try:
            from db.mysql_client import get_db, q
            db = get_db()
            rows = q(db, "raw_materials").select("material_name, stock_quantity, unit").execute()
            if rows.data:
                for r in rows.data:
                    db_stock[r["material_name"]] = {
                        "qty": float(r["stock_quantity"] or 0),
                        "unit": r.get("unit", "kg"),
                    }
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("dashboard_format_materials: DB query failed: %s", e)

        DEFAULT_WASTE = 0.05
        # Aggregate by DB material name (multiple scheduler keys may map to same DB row)
        agg = {}
        for mat_key, weekly_need_g in weekly_summary.get("materials", {}).items():
            db_name, divisor = _MATERIAL_DB_MAP.get(mat_key, (mat_key, 1000))
            if db_name not in agg:
                agg[db_name] = {"weekly_need": 0, "unit": "kg" if divisor > 1 else "pcs"}
            agg[db_name]["weekly_need"] += weekly_need_g / divisor

        # Estimate beverage materials from forecast
        total_beverage_units = 0
        if forecast:
            for d in forecast.values():
                for p, v in d.items():
                    if p in DRINK_NAMES:
                        total_beverage_units += v.get("q50", 0)
        if total_beverage_units > 0:
            # ~15g coffee per cup, ~3g tea per cup
            agg.setdefault("Coffee Beans", {"weekly_need": 0, "unit": "kg"})
            agg.setdefault("Tea Leaves", {"weekly_need": 0, "unit": "kg"})
            agg["Coffee Beans"]["weekly_need"] += round(total_beverage_units * 0.015, 3)
            agg["Tea Leaves"]["weekly_need"] += round(total_beverage_units * 0.003, 3)
            # 1 cup + 1 lid per beverage
            agg.setdefault("Cups", {"weekly_need": 0, "unit": "pcs"})
            agg.setdefault("Cup Large", {"weekly_need": 0, "unit": "pcs"})
            agg.setdefault("Cup Regular", {"weekly_need": 0, "unit": "pcs"})
            agg.setdefault("Lids", {"weekly_need": 0, "unit": "pcs"})
            agg["Cups"]["weekly_need"] += total_beverage_units * 0.5
            agg["Cup Regular"]["weekly_need"] += total_beverage_units * 0.5
            agg["Lids"]["weekly_need"] += total_beverage_units

        procurement = {}
        for db_name, a in agg.items():
            weekly_need_db = round(a["weekly_need"], 2)
            info = db_stock.get(db_name, {"qty": 0, "unit": "kg"})
            stock_db_units = info["qty"]
            unit = info.get("unit", "kg")
            adjusted = round(weekly_need_db * (1 + DEFAULT_WASTE), 2)
            to_order = round(max(0, adjusted - stock_db_units), 2)

            if stock_db_units < weekly_need_db * 0.5:
                alert = "urgent"
            elif stock_db_units < weekly_need_db:
                alert = "order"
            else:
                alert = "ok"

            procurement[db_name] = {
                "weekly_need": weekly_need_db,
                "adjusted_need": adjusted,
                "current_stock": stock_db_units,
                "to_order": to_order,
                "unit": unit,
                "alert": alert,
            }
        return {
            "week": f"{weekly_summary.get('week_start', '-')} ~ {weekly_summary.get('week_end', '-')}",
            "waste_rate_default": DEFAULT_WASTE,
            "items": procurement,
        }


def run_paper_evaluation(save_path=None):
    """
    Paper experiment mode: evaluate S3 scheduler on test set with real lag features.

    Uses S2 8/1/1 split test period (2023-07-01 to 2023-12-31).
    For each 7-day window:
      1. S2 models predict demand using real lag features from test data
      2. S3 scheduler generates bake plan
      3. Compare against actual sales (ground truth quantity)

    Returns:
        dict with weekly and aggregate evaluation metrics suitable for paper.
    """
    _EVAL_TEST_CSV = _os.path.join(DATA_DIR, "xgboost_test.csv")
    _EVAL_META_CSV = _os.path.join(DATA_DIR, "bakery_sales_raw.csv")

    if not _os.path.exists(_EVAL_TEST_CSV):
        print("Test data not found. Run s2_forecasting/preprocess.py first.")
        return None

    test_df = pd.read_csv(_EVAL_TEST_CSV)
    test_df["date"] = pd.to_datetime(test_df["date"])

    # Product mapping
    raw = pd.read_csv(_EVAL_META_CSV)
    all_products = sorted(raw["product_name"].unique())
    pid_to_name = {i: p for i, p in enumerate(all_products)}
    name_to_pid = {p: i for i, p in enumerate(all_products)}

    # Load S2 models
    if not _init_s2_models():
        print("S2 models not available. Cannot run evaluation.")
        return None

    # Weather means (from test period itself for accurate eval)
    weather_test = test_df.groupby("month")[["temp_mean", "temp_max", "temp_min", "humidity", "precipitation"]].mean().to_dict("index")

    # Get all unique weeks in test period
    test_dates = sorted(test_df["date"].unique())
    test_start = test_dates[0]
    # Align to Monday
    if test_start.weekday() != 0:
        test_start -= timedelta(days=test_start.weekday())

    weeks = []
    cursor = test_start
    while cursor <= test_dates[-1]:
        week_end = cursor + timedelta(days=6)
        weeks.append((cursor, week_end))
        cursor += timedelta(days=7)

    print(f"Evaluation: {len(weeks)} weeks from {weeks[0][0].date()} to {weeks[-1][1].date()}")
    print(f"Test period: {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")

    s = Scheduler()
    weekly_results = []
    all_actual_sales = {}
    all_predicted_sales = {}

    for week_idx, (week_start, week_end) in enumerate(weeks):
        start_str = week_start.strftime("%Y-%m-%d")
        end_str = week_end.strftime("%Y-%m-%d")

        # Build 7-day forecast with real lag features from test data
        forecast = {}
        day1_stock = {}
        actual_demand = {}

        for day_offset in range(7):
            day_dt = week_start + timedelta(days=day_offset)
            day_str = day_dt.strftime("%Y-%m-%d")

            day_data = test_df[test_df["date"] == day_str]
            if len(day_data) == 0:
                continue

            day_forecast = {}
            day_actual = {}
            m = day_dt.month
            w = day_dt.weekday()
            wm = weather_test.get(m, {"temp_mean": 20, "temp_max": 25, "temp_min": 15, "humidity": 70, "precipitation": 3})

            for _, row in day_data.iterrows():
                pid = int(row["product_id"])
                name = pid_to_name.get(pid, str(pid))

                # Build feature row with real lag values
                row_df = pd.DataFrame([{
                    "product_id": pid,
                    "temp_mean": wm["temp_mean"],
                    "temp_max": wm["temp_max"],
                    "temp_min": wm["temp_min"],
                    "humidity": wm["humidity"],
                    "precipitation": wm["precipitation"],
                    "day_of_week": w,
                    "month": m,
                    "is_weekend": 1 if w >= 5 else 0,
                    "is_holiday": 0,
                    "lag_1": row["lag_1"],
                    "lag_7_avg": row["lag_7_avg"],
                    "lag_30_avg": row["lag_30_avg"],
                }])[_S2_META["feature_cols"]]

                q10_val = max(0, int(round(_S2_MODELS["q10"].predict(row_df)[0])))
                q50_val = max(0, int(round(_S2_MODELS["q50"].predict(row_df)[0])))
                q90_val = max(0, int(round(_S2_MODELS["q90"].predict(row_df)[0])))

                if name not in DRINK_NAMES:  # Only breads for S3
                    day_forecast[name] = {"q10": q10_val, "q50": q50_val, "q90": q90_val}
                day_actual[name] = int(row["quantity"])

                # Track actual vs predicted
                if name not in all_actual_sales:
                    all_actual_sales[name] = 0
                    all_predicted_sales[name] = 0
                all_actual_sales[name] += int(row["quantity"])
                all_predicted_sales[name] += q50_val

            if day_forecast:
                forecast[day_str] = day_forecast
                actual_demand[day_str] = day_actual

        if week_idx == 0:
            day1_stock = {p: 0 for p in s.breads}

        if not forecast:
            continue

        result = s.generate_7day_plan(start_str, day1_stock, forecast)

        # Compare with actual (breads only)
        total_actual_bread_sales = 0
        total_planned_bake = result["weekly_summary"]["total_bake"]
        daily_comparison = []

        for plan in result["plans"]:
            d = plan["date"]
            if d in actual_demand:
                act = sum(v for k, v in actual_demand[d].items() if k not in DRINK_NAMES)
                daily_comparison.append({
                    "date": d,
                    "planned_bake": plan["total_bake"],
                    "actual_bread_sales": act,
                    "profit": plan["profit"],
                })
                total_actual_bread_sales += act

        weekly_results.append({
            "week_start": start_str,
            "week_end": end_str,
            "total_bake": total_planned_bake,
            "total_actual_bread_sales": total_actual_bread_sales,
            "bake_actual_ratio": round(total_planned_bake / max(total_actual_bread_sales, 1), 3),
            "profit": result["weekly_summary"]["total_profit"],
            "waste": result["weekly_summary"]["total_waste"],
            "shortage": result["weekly_summary"]["total_shortage"],
            "daily": daily_comparison,
        })

        # Update carryover for next week (from this week's last day)
        last_plan = result["plans"][-1]
        for p in s.breads:
            b = last_plan["bake_plan"].get(p, 0)
            fs = last_plan["fresh_sold"].get(p, 0)
            unsold = max(0, b - fs)
            day1_stock[p] = unsold

    # Aggregate evaluation metrics
    total_bake = sum(w["total_bake"] for w in weekly_results)
    total_actual = sum(w["total_actual_bread_sales"] for w in weekly_results)
    total_profit = sum(w["profit"] for w in weekly_results)

    eval_report = {
        "evaluation_period": f"{test_dates[0].date()} to {test_dates[-1].date()}",
        "total_weeks": len(weekly_results),
        "aggregate": {
            "total_bake_planned": total_bake,
            "total_actual_bread_sales": total_actual,
            "bake_to_actual_ratio": round(total_bake / max(total_actual, 1), 3),
            "total_profit_cny": round(total_profit, 2),
            "avg_weekly_profit": round(total_profit / max(len(weekly_results), 1), 2),
            "avg_capacity_pct": round(np.mean([w["total_bake"] for w in weekly_results]) / (BREAD_CAPACITY * 7) * 100, 1),
        },
        "weekly_results": weekly_results,
        "per_product": {},
    }

    # Per-product comparison
    for name in sorted(all_actual_sales.keys()):
        actual = all_actual_sales[name]
        predicted = all_predicted_sales[name]
        eval_report["per_product"][name] = {
            "actual_sales": actual,
            "predicted_q50": predicted,
            "prediction_error_pct": round(abs(predicted - actual) / max(actual, 1) * 100, 1),
        }

    # Print summary
    print("\n" + "=" * 60)
    print("  PAPER EVALUATION RESULTS")
    print("=" * 60)
    agg = eval_report["aggregate"]
    print(f"\n  Period: {eval_report['evaluation_period']} ({len(weekly_results)} weeks)")
    print(f"  Total bake planned:  {agg['total_bake_planned']:>6d} units")
    print(f"  Total actual bread sales:{agg['total_actual_bread_sales']:>6d} units")
    print(f"  Bake/Actual ratio:   {agg['bake_to_actual_ratio']:>6.3f}")
    print(f"  Total profit:        CNY {agg['total_profit_cny']:>10,.2f}")
    print(f"  Avg weekly profit:   CNY {agg['avg_weekly_profit']:>10,.2f}")
    print(f"  Avg capacity usage:  {agg['avg_capacity_pct']:>6.1f}%")

    # Top prediction errors
    print("\n  Top 10 Prediction Errors (|predicted - actual| / actual):")
    errors = sorted(eval_report["per_product"].items(), key=lambda x: x[1]["prediction_error_pct"], reverse=True)[:10]
    for name, info in errors:
        print(f"    {name:30s}: actual={info['actual_sales']:5d}  predicted={info['predicted_q50']:5d}  error={info['prediction_error_pct']:5.1f}%")

    if save_path:
        save_dir = _os.path.dirname(save_path)
        if save_dir:
            _os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(eval_report, f, indent=2, default=str)
        print(f"\n  Saved to {save_path}")

    return eval_report



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3 CP-SAT Production Scheduler")
    parser.add_argument("--full", action="store_true", help="Solve for all 30 bread products")
    parser.add_argument("--7day", action="store_true", help="Run 7-day rolling plan with dashboard output")
    parser.add_argument("--eval", action="store_true", help="Paper evaluation mode: test set with real lag features")
    parser.add_argument("--save", type=str, help="Save result to JSON file")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD")
    args = parser.parse_args()

    s = Scheduler()
    print(f"Products: {len(s.breads)} breads + {len(s.drinks)} drinks")
    print(f"Capacity: {BREAD_CAPACITY} breads/day\n")

    if getattr(args, "eval", False):
        run_paper_evaluation(args.save)
        sys.exit(0)

    elif getattr(args, "7day", False):
        start_date = args.date or datetime.now().strftime("%Y-%m-%d")
        if datetime.strptime(start_date, "%Y-%m-%d").weekday() != 0:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            start_dt -= timedelta(days=start_dt.weekday())
            start_date = start_dt.strftime("%Y-%m-%d")
            print(f"Adjusted to Monday: {start_date}")

        day1_stock = {p: np.random.randint(0, 5) for p in s.breads}
        forecast = generate_7day_s2_forecast(start_date)
        print(f"Using real S2 quantile models for demand forecast")

        result = s.generate_7day_plan(start_date, day1_stock, forecast)
        ws = result["weekly_summary"]

        print("=" * 60)
        print(f"  7-Day Rolling Plan: {start_date} ~ {result['dashboard_7day']['week_end']}")
        print(f"  Buffer applied: {DEMAND_BUFFER}x demand")
        print("=" * 60)
        print(f"\n  Weekly Summary:")
        print(f"    Total bake:    {ws['total_bake']:>5d} units")
        print(f"    Total profit:  CNY {ws['total_profit']:>10,.2f}")
        print(f"    Total revenue: CNY {ws['total_revenue']:>10,.2f}")
        print(f"    Total waste:   {ws['total_waste']:>5d} units")
        print(f"    Total shortage:{ws['total_shortage']:>5d} units")

        print(f"\n  Scenario Analysis:")
        for scen in SCENARIO_LABELS:
            sc = ws["scenarios"][scen]
            print(f"    {scen.upper()}: profit CNY {sc['profit']:>10,.2f}  waste {sc['waste']:4d}  shortage {sc['shortage']:4d}")

        print(f"\n  Daily Breakdown:")
        for plan in result["plans"]:
            p = plan
            print(f"    {p['date']}: bake {p['total_bake']:3d} ({p['capacity_used_pct']:5.1f}%)  profit CNY {p['profit']:>8,.2f}")

        print(f"\n  Top 10 Products (weekly bake):")
        for p, q in ws["top_products"]:
            print(f"    {p:30s}: {q:4d}")

        if args.save:
            output = {
                "week_start": result["dashboard_7day"]["week_start"],
                "week_end": result["dashboard_7day"]["week_end"],
                "weekly_summary": ws,
                "dashboard_7day": result["dashboard_7day"],
                "dashboard_materials": result["dashboard_materials"],
                "plans": result["plans"],
            }
            save_dir = _os.path.dirname(args.save)
            if save_dir:
                _os.makedirs(save_dir, exist_ok=True)
            with open(args.save, "w") as f:
                json.dump(output, f, indent=2, default=str)
            print(f"\n  Saved to {args.save}")

    elif args.full:
        # Full 30-product solve with synthetic stock/demand
        day1_stock = {}
        demand = {}
        for p in s.breads:
            day1_stock[p] = np.random.randint(0, 5)
            demand[p] = np.random.randint(5, 40)
        
        q10 = {p: max(0, demand[p] - np.random.randint(3, 8)) for p in demand}
        q90 = {p: demand[p] + np.random.randint(3, 12) for p in demand}
        
        result = s.solve_scenarios(day1_stock, q10, demand, q90)
    else:
        # Demo with 6 products
        day1_stock = {
            "croissant": 3, "baguette": 1, "donut": 0,
            "sourdough": 0, "croissant_chocolate": 2, "bread_roll": 5,
        }
        q50 = {
            "croissant": 22, "baguette": 5, "donut": 15,
            "sourdough": 4, "croissant_chocolate": 18, "bread_roll": 20,
        }
        q10 = {p: max(0, int(v * 0.55)) for p, v in q50.items()}
        q90 = {p: int(v * 1.55) for p, v in q50.items()}
        
        result = s.solve_scenarios(day1_stock, q10, q50, q90)

    if not getattr(args, "7day", False) and not getattr(args, "eval", False):
        print("=" * 60)
        print(f"  Status: {result['status']}")
        print(f"  Total bake: {result['total_bake']} / {BREAD_CAPACITY} ({result['capacity_used_pct']}%)")
        print(f"  Expected profit: CNY {result['profit']:,.2f}")
        print(f"  Expected revenue: CNY {result['revenue']:,.2f}")
    
        print(f"\n  Bake Plan (top 15 by quantity):")
        sorted_plan = sorted(result["bake_plan"].items(), key=lambda x: -x[1])
        for p, b in sorted_plan[:15]:
            fs = result["fresh_sold"].get(p, 0)
            ds = result["day1_sold"].get(p, 0)
            w = result["waste"].get(p, 0)
            sh = result["shortage"].get(p, 0)
            print(f"    {p:25s}: bake {b:3d}  sell_fresh {fs:3d}  sell_d1 {ds:2d}  waste {w:2d}  short {sh:2d}")
    
        if "scenario_q10" in result:
            print(f"\n  Scenario Analysis:")
            for scen in ["scenario_q10", "scenario_q50", "scenario_q90"]:
                sc = result.get(scen, {})
                print(f"    {scen[-3:]}: profit CNY {sc['profit']:>8,.2f}  "
                      f"waste {sc['waste_units']:3d}  shortage {sc['shortage_units']:3d}  "
                      f"sales {sc['sales_units']:3d}")
    
        if result.get("materials"):
            print(f"\n  Raw Materials Needed:")
            for mat, amount in result["materials"].items():
                print(f"    {mat:15s}: {amount:>8.1f}")

        if args.save:
            save_dir = _os.path.dirname(args.save)
            if save_dir:
                _os.makedirs(save_dir, exist_ok=True)
            with open(args.save, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n  Saved to {args.save}")

