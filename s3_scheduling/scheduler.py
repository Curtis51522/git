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
  - 7-day rolling plan with fresh-to-Day-1 stock carryover
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
  python s3_scheduling/scheduler.py --7day --date 2026-06-30 --save plan.json
  python s3_scheduling/scheduler.py --eval --save paper_eval.json
"""

import argparse
import json
import logging
import os as _os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from ortools.sat.python import cp_model
from s2_forecasting.feature_contract import FORECAST_FEATURES

# ============================================================
# CONFIG
# ============================================================
_BASE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.path.join(_BASE_DIR, "data")
OUT_DIR = _os.path.join(_BASE_DIR, "s3_scheduling", "outputs")
_os.makedirs(OUT_DIR, exist_ok=True)

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
    return result


def generate_7day_s2_forecast(start_date):
    """
    Generate 7 days of S2 predictions via forecast API.

    Args:
        start_date: str "YYYY-MM-DD" (Monday)

    Returns:
        dict {date: {product_name: {"q50": int, "lower": int, "upper": int}}}
    """
    from api.module2_forecast import _do_forecast

    d0 = datetime.strptime(start_date, "%Y-%m-%d")
    forecast = {
        (d0 + timedelta(days=offset)).strftime("%Y-%m-%d"): {}
        for offset in range(7)
    }
    try:
        response = _do_forecast(None, 7, True, start_date)
    except Exception as exc:
        raise RuntimeError("S2 forecast generation failed") from exc

    rows = response.get("forecasts", []) if response.get("status") == "ok" else []
    if not rows:
        raise RuntimeError("S2 forecast returned no rows")

    for row in rows:
        forecast_date = str(row.get("forecast_date", ""))
        product_name = str(row.get("product_name", ""))
        if forecast_date not in forecast or not product_name:
            continue
        forecast[forecast_date][product_name] = {
            "q10": int(row.get("lower_bound", 0) or 0),
            "q50": int(row.get("predicted_demand", 0) or 0),
            "q90": int(row.get("upper_bound", 0) or 0),
        }

    missing_dates = [date for date, products in forecast.items() if not products]
    if missing_dates:
        raise RuntimeError("S2 forecast missing dates: " + ", ".join(missing_dates))
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

    with open(_MODEL_Q10, "rb") as model_file:
        q10_model = _pk.load(model_file)
    with open(_MODEL_Q50, "rb") as model_file:
        q50_model = _pk.load(model_file)
    with open(_MODEL_Q90, "rb") as model_file:
        q90_model = _pk.load(model_file)
    _S2_MODELS = {"q10": q10_model, "q50": q50_model, "q90": q90_model}
    _S2_META = {"feature_cols": list(FORECAST_FEATURES)}
    return True


# ============================================================
# RAW MATERIAL ESTIMATION
# ============================================================
# Loads recipes from MySQL product_recipes table

def load_recipe_from_db(product_name):
    """Query product_recipes table for a single product's materials (kg/pcs/L per unit)."""
    try:
        from db.mysql_client import get_db
        db = get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT material_name, quantity_per_unit FROM product_recipes WHERE product_name = %s",
            (product_name,)
        )
        rows = cur.fetchall()
        cur.close()
        return {r[0]: float(r[1]) for r in rows} if rows else {}
    except Exception:
        logging.getLogger(__name__).exception("Failed to load recipe for %s", product_name)
        return {}

def estimate_raw_materials(bake_plan):
    """
    Estimate raw material needs from a bake plan using DB product_recipes.

    Returns dict: {material_name: total_kg_or_pcs}
    """
    materials = {}  # material_name -> total (kg for dry, L for liquid, pcs for items)

    for product_key, quantity in bake_plan.items():
        if quantity <= 0:
            continue

        recipe = load_recipe_from_db(product_key)
        if not recipe:
            continue

        for mat_name, qty_per_unit in recipe.items():
            total = quantity * qty_per_unit
            materials[mat_name] = materials.get(mat_name, 0) + total

    # Round all values
    return {k: round(v, 3) for k, v in materials.items() if v > 0}

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
            model.Add(day1_sold[p] == min(s, d))
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
        result["materials"] = estimate_raw_materials(result["bake_plan"])
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

    def replay_actual_outcome(self, bake_plan, day1_stock, actual_demand):
        """
        Replay a fixed bake plan against actual sales.

        Day-1 stock is sold first at discount. Unsold Day-1 stock expires at the
        end of the day, while unsold fresh bake becomes the next day's Day-1 stock.
        """
        profit = 0.0
        revenue = 0.0
        waste_total = 0
        shortage_total = 0
        sales_total = 0
        demand_total = 0
        day1_sold = {}
        fresh_sold = {}
        waste = {}
        shortage = {}
        next_day1_stock = {}

        product_names = set(self.breads) | set(bake_plan) | set(day1_stock) | set(actual_demand)
        for product in sorted(product_names):
            if product not in self.products or self.products[product].get("is_drink"):
                continue

            price = self.products[product]["price"]
            bake_units = int(bake_plan.get(product, 0))
            stock_units = int(day1_stock.get(product, 0))
            demand_units = int(actual_demand.get(product, 0))

            sold_day1 = min(stock_units, demand_units)
            remaining_demand = max(0, demand_units - sold_day1)
            sold_fresh = min(bake_units, remaining_demand)
            expired_day1 = max(0, stock_units - sold_day1)
            unsold_fresh = max(0, bake_units - sold_fresh)
            unmet_demand = max(0, demand_units - sold_day1 - sold_fresh)

            product_revenue = price * DAY1_DISCOUNT * sold_day1 + price * sold_fresh
            product_profit = (
                product_revenue
                - price * PRODUCTION_COST_RATIO * bake_units
                - price * PRODUCTION_COST_RATIO * WASTE_COST_RATIO * expired_day1
                - price * STOCKOUT_COST_RATIO * unmet_demand
            )

            day1_sold[product] = sold_day1
            fresh_sold[product] = sold_fresh
            waste[product] = expired_day1
            shortage[product] = unmet_demand
            next_day1_stock[product] = unsold_fresh
            revenue += product_revenue
            profit += product_profit
            waste_total += expired_day1
            shortage_total += unmet_demand
            sales_total += sold_day1 + sold_fresh
            demand_total += demand_units

        fill_rate = sales_total / demand_total if demand_total else 1.0
        return {
            "profit": round(profit, 2),
            "revenue": round(revenue, 2),
            "waste_units": waste_total,
            "shortage_units": shortage_total,
            "sales_units": sales_total,
            "demand_units": demand_total,
            "fill_rate": round(fill_rate, 3),
            "day1_sold": day1_sold,
            "fresh_sold": fresh_sold,
            "waste": waste,
            "shortage": shortage,
            "next_day1_stock": next_day1_stock,
        }

    def build_q50_baseline_plan(self, day_forecast, capacity=None):
        """Build a simple buffered Q50 policy for paper baseline comparison."""
        if capacity is None:
            capacity = BREAD_CAPACITY

        plan = {}
        for product, forecast in day_forecast.items():
            if product not in self.breads:
                continue
            q50 = forecast.get("q50", 0)
            plan[product] = max(0, int(q50 * DEMAND_BUFFER))

        total_units = sum(plan.values())
        if total_units <= capacity or total_units == 0:
            return plan

        scaled = {}
        remaining_capacity = capacity
        ordered_items = sorted(plan.items(), key=lambda item: item[1], reverse=True)
        for idx, (product, units) in enumerate(ordered_items):
            if idx == len(ordered_items) - 1:
                scaled[product] = max(0, remaining_capacity)
                break
            scaled_units = int(units * capacity / total_units)
            scaled[product] = max(0, scaled_units)
            remaining_capacity -= scaled[product]
        return scaled



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
            result = self.solve_scenarios(stock, q10_demand, q50_demand, q90_demand)
            plans.append({"date": date_str, **result})
            next_stock = {}
            for p in self.breads:
                b = result["bake_plan"].get(p, 0)
                fs = result["fresh_sold"].get(p, 0)
                unsold_fresh = max(0, b - fs)
                next_stock[p] = unsold_fresh
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
            "profit_definition": "after_waste_and_shortage_risk_allowances",
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
        # Fetch real stock from database
        db_stock = {}
        untracked_materials = set()
        stock_data_available = False
        db = None
        try:
            from db.mysql_client import get_db, q
            db = get_db()
            rows = (
                q(db, "raw_materials")
                .select("material_name, stock_quantity, unit, track_inventory")
                .execute()
            )
            if rows.data:
                for r in rows.data:
                    if not bool(r.get("track_inventory", True)):
                        untracked_materials.add(r["material_name"])
                        continue
                    db_stock[r["material_name"]] = {
                        "qty": float(r["stock_quantity"] or 0),
                        "unit": r.get("unit", "kg"),
                    }
                stock_data_available = True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("dashboard_format_materials: DB query failed: %s", e)
        finally:
            if db is not None:
                db.close()

        DEFAULT_WASTE = 0.05
        if not stock_data_available:
            return {
                "week": f"{weekly_summary.get('week_start', '-')} ~ {weekly_summary.get('week_end', '-')}",
                "waste_rate_default": DEFAULT_WASTE,
                "stock_data_available": False,
                "error": "raw_material_stock_unavailable",
                "items": {},
            }

        # Materials now use DB names directly (kg for dry/liquid, pcs for items)
        agg = {}
        for mat_name, weekly_need in weekly_summary.get("materials", {}).items():
            # Skip packaging/secondary that come from beverage estimation
            unit = "kg"
            if mat_name in ("Cup Large", "Cup Regular", "Lids", "Box", "Packaging Bag", "Packaging Box"):
                unit = "pcs"
            elif mat_name == "Milk":
                unit = "L"
            agg[mat_name] = {"weekly_need": float(weekly_need), "unit": unit}

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
            agg.setdefault("Cup Regular", {"weekly_need": 0, "unit": "pcs"})
            agg.setdefault("Cup Large", {"weekly_need": 0, "unit": "pcs"})
            agg.setdefault("Lids", {"weekly_need": 0, "unit": "pcs"})
            agg["Cup Regular"]["weekly_need"] += total_beverage_units * 0.7
            agg["Cup Large"]["weekly_need"] += total_beverage_units * 0.3
            agg["Lids"]["weekly_need"] += total_beverage_units

        procurement = {}
        for db_name, a in agg.items():
            if db_name in untracked_materials:
                continue
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

            waste_amount = round(adjusted - weekly_need_db, 2)
            procurement[db_name] = {
                "weekly_need": weekly_need_db,
                "waste_amount": waste_amount,
                "adjusted_need": adjusted,
                "current_stock": stock_db_units,
                "to_order": to_order,
                "unit": unit,
                "alert": alert,
            }
        return {
            "week": f"{weekly_summary.get('week_start', '-')} ~ {weekly_summary.get('week_end', '-')}",
            "waste_rate_default": DEFAULT_WASTE,
            "stock_data_available": True,
            "items": procurement,
        }


def run_paper_evaluation(save_path=None):
    """
    Paper experiment mode: evaluate S3 scheduler on test set with real lag features.

    Uses the test period defined by the current S2 preprocessing outputs.
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
    id_name = raw[["product_id", "product_name"]].drop_duplicates()
    pid_to_name = dict(zip(id_name["product_id"], id_name["product_name"]))
    # Load S2 models
    if not _init_s2_models():
        print("S2 models not available. Cannot run evaluation.")
        return None

    # Weather means (from test period itself for accurate eval)
    weather_test = test_df.groupby("month")[["temp_mean", "temp_range", "is_cold_day", "is_hot_day"]].mean().to_dict("index")

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
    scheduler_day1_stock = {p: 0 for p in s.breads}
    baseline_day1_stock = {p: 0 for p in s.breads}

    for week_start, week_end in weeks:
        start_str = week_start.strftime("%Y-%m-%d")
        end_str = week_end.strftime("%Y-%m-%d")

        # Build 7-day forecast with real lag features from test data
        forecast = {}
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
            wm = weather_test.get(m, {"temp_mean": 20, "temp_range": 0, "is_cold_day": 0, "is_hot_day": 0})

            for _, row in day_data.iterrows():
                pid = int(row["product_id"])
                name = pid_to_name.get(pid, str(pid))

                # Build feature row with real lag values from test data
                row_df = pd.DataFrame([{
                    "product_id": pid,
                    "category": row.get("category", 1),
                    "daily_tickets": row.get("daily_tickets", 40),
                    "day_of_week": w,
                    "month": m,
                    "is_weekend": 1 if w >= 5 else 0,
                    "is_holiday": row.get("is_holiday", 0),
                    "lag_1": row["lag_1"],
                    "lag_7_avg": row["lag_7_avg"],
                    "lag_30_avg": row["lag_30_avg"],
                    "roll_std_7": row.get("roll_std_7", 0),
                    "roll_std_14": row.get("roll_std_14", 0),
                    "trend_7": row.get("trend_7", 0),
                    "is_day1": row.get("is_day1", 0),
                    "is_top3": row.get("is_top3", 0),
                    "discount_pct": row.get("discount_pct", 0),
                    "is_member_day": row.get("is_member_day", 0),
                    "is_rainy": row.get("is_rainy", 0),
                    "temp_mean": wm["temp_mean"],
                    "temp_range": wm["temp_range"],
                    "is_cold_day": wm["is_cold_day"],
                    "is_hot_day": wm["is_hot_day"],
                    "large_ratio": row.get("large_ratio", 0),
                    "cold_ratio": row.get("cold_ratio", 0),
                    "sweetness_avg": row.get("sweetness_avg", 0),
                    "ice_avg": row.get("ice_avg", 0),
                    "temp_hot_ratio": row.get("temp_hot_ratio", 0),
                }])[_S2_META["feature_cols"]]

                q10_val = max(0, int(round(_S2_MODELS["q10"].predict(row_df)[0])))
                q50_val = max(0, int(round(_S2_MODELS["q50"].predict(row_df)[0])))
                q90_val = max(0, int(round(_S2_MODELS["q90"].predict(row_df)[0])))

                if name not in DRINK_NAMES:  # Only breads for S3
                    day_forecast[name] = {"q10": q10_val, "q50": q50_val, "q90": q90_val}
                day_actual[name] = int(row["quantity"])

                # Track actual vs predicted (breads only)
                if name not in DRINK_NAMES:
                    if name not in all_actual_sales:
                        all_actual_sales[name] = 0
                        all_predicted_sales[name] = 0
                    all_actual_sales[name] += int(row["quantity"])
                    all_predicted_sales[name] += q50_val

            if day_forecast:
                forecast[day_str] = day_forecast
                actual_demand[day_str] = day_actual

        if not forecast:
            continue

        result = s.generate_7day_plan(start_str, scheduler_day1_stock, forecast)

        # Compare with actual (breads only)
        total_actual_bread_sales = 0
        total_planned_bake = result["weekly_summary"]["total_bake"]
        scheduler_actual_profit = 0.0
        scheduler_actual_waste = 0
        scheduler_actual_shortage = 0
        scheduler_actual_sales = 0
        scheduler_actual_demand = 0
        baseline_total_bake = 0
        baseline_actual_profit = 0.0
        baseline_actual_waste = 0
        baseline_actual_shortage = 0
        baseline_actual_sales = 0
        baseline_actual_demand = 0
        week_scheduler_stock = dict(scheduler_day1_stock)
        week_baseline_stock = dict(baseline_day1_stock)
        daily_comparison = []

        for plan in result["plans"]:
            d = plan["date"]
            if d in actual_demand:
                actual_bread_demand = {
                    k: v for k, v in actual_demand[d].items() if k not in DRINK_NAMES
                }
                act = sum(actual_bread_demand.values())
                scheduler_outcome = s.replay_actual_outcome(
                    plan["bake_plan"],
                    week_scheduler_stock,
                    actual_bread_demand,
                )
                baseline_plan = s.build_q50_baseline_plan(forecast.get(d, {}))
                baseline_outcome = s.replay_actual_outcome(
                    baseline_plan,
                    week_baseline_stock,
                    actual_bread_demand,
                )

                daily_comparison.append({
                    "date": d,
                    "planned_bake": plan["total_bake"],
                    "baseline_bake": sum(baseline_plan.values()),
                    "actual_bread_sales": act,
                    "planned_profit": plan["profit"],
                    "actual_replay_profit": scheduler_outcome["profit"],
                    "actual_replay_waste": scheduler_outcome["waste_units"],
                    "actual_replay_shortage": scheduler_outcome["shortage_units"],
                    "actual_replay_fill_rate": scheduler_outcome["fill_rate"],
                    "baseline_profit": baseline_outcome["profit"],
                    "baseline_waste": baseline_outcome["waste_units"],
                    "baseline_shortage": baseline_outcome["shortage_units"],
                    "baseline_fill_rate": baseline_outcome["fill_rate"],
                })
                total_actual_bread_sales += act
                scheduler_actual_profit += scheduler_outcome["profit"]
                scheduler_actual_waste += scheduler_outcome["waste_units"]
                scheduler_actual_shortage += scheduler_outcome["shortage_units"]
                scheduler_actual_sales += scheduler_outcome["sales_units"]
                scheduler_actual_demand += scheduler_outcome["demand_units"]
                baseline_total_bake += sum(baseline_plan.values())
                baseline_actual_profit += baseline_outcome["profit"]
                baseline_actual_waste += baseline_outcome["waste_units"]
                baseline_actual_shortage += baseline_outcome["shortage_units"]
                baseline_actual_sales += baseline_outcome["sales_units"]
                baseline_actual_demand += baseline_outcome["demand_units"]
                week_scheduler_stock = scheduler_outcome["next_day1_stock"]
                week_baseline_stock = baseline_outcome["next_day1_stock"]

        scheduler_fill_rate = (
            scheduler_actual_sales / scheduler_actual_demand
            if scheduler_actual_demand else 1.0
        )
        baseline_fill_rate = (
            baseline_actual_sales / baseline_actual_demand
            if baseline_actual_demand else 1.0
        )

        weekly_results.append({
            "week_start": start_str,
            "week_end": end_str,
            "total_bake": total_planned_bake,
            "baseline_total_bake": baseline_total_bake,
            "total_actual_bread_sales": total_actual_bread_sales,
            "bake_actual_ratio": round(total_planned_bake / max(total_actual_bread_sales, 1), 3),
            "profit": round(scheduler_actual_profit, 2),
            "planned_profit": result["weekly_summary"]["total_profit"],
            "baseline_profit": round(baseline_actual_profit, 2),
            "profit_lift_vs_baseline": round(scheduler_actual_profit - baseline_actual_profit, 2),
            "waste": result["weekly_summary"]["total_waste"],
            "shortage": result["weekly_summary"]["total_shortage"],
            "actual_replay_waste": scheduler_actual_waste,
            "actual_replay_shortage": scheduler_actual_shortage,
            "actual_replay_fill_rate": round(scheduler_fill_rate, 3),
            "baseline_waste": baseline_actual_waste,
            "baseline_shortage": baseline_actual_shortage,
            "baseline_fill_rate": round(baseline_fill_rate, 3),
            "daily": daily_comparison,
        })

        scheduler_day1_stock = week_scheduler_stock
        baseline_day1_stock = week_baseline_stock

    # Aggregate evaluation metrics
    total_bake = sum(w["total_bake"] for w in weekly_results)
    baseline_total_bake = sum(w["baseline_total_bake"] for w in weekly_results)
    total_actual = sum(w["total_actual_bread_sales"] for w in weekly_results)
    total_profit = sum(w["profit"] for w in weekly_results)
    total_planned_profit = sum(w["planned_profit"] for w in weekly_results)
    baseline_profit = sum(w["baseline_profit"] for w in weekly_results)
    actual_waste = sum(w["actual_replay_waste"] for w in weekly_results)
    actual_shortage = sum(w["actual_replay_shortage"] for w in weekly_results)
    baseline_waste = sum(w["baseline_waste"] for w in weekly_results)
    baseline_shortage = sum(w["baseline_shortage"] for w in weekly_results)
    actual_sales_served = max(total_actual - actual_shortage, 0)
    baseline_sales_served = max(total_actual - baseline_shortage, 0)

    eval_report = {
        "evaluation_period": f"{test_dates[0].date()} to {test_dates[-1].date()}",
        "total_weeks": len(weekly_results),
        "aggregate": {
            "total_bake_planned": total_bake,
            "baseline_total_bake": baseline_total_bake,
            "total_actual_bread_sales": total_actual,
            "bake_to_actual_ratio": round(total_bake / max(total_actual, 1), 3),
            "total_profit_cny": round(total_profit, 2),
            "total_planned_profit_cny": round(total_planned_profit, 2),
            "baseline_profit_cny": round(baseline_profit, 2),
            "profit_lift_vs_baseline_cny": round(total_profit - baseline_profit, 2),
            "actual_replay_waste_units": actual_waste,
            "actual_replay_shortage_units": actual_shortage,
            "actual_replay_fill_rate": round(actual_sales_served / max(total_actual, 1), 3),
            "baseline_waste_units": baseline_waste,
            "baseline_shortage_units": baseline_shortage,
            "baseline_fill_rate": round(baseline_sales_served / max(total_actual, 1), 3),
            "baseline_policy": "buffered_q50_capacity_scaled",
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
    print(f"  Baseline profit:     CNY {agg['baseline_profit_cny']:>10,.2f}")
    print(f"  Profit lift:         CNY {agg['profit_lift_vs_baseline_cny']:>10,.2f}")
    print(f"  Actual replay waste: {agg['actual_replay_waste_units']:>6d} units")
    print(f"  Actual replay shortage:{agg['actual_replay_shortage_units']:>4d} units")
    print(f"  Actual fill rate:    {agg['actual_replay_fill_rate']:>6.3f}")
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
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2, default=str)
        print(f"\n  Saved to {save_path}")

    return eval_report



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3 CP-SAT Production Scheduler")
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

        day1_stock = {p: 0 for p in s.breads}
        forecast = generate_7day_s2_forecast(start_date)
        print("Using real S2 quantile models for demand forecast")

        result = s.generate_7day_plan(start_date, day1_stock, forecast)
        ws = result["weekly_summary"]

        print("=" * 60)
        print(f"  7-Day Rolling Plan: {start_date} ~ {result['dashboard_7day']['week_end']}")
        print(f"  Buffer applied: {DEMAND_BUFFER}x demand")
        print("=" * 60)
        print("\n  Weekly Summary:")
        print(f"    Total bake:    {ws['total_bake']:>5d} units")
        print(f"    Total profit:  CNY {ws['total_profit']:>10,.2f}")
        print(f"    Total revenue: CNY {ws['total_revenue']:>10,.2f}")
        print(f"    Total waste:   {ws['total_waste']:>5d} units")
        print(f"    Total shortage:{ws['total_shortage']:>5d} units")

        print("\n  Scenario Analysis:")
        for scen in SCENARIO_LABELS:
            sc = ws["scenarios"][scen]
            print(f"    {scen.upper()}: profit CNY {sc['profit']:>10,.2f}  waste {sc['waste']:4d}  shortage {sc['shortage']:4d}")

        print("\n  Daily Breakdown:")
        for plan in result["plans"]:
            p = plan
            print(f"    {p['date']}: bake {p['total_bake']:3d} ({p['capacity_used_pct']:5.1f}%)  profit CNY {p['profit']:>8,.2f}")

        print("\n  Top 10 Products (weekly bake):")
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
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)
            print(f"\n  Saved to {args.save}")

    else:
        parser.error("select either --7day or --eval")

