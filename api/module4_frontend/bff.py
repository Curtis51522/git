from fastapi import APIRouter, HTTPException, Depends, Query
from passlib.context import CryptContext
from datetime import datetime, timedelta
from decimal import Decimal
import hmac
import json
import logging
import sys, os

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings as cfg
from db.mysql_client import get_db, q
from api.auth import create_access_token, get_current_user, require_manager
from models.schemas import LoginRequest, LoginResponse, is_positive_integer
from api.module4_frontend.beverage_options import (
    beverage_unit_price,
    bundle_price_values,
    discounted_unit_values,
    is_beverage,
    list_beverage_capabilities,
    normalize_beverage_item,
    round_pos_money,
)

router = APIRouter(prefix="/s4", tags=["Module 4 - BFF"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
logger = logging.getLogger("s4.bff")

COFFEE_BREAD_PAIRS = {
    "Latte": ["Croissant","Danish"],
    "Americano": ["Muffin","Donut"],
    "Cappuccino": ["Cinnamon Roll","Sourdough"],
    "Cold Brew": ["Bagel","Croissant"],
    "Espresso": ["Baguette"],
    "Flat White": ["Croissant","Muffin"],
    "Mocha": ["Donut","Cinnamon Roll"],
}


def _discount_rate(value):
    try:
        return min(max(float(value), 0.0), 0.5)
    except (TypeError, ValueError):
        return 0.0


def _resolve_checkout_discount(item, allowed_dynamic, freshness_rate):
    requested_rate = _discount_rate(item.get("discount_rate"))
    allowed_rate = _discount_rate((allowed_dynamic or {}).get("discount_pct", 0) / 100)
    dynamic_rate = min(requested_rate, allowed_rate)
    freshness_rate = _discount_rate(freshness_rate)
    if dynamic_rate > freshness_rate:
        return {
            "rate": dynamic_rate,
            "source": str((allowed_dynamic or {}).get("source") or "s5_dynamic"),
            "strategy": str((allowed_dynamic or {}).get("strategy") or ""),
            "reason": str((allowed_dynamic or {}).get("reason") or "Validated S5 discount"),
        }
    if freshness_rate > 0:
        return {
            "rate": freshness_rate,
            "source": "freshness",
            "strategy": "clearance",
            "reason": "Freshness-based discount",
        }
    return {"rate": 0.0, "source": "none", "strategy": "", "reason": ""}


async def _fetch_validated_dynamic_discounts(items):
    products = list(dict.fromkeys(
        item.get("product_name", "")
        for item in items
        if item.get("product_name") and _discount_rate(item.get("discount_rate")) > 0
    ))
    if not products:
        return {}
    try:
        async with httpx.AsyncClient(timeout=cfg.S5_DISCOUNT_TIMEOUT_SECONDS) as client:
            response = await client.post(cfg.S5_DISCOUNT_URL, json={"products": products})
            response.raise_for_status()
            return response.json().get("discounts", {})
    except Exception as exc:
        logger.warning("Dynamic discount validation failed: %s", exc)
        raise HTTPException(503, "Dynamic discount validation unavailable") from exc


# ======================================================================
# Auth helpers
# ======================================================================
@router.post("/login")
async def login(req: LoginRequest):
    db = get_db()
    try:
        r = q(db, "users").select("*").eq("username", req.username).execute()
        if not r.data:
            raise HTTPException(401, "Invalid credentials")
        user = r.data[0]
        stored_hash = str(user.get("password_hash", "") or "")
        if pwd_context.identify(stored_hash) is None:
            if (
                not cfg.ALLOW_LEGACY_PLAINTEXT_LOGIN
                or not stored_hash
                or not hmac.compare_digest(req.password, stored_hash)
            ):
                raise HTTPException(401, "Invalid credentials")
            q(db, "users").update(
                {"password_hash": pwd_context.hash(req.password)}
            ).eq("username", req.username).execute()
        elif not pwd_context.verify(req.password, stored_hash):
            raise HTTPException(401, "Invalid credentials")
        token = create_access_token(user["username"], user["role"])
        return LoginResponse(
            access_token=token,
            username=user["username"],
            role=user["role"],
        )
    finally:
        db.close()


# ======================================================================
# POST /s4/combo -- Product pairing recommendations
# ======================================================================
@router.post("/combo")
async def get_combo(order: dict, _: dict = Depends(get_current_user)):
    """5-dimension bundle recommendation scoring.
    
    Weights (configurable):
    - Flavor Pairing (25%): Bread-coffee affinity matrix
    - Discount Value (20%): Higher discount = better deal for customer
    - Freshness (20%): Day-1/2 items need promotion
    - Inventory Pressure (20%): High stock = push harder
    - Order Context (15%): Complement what's already in cart
    """
    # Weights (sum = 100)
    W_FLAVOR   = 0.25
    W_DISCOUNT = 0.20
    W_FRESH    = 0.20
    W_INV      = 0.20
    W_CONTEXT  = 0.15

    from api.freshness_service import get_discount_rate, update_all_freshness
    
    # Auto-update freshness before scoring
    update_all_freshness()
    
    db = get_db()
    
    # Get all sellable bakery items (not coffee)

    # Get all sellable bakery items dynamically from DB
    cur_bakery = db.cursor()
    cur_bakery.execute("SELECT product_name FROM products WHERE category='bakery'")
    BAKERY_PRODUCTS = {r[0] for r in cur_bakery.fetchall()}
    cur_bakery.close()

    
    r = q(db, "batch_inventory").select("*").gt("quantity", 0).neq("freshness_status", "Expired").execute()
    bakery_batches = [b for b in (r.data or []) if b.get("product_name","") in BAKERY_PRODUCTS]
    
    if not bakery_batches:
        return {"status": "ok", "recommendations": []}
    
    # Aggregate inventory by product
    from collections import defaultdict
    inventory = defaultdict(lambda: {"total_qty": 0, "batches": [], "min_freshness": "Fresh"})
    for b in bakery_batches:
        pn = b["product_name"]
        qty = b.get("quantity", 0)
        inventory[pn]["total_qty"] += qty
        inventory[pn]["batches"].append(b)
        f = b.get("freshness_status", "Fresh")
        # Track "worst" freshness (for discount scoring)
        f_rank = {"Fresh": 0, "Day-1": 1, "Expired": 2}
        if f_rank.get(f, 0) > f_rank.get(inventory[pn]["min_freshness"], 0):
            inventory[pn]["min_freshness"] = f
        # Per-product freshness breakdown
        if "fresh_qty" not in inventory[pn]:
            inventory[pn]["fresh_qty"] = 0
            inventory[pn]["day1_qty"] = 0
        if f == "Day-1":
            inventory[pn]["day1_qty"] += qty
        else:
            inventory[pn]["fresh_qty"] += qty
    
    # Bread-coffee affinity matrix (flavor pairing scores 0-1)
    # LLM-generated bread-coffee affinity matrix (cached, DeepSeek-powered)
    from api.module4_frontend.pairing_llm import get_pairing_matrix
    PAIRING_MATRIX = get_pairing_matrix()
    
    # Get all beverages dynamically from DB
    cur_coffee = db.cursor()
    cur_coffee.execute("SELECT product_name, selling_price FROM products WHERE category='beverage' ORDER BY product_name")
    COFFEE_DRINKS = [
        {"name": r[0].replace('_', ' ').title(), "key": r[0], "price": float(r[1])}
        for r in cur_coffee.fetchall()
    ]
    cur_coffee.close()
    
    # Cart context: which breads does the customer already have?
    order_items = order.get("items", [])
    business_events = order.get("business_events", [])
    cart_breads = set()
    cart_coffee_keys = set()
    for item in order_items:
        pn = item.get("product_name", "")
        if pn in BAKERY_PRODUCTS:
            cart_breads.add(pn)
        for c in COFFEE_DRINKS:
            if pn == c["key"] or pn == c["name"]:
                cart_coffee_keys.add(c["key"])

    # Determine scoring direction:
    # Cart has bread -> recommend coffee pairings (bread->coffee)
    # Cart has coffee -> recommend bread pairings (coffee->bread)  
    # Cart has both -> both directions
    all_scores = []
    max_inv = max(inv["total_qty"] for inv in inventory.values()) if inventory else 1

    # Strict mode: when cart has both bread and coffee, only pair cart items
    strict_mode = False  # Always consider full inventory; cart items boosted via cart_boost

    # --- Direction 1: Bread -> Coffee (cart has bread or empty) ---
    if cart_breads or not cart_coffee_keys:
        target_breads = cart_breads if cart_breads else set(inventory.keys())
        for pn, inv_data in inventory.items():
            if pn not in target_breads:
                continue
            pairings = PAIRING_MATRIX.get(pn, {})
            freshness = inv_data.get("min_freshness", "Fresh")
            # Use override discount if provided, otherwise fall back to freshness-based
            discount_overrides = order.get("discount_overrides", {})
            discount = (discount_overrides.get(pn, 0) / 100.0) if pn in discount_overrides else get_discount_rate(freshness)
            inv_pressure = inv_data["total_qty"] / max(max_inv, 1)
            target_coffees = [c for c in COFFEE_DRINKS if not strict_mode or c["key"] in cart_coffee_keys]
            for coffee in target_coffees:
                ck = coffee["key"]
                flavor_score = pairings.get(ck, 0.3)
                discount_score = discount * 3.33
                f_map = {"Fresh": 0.3, "Day-1": 0.8, "Expired": 1.0}
                freshness_score = f_map.get(freshness, 0.5)
                inv_score = min(inv_pressure, 1.0)
                context_score = 1.0 if ck not in cart_coffee_keys else 0.5
                cart_boost = (0.20 if pn in cart_breads else 0.0) + (0.20 if ck in cart_coffee_keys else 0.0) + (0.10 if (pn in cart_breads and ck in cart_coffee_keys) else 0.0)
                total = (W_FLAVOR*flavor_score + W_DISCOUNT*discount_score + W_FRESH*freshness_score + W_INV*inv_score + W_CONTEXT*context_score + cart_boost)
                pricing = bundle_price_values(get_product_prices().get(pn, 5.0), discount, coffee["price"], "regular")
                bundle_price = float(pricing["total"])
                savings = float(pricing["savings"])
                all_scores.append({
                    "product_name": pn, "coffee_name": coffee["name"], "coffee_key": ck,
                    "products": f"{pn.replace('_',' ').title()} + {coffee['name']}",
                    "direction": "bread_to_coffee",
                    "total_score": round(total, 3), "total_price": round(bundle_price, 2),
                    "savings": round(savings, 2), "freshness_status": freshness,
                    "stock_qty": inv_data["total_qty"],
                    "scoring_breakdown": {
                        "flavor_pairing": round(W_FLAVOR*flavor_score, 3),
                        "discount": round(W_DISCOUNT*discount_score, 3),
                        "freshness": round(W_FRESH*freshness_score, 3),
                        "inventory": round(W_INV*inv_score, 3),
                        "context": round(W_CONTEXT*context_score, 3),
                        "cart_boost": round(cart_boost, 3),
                        "bread": pn,
                        "coffee": coffee["key"],
                    },
                })

    # --- Direction 2: Coffee -> Bread (cart has coffee) ---
    if cart_coffee_keys:
        for pn, inv_data in inventory.items():
            if strict_mode and pn not in cart_breads:
                continue
            pairings = PAIRING_MATRIX.get(pn, {})
            freshness = inv_data.get("min_freshness", "Fresh")
            # Use override discount if provided, otherwise fall back to freshness-based
            discount_overrides = order.get("discount_overrides", {})
            discount = (discount_overrides.get(pn, 0) / 100.0) if pn in discount_overrides else get_discount_rate(freshness)
            inv_pressure = inv_data["total_qty"] / max(max_inv, 1)
            discount_score = discount * 3.33
            f_map = {"Fresh": 0.3, "Day-1": 0.8, "Expired": 1.0}
            freshness_score = f_map.get(freshness, 0.5)
            inv_score = min(inv_pressure, 1.0)
            for coffee in COFFEE_DRINKS:
                ck = coffee["key"]
                if ck not in cart_coffee_keys:
                    continue
                flavor_score = pairings.get(ck, 0.3)
                # Bread NOT in cart = higher context (new recommendation)
                context_score = 1.0 if pn not in cart_breads else 0.5
                coffee_boost = (0.20 if ck in cart_coffee_keys else 0.0) + (0.20 if pn in cart_breads else 0.0) + (0.10 if (pn in cart_breads and ck in cart_coffee_keys) else 0.0)
                total = (W_FLAVOR*flavor_score + W_DISCOUNT*discount_score + W_FRESH*freshness_score + W_INV*inv_score + W_CONTEXT*context_score + coffee_boost)
                pricing = bundle_price_values(get_product_prices().get(pn, 5.0), discount, coffee["price"], "regular")
                bundle_price = float(pricing["total"])
                savings = float(pricing["savings"])
                all_scores.append({
                    "product_name": pn, "coffee_name": coffee["name"], "coffee_key": ck,
                    "products": f"{pn.replace('_',' ').title()} + {coffee['name']}",
                    "direction": "coffee_to_bread",
                    "total_score": round(total, 3), "total_price": round(bundle_price, 2),
                    "savings": round(savings, 2), "freshness_status": freshness,
                    "stock_qty": inv_data["total_qty"],
                    "scoring_breakdown": {
                        "flavor_pairing": round(W_FLAVOR*flavor_score, 3),
                        "discount": round(W_DISCOUNT*discount_score, 3),
                        "freshness": round(W_FRESH*freshness_score, 3),
                        "inventory": round(W_INV*inv_score, 3),
                        "context": round(W_CONTEXT*context_score, 3),
                        "coffee_boost": round(coffee_boost, 3),
                        "bread": pn,
                        "coffee": coffee["key"],
                    },
                })

    # Priority boost from RecommendationAgent (additive, proportional to strategy weight)
    priority_products = order.get("priority_products", [])
    if priority_products:
        BOOST_ADDITIVE = {2.5: 0.15, 2.0: 0.12, 1.8: 0.10, 1.5: 0.08}
        for s in all_scores:
            for pp in priority_products:
                pp_product = pp.get("product", "").lower().replace(" ", "_")
                pp_coffee = pp.get("coffee", "").lower().replace(" ", "_")
                boost = float(pp.get("boost", 1.5))
                add_bonus = BOOST_ADDITIVE.get(boost, 0.05)
                if s["product_name"] == pp_product and s["coffee_key"] == pp_coffee:
                    s["total_score"] = min(round(s["total_score"] + add_bonus, 3), 1.0)
                    s["priority_boost"] = add_bonus
                    break

    def business_event_context_for(product_name):
        for event in business_events:
            if not event or not event.get("active", True):
                continue
            products = event.get("products") or []
            if product_name not in products:
                continue
            return {
                "id": event.get("id"),
                "event_type": event.get("event_type"),
                "label": event.get("label"),
                "discount_pct": event.get("discount_pct"),
                "start_date": event.get("start_date"),
                "end_date": event.get("end_date"),
            }
        return None

    # Sort by score descending
    # Sort by savings when cart has only coffee (different bread discounts matter), otherwise by total_score
    if cart_coffee_keys and not cart_breads:
        all_scores.sort(key=lambda x: x["savings"], reverse=True)
    else:
        all_scores.sort(key=lambda x: x["total_score"], reverse=True)

    # Pick top-3: Top-1 from cart breads (relevance), rest from full inventory (diversity + clearance)
    top3 = []
    seen_products = set()

    # Pass 1a: pick best cart bread first (guarantees customer-relevant Top-1)
    if cart_breads:
        for s in all_scores:
            if s["product_name"] in cart_breads and s["product_name"] not in seen_products:
                top3.append(s)
                seen_products.add(s["product_name"])
                break

    # Pass 1b: pick diverse from remaining (all inventory, including non-cart)
    for s in all_scores:
        if s["product_name"] not in seen_products:
            top3.append(s)
            seen_products.add(s["product_name"])

    # Pass 2: if still under 3, fill with any (allows same bread, different coffee)
    if len(top3) < 3:
        for s in all_scores:
            if s not in top3:
                top3.append(s)
            if len(top3) >= 3:
                break

    top3 = top3[:3]
    for s in top3:
        event_context = business_event_context_for(s["product_name"])
        if event_context:
            s["business_event_context"] = event_context

    freshness_breakdown = {}
    for pn, inv_data in inventory.items():
        total = inv_data.get("total_qty", 1)
        day1 = inv_data.get("day1_qty", 0)
        freshness_breakdown[pn] = {
            "total_qty": total,
            "day1_qty": day1,
            "fresh_qty": inv_data.get("fresh_qty", 0),
            "day1_ratio": round(day1 / max(total, 1), 2),
        }
    return {"status": "ok", "recommendations": top3, "freshness": freshness_breakdown, "weights": {
        "flavor_pairing": int(W_FLAVOR*100),
        "discount_value": int(W_DISCOUNT*100),
        "freshness": int(W_FRESH*100),
        "inventory_pressure": int(W_INV*100),
        "order_context": int(W_CONTEXT*100),
    }}

# Product prices - read from DB (single source of truth)
_product_prices_cache = None

# Default Malaysian bakery prices (fallback when DB not available)
_DEFAULT_PRICES = {
    "donut": 6.50, "croissant": 7.50, "bread_coconut": 5.50,
    "bread_roll": 5.00, "chiffon": 8.00, "croissant_chocolate": 8.50
}

def get_product_prices():
    """Return {product_name: unit_price} dict, cached after first successful DB read."""
    global _product_prices_cache
    db = None
    if _product_prices_cache is not None and len(_product_prices_cache) > 0:
        return _product_prices_cache
    try:
        db = get_db()
        r = q(db, "products").select("*").execute()
        if r.data and len(r.data) > 0:
            _product_prices_cache = {}
            for row in r.data:
                _product_prices_cache[row["product_name"]] = float(row.get("selling_price", row.get("unit_price", 0)))
            return _product_prices_cache
    except Exception:
        pass
    finally:
        if db is not None and hasattr(db, "close"):
            db.close()
    # DB not ready or empty -- use defaults (retry DB on next call)
    _product_prices_cache = None
    return dict(_DEFAULT_PRICES)

_product_costs_cache = None

_DEFAULT_COSTS = {
    "donut": 2.00, "croissant": 2.50, "bread_coconut": 1.80,
    "bread_roll": 1.50, "chiffon": 2.50, "croissant_chocolate": 2.80
}

def get_product_costs():
    """Return {product_name: cost_price} dict, cached after first successful DB read."""
    global _product_costs_cache
    db = None
    if _product_costs_cache is not None and len(_product_costs_cache) > 0:
        return _product_costs_cache
    try:
        db = get_db()
        r = q(db, "products").select("*").execute()
        if r.data and len(r.data) > 0:
            _product_costs_cache = {}
            for row in r.data:
                _product_costs_cache[row["product_name"]] = float(row.get("cost_price", 0))
            return _product_costs_cache
    except Exception:
        pass
    finally:
        if db is not None and hasattr(db, "close"):
            db.close()
    _product_costs_cache = None
    return dict(_DEFAULT_COSTS)

# Use get_product_prices() directly; this module-level reference is kept for backward compat
# but will only be populated after first successful DB read
PRODUCT_PRICES = {}





# ======================================================================

# ======================================================================
# GET /s4/beverages/options -- Return beverage customization capabilities
# ======================================================================
@router.get("/beverages/options", dependencies=[Depends(get_current_user)])
async def get_beverage_options():
    return {
        "status": "ok",
        "beverages": list_beverage_capabilities(),
    }

# ======================================================================
# GET /s4/products -- Return product prices from DB
# ======================================================================
@router.get("/products", dependencies=[Depends(get_current_user)])
async def list_products():
    """Return all product prices from the database."""
    global _product_prices_cache
    db = None
    try:
        db = get_db()
        r = q(db, "products").select("*").execute()
        if r.data:
            products = []
            refreshed_prices = {}
            for row in r.data:
                product_price = float(row.get("selling_price", row.get("unit_price", 0)))
                refreshed_prices[row["product_name"]] = product_price
                products.append({
                    "product_name": row["product_name"],
                    "unit_price": product_price,
                    "cost_price": float(row.get("material_cost", row.get("cost_price", 0))),
                })
            _product_prices_cache = refreshed_prices
            return {"status": "ok", "products": products}
    except Exception:
        pass
    finally:
        if db is not None:
            db.close()
    # Fallback: return every cached product price.
    prices = get_product_prices()
    costs = get_product_costs()
    products = []
    for name, price in prices.items():
        products.append({
            "product_name": name,
            "unit_price": float(price) if price else 0,
            "cost_price": float(costs.get(name, 0)),
        })
    return {"status": "ok", "products": products}


def _load_checkout_products(cur, items):
    product_names = list(dict.fromkeys(item["product_name"] for item in items))
    placeholders = ",".join(["%s"] * len(product_names))
    cur.execute(
        f"""
        SELECT product_name, selling_price, material_cost, wastage_pct
        FROM products
        WHERE product_name IN ({placeholders})
        ORDER BY product_name
        FOR UPDATE
        """,
        product_names,
    )
    products = {}
    for product_name, selling_price, material_cost, wastage_pct in cur.fetchall():
        if product_name in products:
            raise HTTPException(409, f"Duplicate canonical product row: {product_name}")
        if selling_price is None:
            continue
        price = Decimal(str(selling_price))
        if price <= 0:
            raise HTTPException(409, f"Invalid canonical price for: {product_name}")
        products[product_name] = {
            "selling_price": price,
            "material_cost": Decimal(str(material_cost or 0)),
            "wastage_pct": Decimal(
                str(0.03 if wastage_pct is None else wastage_pct)
            ),
        }

    missing = sorted(set(product_names) - set(products))
    if missing:
        raise HTTPException(409, f"Missing canonical price for: {', '.join(missing)}")
    return products


def _price_checkout_items(items, resolved_discounts, products):
    priced_items = []
    subtotal = Decimal("0.0")
    discount_total = Decimal("0.0")
    for item_index, item in enumerate(items):
        product_name = item["product_name"]
        quantity = item["quantity"]
        base_price = products[product_name]["selling_price"]
        if is_beverage(product_name):
            base_price = beverage_unit_price(base_price, item["size"])
        resolved_discount = resolved_discounts[item_index]
        discount_rate = (
            resolved_discount["rate"] if not is_beverage(product_name) else 0.0
        )
        unit_values = discounted_unit_values(base_price, discount_rate)
        quantity_decimal = Decimal(quantity)
        priced_item = {
            "item": item,
            "quantity": quantity,
            "quantity_decimal": quantity_decimal,
            "unit_price": unit_values["unit_price"],
            "discount_rate": discount_rate,
            "line_subtotal": unit_values["unit_price"] * quantity_decimal,
            "line_discount": unit_values["unit_discount"] * quantity_decimal,
            "line_final": unit_values["discounted_unit_price"] * quantity_decimal,
            "resolved_discount": resolved_discount,
        }
        priced_items.append(priced_item)
        subtotal += priced_item["line_subtotal"]
        discount_total += priced_item["line_discount"]
    return priced_items, subtotal, discount_total


def _build_material_requirements(cur, items, products, dine_type):
    quantities_by_product = {}
    for item in items:
        if not is_beverage(item["product_name"]):
            continue
        product_name = item["product_name"]
        quantities_by_product[product_name] = (
            quantities_by_product.get(product_name, 0) + item["quantity"]
        )

    requirements = {}
    recipe_products = set()
    product_names = sorted(quantities_by_product)
    if product_names:
        placeholders = ",".join(["%s"] * len(product_names))
        cur.execute(
            f"""
            SELECT pr.product_name, pr.material_name, pr.quantity_per_unit,
                   rm.unit, rm.category
            FROM product_recipes pr
            LEFT JOIN raw_materials rm ON rm.material_name = pr.material_name
            WHERE pr.product_name IN ({placeholders})
            ORDER BY pr.product_name, pr.material_name
            """,
            product_names,
        )
        for product_name, material_name, per_unit, unit, category in cur.fetchall():
            if product_name not in quantities_by_product or not material_name:
                raise HTTPException(409, "Invalid product recipe row")
            if not unit:
                raise HTTPException(409, f"Missing raw material unit for {material_name}")
            try:
                quantity_per_unit = Decimal(str(per_unit))
            except (ArithmeticError, TypeError, ValueError) as exc:
                raise HTTPException(
                    409,
                    f"Invalid recipe quantity for {product_name}: {per_unit}",
                ) from exc
            if not quantity_per_unit.is_finite() or quantity_per_unit <= 0:
                raise HTTPException(
                    409,
                    f"Invalid recipe quantity for {product_name}: {per_unit}",
                )
            recipe_products.add(product_name)
            required = quantity_per_unit * Decimal(
                quantities_by_product[product_name]
            )
            if unit != "pcs" and (category or "") != "packaging":
                required *= Decimal("1") + products[product_name]["wastage_pct"]
            required = required.quantize(Decimal("0.000001"))
            requirements[material_name] = requirements.get(
                material_name, Decimal("0")
            ) + required

        missing_recipes = sorted(set(product_names) - recipe_products)
        if missing_recipes:
            raise HTTPException(
                409,
                f"Missing product recipe: {', '.join(missing_recipes)}",
            )

    for item in items:
        if not is_beverage(item["product_name"]):
            continue
        cup_name = "Cup Large" if item["size"] == "large" else "Cup Regular"
        requirements[cup_name] = requirements.get(
            cup_name, Decimal("0")
        ) + Decimal(item["quantity"])

    if dine_type == "takeaway":
        for material_name in ("Packaging Bag", "Packaging Box"):
            requirements[material_name] = requirements.get(
                material_name, Decimal("0")
            ) + Decimal("1")
    return requirements


def _lock_and_validate_materials(cur, requirements):
    if not requirements:
        return {}
    material_names = sorted(requirements)
    placeholders = ",".join(["%s"] * len(material_names))
    cur.execute(
        f"""
        SELECT material_name, stock_quantity, unit, unit_price
        FROM raw_materials
        WHERE material_name IN ({placeholders})
        ORDER BY material_name
        FOR UPDATE
        """,
        material_names,
    )
    rows = {
        material_name: (
            Decimal(str(stock_quantity)),
            unit,
            Decimal(str(unit_price or 0)),
        )
        for material_name, stock_quantity, unit, unit_price in cur.fetchall()
    }
    missing = sorted(set(material_names) - set(rows))
    if missing:
        raise HTTPException(409, f"Missing material stock: {', '.join(missing)}")

    shortages = []
    for material_name in material_names:
        available = rows[material_name][0]
        required = requirements[material_name]
        if available < required:
            shortages.append(
                f"{material_name} (need {required}, available {available})"
            )
    if shortages:
        raise HTTPException(409, f"Insufficient material stock: {'; '.join(shortages)}")
    return {
        name: {"unit": values[1], "unit_price": values[2]}
        for name, values in rows.items()
    }


def _packaging_material_cost(requirements, material_info):
    return sum(
        (
            requirements.get(material_name, Decimal("0"))
            * material_info.get(material_name, {}).get("unit_price", Decimal("0"))
        )
        for material_name in ("Packaging Bag", "Packaging Box")
    )

# POST /s4/checkout/complete -- Complete payment + deduct inventory
# ======================================================================
@router.post("/checkout/complete", dependencies=[Depends(get_current_user)])
async def checkout_complete(payload: dict):
    """Process checkout: deduct inventory via FIFO, apply freshness discounts, generate receipt."""
    items = payload.get("items", [])
    if not items:
        raise HTTPException(400, "No items in cart")

    normalized_items = []
    for raw_item in items:
        if not isinstance(raw_item, dict):
            raise HTTPException(400, "Each checkout item must be an object")
        item = dict(raw_item)
        product_name = item.get("product_name", "")
        if not is_positive_integer(item.get("quantity")):
            raise HTTPException(
                400,
                f"Invalid quantity for '{product_name}': expected a positive integer",
            )
        if is_beverage(product_name):
            try:
                item.update(normalize_beverage_item(product_name, item))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        normalized_items.append(item)
    items = normalized_items

    from api.module1_yolo import deduct_inventory
    from models.schemas import DeductRequest

    # Split items: bakery (deduct from inventory) vs coffee (no inventory limit)
    bakery_items = []
    coffee_items = []
    BAKERY_KEYS = {"apple_pie","bagel","baguette","bread_coconut","bread_roll","brioche","brownie","chiffon","chocolate_cake","chocopie","cookie","cornbread","cream_horn","croissant","croissant_chocolate","donut","eggtart","flatbread","macaron","mantequilla","melon_bread","muffin","pancake","pandesal","pizza_bread","pullman","soboru_bread","sourdough","stickbread","tostada"}
    unknown_items = []
    for item in items:
        pn = item.get("product_name", "")
        if pn in BAKERY_KEYS:
            bakery_items.append(item)
        elif is_beverage(pn):
            coffee_items.append(item)
        else:
            unknown_items.append(pn)

    if unknown_items:
        return {
            "status": "error",
            "deducted": [],
            "errors": [f"Unknown product(s): {', '.join(unknown_items)}"],
            "receipt": None,
            "message": f"Checkout rejected: unknown products {unknown_items}",
        }

    for item in bakery_items:
        freshness = item.get("freshness")
        if freshness not in {"Fresh", "Day-1"}:
            raise HTTPException(
                400,
                f"Invalid bakery freshness for '{item.get('product_name', '')}': {freshness}",
            )

    from api.freshness_service import get_discount_rate

    validated_dynamic_discounts = await _fetch_validated_dynamic_discounts(
        [item for item in items if not is_beverage(item.get("product_name", ""))]
    )
    resolved_discounts = []
    for item in items:
        product_name = item.get("product_name", "")
        if is_beverage(product_name):
            resolved_discounts.append({"rate": 0.0, "source": "none", "strategy": "", "reason": ""})
            continue
        freshness = item.get("freshness", "Fresh")
        freshness_rate = get_discount_rate(freshness) if freshness == "Day-1" else 0.0
        resolved_discounts.append(_resolve_checkout_discount(
            item=item,
            allowed_dynamic=validated_dynamic_discounts.get(product_name, {}),
            freshness_rate=freshness_rate,
        ))

    priced_items = []
    receipt_items = []
    subtotal = Decimal("0.0")
    discount_total = Decimal("0.0")
    total = Decimal("0.0")
    savings = Decimal("0.0")
    product_data = {}
    material_requirements = {}
    material_info = {}
    
    deducted = []
    all_errors = []
    status = "ok"
    db = None
    cur = None

    try:
        db = get_db(autocommit=False)
        cur = db.cursor()
        product_data = _load_checkout_products(cur, items)
        priced_items, subtotal, discount_total = _price_checkout_items(
            items,
            resolved_discounts,
            product_data,
        )
        priced_by_item_id = {
            id(priced_item["item"]): priced_item for priced_item in priced_items
        }

        for priced_item in priced_items:
            item = priced_item["item"]
            resolved_discount = priced_item["resolved_discount"]
            is_beverage_item = is_beverage(item["product_name"])
            receipt_items.append({
                "product_name": item["product_name"],
                "quantity": priced_item["quantity"],
                "unit_price": float(priced_item["unit_price"]),
                "discount_pct": int(priced_item["discount_rate"] * 100),
                "discount_amount": float(priced_item["line_discount"]),
                "line_total": float(priced_item["line_final"]),
                "discount_source": resolved_discount["source"],
                "discount_strategy": resolved_discount["strategy"],
                "discount_reason": resolved_discount["reason"],
                "size": item.get("size") if is_beverage_item else None,
                "temperature": item.get("temperature") if is_beverage_item else None,
                "sugar": item.get("sugar") if is_beverage_item else None,
                "ice_level": item.get("ice_level") if is_beverage_item else None,
            })

        total = round_pos_money(subtotal - discount_total)
        savings = discount_total
        dine_type = payload.get("dine_type", "dine_in")
        packaging_fee = (
            Decimal("0.3") if dine_type == "takeaway" else Decimal("0.0")
        )
        if packaging_fee > 0:
            total = round_pos_money(total + packaging_fee)
        material_requirements = _build_material_requirements(
            cur,
            items,
            product_data,
            dine_type,
        )
        material_info = _lock_and_validate_materials(
            cur,
            material_requirements,
        )
        now = datetime.now()
        payment_method = payload.get("payment_method", "cash")
        receipt_id = (
            f"RCP-{now.strftime('%Y%m%d%H%M%S')}-"
            f"{now.microsecond // 1000:03d}"
        )

        # Deduct bakery items via FIFO on the checkout transaction.
        result = None
        if bakery_items:
            deduction_items = []
            for item in bakery_items:
                priced_item = priced_by_item_id[id(item)]
                deduction_items.append(
                    {
                        **item,
                        "unit_price": float(priced_item["unit_price"]),
                        "discount_applied": float(priced_item["discount_rate"]),
                    }
                )
            req = DeductRequest(items=deduction_items, receipt_id=receipt_id)
            result = await deduct_inventory(req, db=db)
            deducted.extend(result.deducted)
            all_errors.extend(result.errors)
            status = result.status

        if all_errors:
            attempted_deductions = list(deducted)
            db.rollback()
            return {
                "status": status,
                "deducted": [],
                "attempted_deductions": attempted_deductions,
                "errors": all_errors,
                "receipt": None,
                "message": f"0 items deducted; transaction rolled back after {len(all_errors)} item failures",
            }

        # Beverages have no finished-goods stock limit, but their outflows are
        # still part of the same checkout transaction.
        for item in coffee_items:
            pn = item.get("product_name", "")
            qty = item["quantity"]
            price = priced_by_item_id[id(item)]["unit_price"]
            q(db, "inventory_transactions").insert({
                "transaction_type": "outflow",
                "batch_id": None,
                "product_name": pn,
                "quantity": qty,
                "unit_price": float(price),
                "discount_applied": 0,
                "freshness_status": "Fresh",
                "receipt_id": receipt_id,
                "disposition": "sold",
            }).execute()
            deducted.append({
                "product_name": pn,
                "batch_id": None,
                "quantity_deducted": qty,
                "remaining_after": 0,
            })

        # ---- Record to orders / order_items / payments tables ----
        # Calculate order totals
        order_subtotal = subtotal
        order_discount = discount_total
        order_total = total
        order_cost = _packaging_material_cost(
            material_requirements,
            material_info,
        )
        order_item_count = 0

        for priced_item in priced_items:
            item = priced_item["item"]
            pn = item.get("product_name","")
            uprice = priced_item["unit_price"]
            quantity_decimal = priced_item["quantity_decimal"]
            mat_cost = product_data[pn]["material_cost"]
            wastage_pct = product_data[pn]["wastage_pct"]
            actual_cost = mat_cost * (Decimal("1") + wastage_pct)
            order_cost += actual_cost * quantity_decimal
            order_item_count += priced_item["quantity"]

        order_profit = order_total - order_cost

        # INSERT orders
        cur.execute(
            "INSERT INTO orders (ticket_id, order_date, order_time, subtotal, discount_total, total_amount, total_profit, item_count, state, dine_type) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (receipt_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), float(order_subtotal), float(order_discount), float(order_total), float(order_profit), order_item_count, "paid", dine_type)
        )
        order_id = cur.lastrowid

        # INSERT order_items
        for priced_item in priced_items:
            item = priced_item["item"]
            pn = item.get("product_name","")
            qty = priced_item["quantity"]
            uprice = priced_item["unit_price"]
            mat_cost = product_data[pn]["material_cost"]
            wastage_pct = product_data[pn]["wastage_pct"]
            actual_cost = mat_cost * (Decimal("1") + wastage_pct)
            freshness = item.get("freshness", "Fresh")
            disc_rate = priced_item["discount_rate"]
            line_profit = float(priced_item["line_final"] - (actual_cost * priced_item["quantity_decimal"]))

            cur.execute(
                "INSERT INTO order_items (order_id, product_name, quantity, unit_price, discount_rate, line_total, line_profit, freshness, coffee_size, coffee_temp, coffee_ice, coffee_sugar) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    order_id,
                    pn,
                    qty,
                    float(uprice),
                    disc_rate,
                    float(priced_item["line_final"]), line_profit,
                    freshness,
                    item.get("size") if is_beverage(pn) else None,
                    item.get("temperature") if is_beverage(pn) else None,
                    item.get("ice_level") if is_beverage(pn) else None,
                    item.get("sugar") if is_beverage(pn) else None,
                )
            )

        for material_name in sorted(material_requirements):
            required = material_requirements[material_name]
            cur.execute(
                """
                UPDATE raw_materials
                SET stock_quantity = stock_quantity - %s
                WHERE material_name = %s AND stock_quantity >= %s
                """,
                (required, material_name, required),
            )
            if cur.rowcount != 1:
                raise HTTPException(
                    409,
                    f"Material stock changed during checkout: {material_name}",
                )
            cur.execute(
                "INSERT INTO material_transactions (material_name, transaction_type, quantity, unit, reference) VALUES (%s,%s,%s,%s,%s)",
                (
                    material_name,
                    "outflow",
                    required,
                    material_info[material_name]["unit"],
                    receipt_id,
                ),
            )
    
        # INSERT payments
        cur.execute(
            "INSERT INTO payments (order_id, amount, payment_method, payment_date) VALUES (%s,%s,%s,%s)",
            (order_id, float(order_total), payment_method, now.strftime("%Y-%m-%d"))
        )

        # Add packaging fee to receipt if applicable
        if packaging_fee > 0:
            receipt_items.append({
                "product_name": "Packaging (Takeaway)",
                "quantity": 1,
                "unit_price": float(packaging_fee),
                "discount_pct": 0,
                "discount_amount": 0,
                "line_total": float(packaging_fee),
            })
        
        receipt = {
            "receipt_id": receipt_id,
            "date": now.strftime("%Y-%m-%d %H:%M"),
            "items": receipt_items,
            "subtotal": float(subtotal),
            "discount_total": float(discount_total),
            "total": float(total),
            "savings": float(savings),
            "order_id": order_id,
        }

        cur.execute(
            "INSERT INTO receipts (receipt_id, items, subtotal, discount_total, total, savings) VALUES (%s,%s,%s,%s,%s,%s)",
            (
                receipt_id,
                json.dumps(receipt_items, separators=(",", ":")),
                float(subtotal),
                float(discount_total),
                float(total),
                float(savings),
            ),
        )
        db.commit()

        return {
            "status": status,
            "deducted": deducted,
            "errors": all_errors,
            "receipt": receipt,
            "message": f"{len(deducted)} items deducted" + (f", {len(all_errors)} items failed" if all_errors else ""),
        }

    except HTTPException:
        if db is not None:
            db.rollback()
        raise
    except Exception as e:
        attempted_deductions = list(deducted)
        if db is not None:
            db.rollback()
        logger.exception("Checkout transaction failed")
        return {
            "status": "error",
            "deducted": [],
            "attempted_deductions": attempted_deductions,
            "errors": [f"Database write failed: {str(e)}"],
            "receipt": None,
            "message": f"Checkout failed: {str(e)}",
        }

    finally:
        if cur is not None:
            cur.close()
        if db is not None:
            db.close()
# GET /s4/revenue/daily -- Revenue dashboard data from MySQL
# ======================================================================
# ======================================================================
# GET /s4/orders/today -- List today's paid orders for refund
# ======================================================================
@router.get("/orders/today", dependencies=[Depends(get_current_user)])
async def orders_today(date: str = None):
    db = get_db()
    cur = db.cursor()
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    cur.execute(
        "SELECT ticket_id, order_time, total_amount, dine_type, state, item_count FROM orders WHERE order_date = %s AND state IN ('paid','refunded') ORDER BY order_time DESC LIMIT 50",
        (date,)
    )
    rows = cur.fetchall()
    orders = []
    for r in rows:
        orders.append({
            "ticket_id": r[0],
            "order_time": str(r[1]),
            "total_amount": float(r[2]),
            "dine_type": r[3],
            "state": r[4],
            "item_count": r[5],
        })
    cur.close()
    return {"orders": orders, "date": date}

# POST /s4/orders/refund -- Void/refund an order
# ======================================================================
# ======================================================================
# GET /s4/orders/receipt -- Get receipt for an order
# ======================================================================
@router.get("/orders/receipt", dependencies=[Depends(get_current_user)])
async def order_receipt(ticket_id: str):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, order_date, order_time, subtotal, discount_total, total_amount, state, dine_type FROM orders WHERE ticket_id = %s", (ticket_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(404, f"Order {ticket_id} not found")
    order_id, order_date, order_time, subtotal, discount_total, total_amount, state, dine_type = row

    cur.execute("SELECT product_name, quantity, unit_price, line_total, freshness, coffee_size, coffee_temp, coffee_sugar, coffee_ice FROM order_items WHERE order_id = %s", (order_id,))
    items = []
    for r in cur.fetchall():
        items.append({
            "product_name": r[0], "quantity": r[1], "unit_price": float(r[2]),
            "line_total": float(r[3]), "freshness": r[4], "size": r[5],
            "temp": r[6], "sugar": r[7], "ice": r[8],
        })
    cur.close()
    return {
        "ticket_id": ticket_id,
        "date": str(order_date),
        "time": str(order_time),
        "subtotal": float(subtotal),
        "discount": float(discount_total),
        "total": float(total_amount),
        "state": state,
        "dine_type": dine_type,
        "items": items,
    }

@router.post("/orders/refund")
async def refund_order(payload: dict, user=Depends(require_manager)):
    """Record a paid order return without restoring sellable inventory."""
    ticket_id = payload.get("ticket_id", "")
    if not ticket_id:
        raise HTTPException(400, "ticket_id required")
    reason = payload.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        raise HTTPException(400, "Refund reason required")
    reason = reason.strip()
    if len(reason) > 255:
        raise HTTPException(400, "Refund reason is too long")
    actor = str(user.get("sub") or "").strip()
    if not actor:
        raise HTTPException(401, "Invalid token")

    db = get_db(autocommit=False)
    cur = db.cursor()
    try:
        cur.execute(
            "SELECT id, state, dine_type FROM orders WHERE ticket_id = %s FOR UPDATE",
            (ticket_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, f"Order {ticket_id} not found")
        order_id, state, dine_type = row
        if state == "refunded":
            raise HTTPException(400, "Order already refunded")
        if state != "paid":
            raise HTTPException(400, f"Cannot refund order in state: {state}")

        cur.execute(
            "SELECT product_name, quantity, freshness, coffee_size FROM order_items WHERE order_id = %s",
            (order_id,),
        )
        items = cur.fetchall()
        expected_by_product = {}
        for product_name, quantity, _freshness, _coffee_size in items:
            expected_by_product[product_name] = (
                expected_by_product.get(product_name, 0) + quantity
            )

        cur.execute(
            """
            SELECT id, batch_id, product_name, quantity, unit_price,
                   discount_applied, freshness_status
            FROM inventory_transactions
            WHERE receipt_id = %s AND transaction_type = 'outflow'
            ORDER BY id
            FOR UPDATE
            """,
            (ticket_id,),
        )
        outflows = cur.fetchall()
        actual_by_product = {}
        for _transaction_id, _batch_id, product_name, quantity, *_rest in outflows:
            actual_by_product[product_name] = (
                actual_by_product.get(product_name, 0) + quantity
            )
        if not outflows or actual_by_product != expected_by_product:
            raise HTTPException(
                409,
                f"Original inventory allocation is unavailable for {ticket_id}",
            )

        returned_units = 0
        for (
            transaction_id,
            batch_id,
            product_name,
            quantity,
            unit_price,
            discount_applied,
            freshness_status,
        ) in outflows:
            cur.execute(
                """
                INSERT INTO inventory_transactions (
                    transaction_type, batch_id, product_name, quantity,
                    unit_price, discount_applied, freshness_status, receipt_id,
                    reversal_of_transaction_id, disposition, reason, performed_by
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    "return",
                    batch_id,
                    product_name,
                    quantity,
                    unit_price,
                    discount_applied,
                    freshness_status,
                    ticket_id,
                    transaction_id,
                    "non_sellable",
                    reason,
                    actor,
                ),
            )
            returned_units += quantity

        cur.execute(
            """
            UPDATE orders
            SET state = 'refunded', refund_reason = %s,
                refunded_by = %s, refunded_at = %s
            WHERE id = %s
            """,
            (reason, actor, datetime.now(), order_id),
        )
        db.commit()
        return {
            "status": "ok",
            "message": f"Order {ticket_id} returned as non-sellable",
            "returned_units": returned_units,
            "disposition": "non_sellable",
        }
    except Exception:
        db.rollback()
        raise
    finally:
        cur.close()
        db.close()

def _get_non_sellable_return_cost(cur, start_date, end_date=None):
    date_clause = "BETWEEN %s AND %s" if end_date else "= %s"
    params = (start_date, end_date) if end_date else (start_date,)
    cur.execute(
        f"""
        SELECT COALESCE(
            SUM(
                it.quantity
                * p.material_cost
                * (1 + COALESCE(p.wastage_pct, 0.03))
            ),
            0
        )
        FROM inventory_transactions it
        JOIN products p ON it.product_name = p.product_name
        WHERE it.transaction_type = 'return'
          AND it.disposition = 'non_sellable'
          AND DATE(it.transaction_time) {date_clause}
        """,
        params,
    )
    return round(float(cur.fetchone()[0] or 0), 2)


def _get_expired_cost(cur, start_date, end_date=None):
    date_clause = "BETWEEN %s AND %s" if end_date else "= %s"
    params = (start_date, end_date) if end_date else (start_date,)
    cur.execute(
        f"""
        SELECT COALESCE(
            SUM(
                it.quantity
                * COALESCE(NULLIF(it.unit_price, 0), p.material_cost)
            ),
            0
        )
        FROM inventory_transactions it
        JOIN products p ON it.product_name = p.product_name
        WHERE it.transaction_type = 'outflow'
          AND it.freshness_status = 'Expired'
          AND p.category = 'bakery'
          AND DATE(it.transaction_time) {date_clause}
        """,
        params,
    )
    return round(float(cur.fetchone()[0] or 0), 2)


def _get_expired_product_breakdown(
    cur,
    start_date,
    total_expired_cost=None,
    end_date=None,
    limit=5,
):
    end_date = end_date or start_date
    cur.execute(
        """
        SELECT expired.product_name,
               expired.expired_qty,
               expired.expired_cost,
               COALESCE(sold.sold_qty, 0) AS sold_qty,
               COALESCE(sold.revenue, 0) AS revenue,
               COALESCE(sold.profit, 0) AS profit
        FROM (
            SELECT it.product_name,
                   SUM(ABS(it.quantity)) AS expired_qty,
                   SUM(
                       ABS(it.quantity)
                       * COALESCE(NULLIF(it.unit_price, 0), p.material_cost)
                   ) AS expired_cost
            FROM inventory_transactions it
            JOIN products p ON it.product_name = p.product_name
            WHERE it.transaction_type = 'outflow'
              AND it.freshness_status = 'Expired'
              AND p.category = 'bakery'
              AND DATE(it.transaction_time) BETWEEN %s AND %s
            GROUP BY it.product_name
        ) expired
        LEFT JOIN (
            SELECT oi.product_name,
                   SUM(oi.quantity) AS sold_qty,
                   SUM(oi.line_total) AS revenue,
                   SUM(oi.line_profit) AS profit
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            WHERE o.order_date BETWEEN %s AND %s
              AND o.state IN ('paid','completed')
            GROUP BY oi.product_name
        ) sold ON sold.product_name = expired.product_name
        ORDER BY expired.expired_cost DESC, expired.product_name
        """,
        (start_date, end_date, start_date, end_date),
    )
    rows = cur.fetchall()
    if total_expired_cost is None:
        total_expired_cost = sum(float(row[2] or 0) for row in rows)
    products = []
    for row in rows:
        expired_qty = int(row[1] or 0)
        expired_cost = round(float(row[2] or 0), 2)
        sold_qty = int(row[3] or 0)
        revenue = round(float(row[4] or 0), 2)
        profit = round(float(row[5] or 0), 2)
        available_qty = sold_qty + expired_qty
        products.append(
            {
                "name": str(row[0]).replace("_", " ").title(),
                "expired_qty": expired_qty,
                "expired_cost": expired_cost,
                "sold_qty": sold_qty,
                "revenue": revenue,
                "profit": profit,
                "margin_pct": round(profit / revenue * 100, 2) if revenue else 0.0,
                "sell_through_pct": (
                    round(sold_qty / available_qty * 100, 2)
                    if available_qty
                    else 0.0
                ),
                "loss_share_pct": (
                    round(expired_cost / total_expired_cost * 100, 2)
                    if total_expired_cost
                    else 0.0
                ),
            }
        )
    return products if limit is None else products[:limit]


@router.get("/revenue/closing-loss", dependencies=[Depends(require_manager)])
async def revenue_closing_loss(start: str = None, end: str = None):
    """Return finished-product closing losses for a selected date range."""
    if start and not end:
        end = start
    elif end and not start:
        start = end

    db = get_db()
    cur = db.cursor()
    try:
        if start is None:
            cur.execute(
                "SELECT MAX(order_date) FROM orders "
                "WHERE state IN ('paid','completed')"
            )
            latest_date = cur.fetchone()[0]
            start = end = str(latest_date or datetime.now().date())

        try:
            start_value = datetime.strptime(start, "%Y-%m-%d").date()
            end_value = datetime.strptime(end, "%Y-%m-%d").date()
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "Dates must use YYYY-MM-DD format") from exc
        if start_value > end_value:
            raise HTTPException(400, "Start date must not be after end date")

        products = _get_expired_product_breakdown(
            cur,
            start,
            end_date=end,
            limit=None,
        )
        total_expired_cost = round(
            sum(float(item["expired_cost"]) for item in products),
            2,
        )
        total_expired_qty = sum(int(item["expired_qty"]) for item in products)
        return {
            "status": "ok",
            "data": {
                "start": start,
                "end": end,
                "total_expired_cost": total_expired_cost,
                "total_expired_qty": total_expired_qty,
                "product_count": len(products),
                "products": products,
            },
        }
    finally:
        cur.close()
        db.close()


def _get_sold_bread_sku_count(cur, date):
    cur.execute(
        """
        SELECT COUNT(DISTINCT oi.product_name)
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_name = p.product_name
        WHERE o.order_date = %s
          AND o.state IN ('paid','completed')
          AND p.category = 'bakery'
        """,
        (date,),
    )
    return int(cur.fetchone()[0] or 0)


def _get_non_sellable_return_cost_by_hour(cur, date):
    cur.execute(
        """
        SELECT HOUR(it.transaction_time),
               COALESCE(
                   SUM(
                       it.quantity
                       * p.material_cost
                       * (1 + COALESCE(p.wastage_pct, 0.03))
                   ),
                   0
               )
        FROM inventory_transactions it
        JOIN products p ON it.product_name = p.product_name
        WHERE it.transaction_type = 'return'
          AND it.disposition = 'non_sellable'
          AND DATE(it.transaction_time) = %s
        GROUP BY HOUR(it.transaction_time)
        ORDER BY HOUR(it.transaction_time)
        """,
        (date,),
    )
    return {
        int(hour): round(float(cost or 0), 2)
        for hour, cost in cur.fetchall()
        if hour is not None
    }


def _revenue_period_expressions(granularity, date_expression):
    if granularity == "week":
        return (
            f"YEARWEEK({date_expression}, 1)",
            f"CONCAT(YEAR({date_expression}), '-W', "
            f"LPAD(WEEK({date_expression}, 1), 2, '0'))",
        )
    if granularity == "month":
        expression = f"DATE_FORMAT({date_expression}, '%Y-%m')"
        return expression, expression
    if granularity == "year":
        expression = f"YEAR({date_expression})"
        return expression, expression
    return date_expression, date_expression


def _get_non_sellable_return_cost_by_period(
    cur,
    start,
    end,
    granularity,
    category,
):
    group_expr, label_expr = _revenue_period_expressions(
        granularity,
        "DATE(it.transaction_time)",
    )
    cur.execute(
        f"""
        SELECT it.product_name, {label_expr} AS period_label,
               {group_expr} AS period_val,
               COALESCE(
                   SUM(
                       it.quantity
                       * p.material_cost
                       * (1 + COALESCE(p.wastage_pct, 0.03))
                   ),
                   0
               ) AS return_cost
        FROM inventory_transactions it
        JOIN products p ON it.product_name = p.product_name
        WHERE DATE(it.transaction_time) BETWEEN %s AND %s
          AND it.transaction_type = 'return'
          AND it.disposition = 'non_sellable'
          AND (%s = 'total' OR p.category = CASE
              WHEN %s = 'bread' THEN 'bakery'
              WHEN %s = 'beverages' THEN 'beverages'
          END)
        GROUP BY it.product_name, period_val, period_label
        ORDER BY period_val, it.product_name
        """,
        (start, end, category, category, category),
    )
    return cur.fetchall()


def _get_expired_cost_by_period(cur, start, end, granularity, category):
    group_expr, label_expr = _revenue_period_expressions(
        granularity,
        "DATE(it.transaction_time)",
    )
    cur.execute(
        f"""
        SELECT it.product_name, {label_expr} AS period_label,
               {group_expr} AS period_val,
               COALESCE(
                   SUM(
                       it.quantity
                       * COALESCE(NULLIF(it.unit_price, 0), p.material_cost)
                   ),
                   0
               ) AS expired_cost
        FROM inventory_transactions it
        JOIN products p ON it.product_name = p.product_name
        WHERE DATE(it.transaction_time) BETWEEN %s AND %s
          AND it.transaction_type = 'outflow'
          AND it.freshness_status = 'Expired'
          AND p.category = 'bakery'
          AND (%s = 'total' OR p.category = CASE
              WHEN %s = 'bread' THEN 'bakery'
              WHEN %s = 'beverages' THEN 'beverages'
          END)
        GROUP BY it.product_name, period_val, period_label
        ORDER BY period_val, it.product_name
        """,
        (start, end, category, category, category),
    )
    return cur.fetchall()


def _get_order_adjustments_by_period(cur, start, end, granularity, category):
    if category != "total":
        return []
    group_expr, label_expr = _revenue_period_expressions(
        granularity,
        "adjustments.order_date",
    )
    cur.execute(
        f"""
        SELECT {label_expr} AS period_label,
               {group_expr} AS period_val,
               COALESCE(SUM(adjustments.revenue_adjustment), 0),
               COALESCE(SUM(adjustments.profit_adjustment), 0)
        FROM (
            SELECT o.id, o.order_date,
                   o.total_amount - COALESCE(SUM(oi.line_total), 0)
                       AS revenue_adjustment,
                   o.total_profit - COALESCE(SUM(oi.line_profit), 0)
                       AS profit_adjustment
            FROM orders o
            LEFT JOIN order_items oi ON oi.order_id = o.id
            WHERE o.order_date BETWEEN %s AND %s
              AND o.state IN ('paid','completed')
            GROUP BY o.id, o.order_date, o.total_amount, o.total_profit
        ) adjustments
        GROUP BY period_val, period_label
        ORDER BY period_val
        """,
        (start, end),
    )
    return cur.fetchall()


# Revenue helpers
def _get_dine_type_breakdown(cur, date):
    cur.execute(
        """
        SELECT dine_type, COUNT(*)
        FROM orders
        WHERE order_date = %s
          AND state IN ('paid','completed')
        GROUP BY dine_type
        """,
        (date,),
    )
    breakdown = {"Dine-in": 0, "Takeaway": 0}
    for dine_type, order_count in cur.fetchall():
        if dine_type == "dine_in":
            breakdown["Dine-in"] = int(order_count or 0)
        elif dine_type == "takeaway":
            breakdown["Takeaway"] = int(order_count or 0)
    return breakdown


def _revenue_change(current, baseline):
    if baseline is None or float(baseline) <= 0:
        return None
    return round((float(current) - float(baseline)) / float(baseline) * 100, 1)


def _order_basket_metrics(
    revenue,
    orders,
    items,
    previous_revenue=None,
    previous_orders=None,
    previous_items=None,
):
    items_per_order_raw = items / orders if orders else 0
    revenue_per_item_raw = revenue / items if items else 0
    items_per_order = round(items_per_order_raw, 2)
    revenue_per_item = round(revenue_per_item_raw, 2)
    previous_items_per_order = (
        previous_items / previous_orders
        if previous_orders and previous_items is not None
        else None
    )
    previous_revenue_per_item = (
        previous_revenue / previous_items
        if previous_revenue is not None and previous_items
        else None
    )
    return {
        "today_items": int(items or 0),
        "items_per_order": items_per_order,
        "items_per_order_change": _revenue_change(
            items_per_order_raw,
            previous_items_per_order,
        ),
        "revenue_per_item": revenue_per_item,
        "revenue_per_item_change": _revenue_change(
            revenue_per_item_raw,
            previous_revenue_per_item,
        ),
    }


def _get_recent_revenue_baseline(cur, date, limit=7):
    cur.execute(
        """
        SELECT o.order_date, COUNT(*) AS orders,
               COALESCE(SUM(o.total_amount), 0) AS revenue
        FROM orders o
        WHERE o.order_date < %s
          AND o.state IN ('paid','completed')
        GROUP BY o.order_date
        ORDER BY o.order_date DESC
        LIMIT %s
        """,
        (date, limit),
    )
    rows = cur.fetchall()
    if not rows:
        return {
            "day_count": 0,
            "start_date": None,
            "end_date": None,
            "avg_revenue": None,
            "avg_orders": None,
            "avg_order_value": None,
        }
    dates = [str(row[0]) for row in rows]
    total_orders = sum(int(row[1] or 0) for row in rows)
    total_revenue = sum(float(row[2] or 0) for row in rows)
    day_count = len(rows)
    return {
        "day_count": day_count,
        "start_date": min(dates),
        "end_date": max(dates),
        "avg_revenue": round(total_revenue / day_count, 2),
        "avg_orders": round(total_orders / day_count, 2),
        "avg_order_value": (
            round(total_revenue / total_orders, 2) if total_orders else None
        ),
    }


@router.get("/revenue/daily")
async def revenue_daily(
    date: str = None,
    _: dict = Depends(require_manager),
):
    """Return revenue dashboard data from MySQL orders/order_items/products tables."""
    from datetime import datetime as dt, timedelta
    
    db = get_db()
    cur = db.cursor()
    
    # Default to latest order date
    if date is None:
        cur.execute("SELECT MAX(order_date) FROM orders WHERE state IN ('paid','completed')")
        date = str(cur.fetchone()[0])
    
    # Today KPIs
    cur.execute("""
        SELECT COUNT(*) as orders, SUM(total_amount) as revenue, SUM(total_profit) as profit,
               COALESCE(SUM(discount_total),0) as discount, COALESCE(SUM(item_count),0) as items
        FROM orders WHERE order_date = %s AND state IN ('paid','completed')
    """, (date,))
    row = cur.fetchone()

    expired_cost = _get_expired_cost(cur, date)
    expired_products = _get_expired_product_breakdown(cur, date, expired_cost)
    sold_bread_sku_count = _get_sold_bread_sku_count(cur, date)
    non_sellable_return_cost = _get_non_sellable_return_cost(cur, date)

    if not row or not row[0]:
        if expired_cost <= 0 and non_sellable_return_cost <= 0:
            return {"status": "ok", "data": None, "message": f"No sales data for {date}"}
        row = (0, 0, 0, 0, 0)

    today_orders = int(row[0])
    today_revenue = round(float(row[1] or 0), 2)
    today_profit = round(float(row[2] or 0), 2)
    avg_order = round(today_revenue / today_orders, 2) if today_orders else 0
    today_discount = round(float(row[3] or 0), 2)
    today_items = int(row[4] or 0)

    today_profit = round(today_profit - expired_cost - non_sellable_return_cost, 2)
    
    # Profit margin
    profit_margin = round(today_profit / today_revenue * 100, 1) if today_revenue else 0
    
    # MTD (Month-to-Date cumulative)
    month_start = date[:8] + "01"
    cur.execute("""
        SELECT COALESCE(SUM(total_amount),0), COALESCE(SUM(total_profit),0), COUNT(*)
        FROM orders
        WHERE order_date >= %s AND order_date <= %s
          AND state IN ('paid','completed')
    """, (month_start, date))
    mtd_row = cur.fetchone()
    mtd_revenue = round(float(mtd_row[0] or 0), 2)
    mtd_profit = round(float(mtd_row[1] or 0), 2)
    mtd_orders = int(mtd_row[2] or 0)
    
    mtd_expired_cost = _get_expired_cost(cur, month_start, date)
    mtd_non_sellable_return_cost = _get_non_sellable_return_cost(
        cur,
        month_start,
        date,
    )
    mtd_profit = round(
        mtd_profit - mtd_expired_cost - mtd_non_sellable_return_cost,
        2,
    )
    
    # Yesterday comparison
    yesterday = (dt.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    cur.execute("""
        SELECT COUNT(*) as orders, SUM(total_amount) as revenue, SUM(total_profit) as profit,
               COALESCE(SUM(discount_total),0) as discount, COALESCE(SUM(item_count),0) as items
        FROM orders WHERE order_date = %s AND state IN ('paid','completed')
    """, (yesterday,))
    yrow = cur.fetchone()
    previous_day_available = bool(yrow and yrow[0])
    if previous_day_available:
        y_orders = int(yrow[0])
        y_revenue = round(float(yrow[1] or 0), 2)
        y_profit = round(
            float(yrow[2] or 0)
            - _get_expired_cost(cur, yesterday)
            - _get_non_sellable_return_cost(cur, yesterday),
            2,
        )
        y_avg = round(y_revenue / y_orders, 2) if y_orders else 0
        y_items = int(yrow[4] or 0)
        rev_change = _revenue_change(today_revenue, y_revenue)
        prof_change = _revenue_change(today_profit, y_profit)
        ord_change = _revenue_change(today_orders, y_orders)
        avg_change = _revenue_change(avg_order, y_avg)
    else:
        y_orders = y_revenue = y_items = None
        rev_change = prof_change = ord_change = avg_change = None

    basket_metrics = _order_basket_metrics(
        revenue=today_revenue,
        orders=today_orders,
        items=today_items,
        previous_revenue=y_revenue,
        previous_orders=y_orders,
        previous_items=y_items,
    )
    recent_baseline = _get_recent_revenue_baseline(cur, date)
    
        # Payment breakdown (real data from payments table)
    cur.execute("""
        SELECT p.payment_method, COUNT(*) as cnt
        FROM payments p JOIN orders o ON p.order_id = o.id
        WHERE o.order_date = %s AND o.state IN ('paid','completed')
        GROUP BY p.payment_method
    """, (date,))
    payment = {"Cash": 0, "Card": 0, "QR": 0}
    for prow in cur.fetchall():
        key = prow[0]
        if not key:
            continue  # skip NULL payment_method (historical orders)
        if key in ("cash",): key = "Cash"
        elif key in ("card",): key = "Card"
        elif key in ("qr",): key = "QR"
        else:
            continue  # skip unknown payment methods
        if key in payment:
            payment[key] = int(prow[1] or 0)

    dine_type_breakdown = _get_dine_type_breakdown(cur, date)
    
    # Category breakdown (bread vs beverages)
    cur.execute("""
        SELECT p.category, SUM(oi.line_total) as revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_name = p.product_name
        WHERE o.order_date = %s AND o.state IN ('paid','completed')
        GROUP BY p.category
    """, (date,))
    cat_data = {"Bread": 0, "Beverages": 0}
    for crow in cur.fetchall():
        cat_key = "Bread" if crow[0] == "bakery" else "Beverages"
        cat_data[cat_key] = round(float(crow[1] or 0), 2)
    
    # Split ranking: bread vs beverages
    BEVERAGE_NAMES = {"latte","americano","cappuccino","mocha","espresso","flat_white","caramel_macchiato","cold_brew","hot_chocolate","matcha_latte","milk_tea","chai_latte","earl_grey","english_breakfast","lemonade"}
    beverage_placeholders = ', '.join(['%s'] * len(BEVERAGE_NAMES))

    # Bread ranking (top 5)
    cur.execute(f"""
        SELECT oi.product_name, SUM(oi.quantity) as qty, SUM(oi.line_total) as revenue,
               SUM(oi.line_profit) as profit
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_name = p.product_name
        WHERE o.order_date = %s
          AND o.state IN ('paid','completed')
          AND oi.product_name NOT IN ({beverage_placeholders})
        GROUP BY oi.product_name
        ORDER BY revenue DESC LIMIT 5
    """, [date] + list(BEVERAGE_NAMES))
    bread_ranking = []
    for rrow in cur.fetchall():
        bread_ranking.append({
            "name": rrow[0].replace("_", " ").title(),
            "qty": int(rrow[1]),
            "revenue": round(float(rrow[2]), 2),
            "profit": round(float(rrow[3]), 2)
        })

    # Beverages ranking (top 5)
    cur.execute(f"""
        SELECT oi.product_name, SUM(oi.quantity) as qty, SUM(oi.line_total) as revenue,
               SUM(oi.line_profit) as profit
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_name = p.product_name
        WHERE o.order_date = %s
          AND o.state IN ('paid','completed')
          AND oi.product_name IN ({beverage_placeholders})
        GROUP BY oi.product_name
        ORDER BY revenue DESC LIMIT 5
    """, [date] + list(BEVERAGE_NAMES))
    beverage_ranking = []
    for rrow in cur.fetchall():
        beverage_ranking.append({
            "name": rrow[0].replace("_", " ").title(),
            "qty": int(rrow[1]),
            "revenue": round(float(rrow[2]), 2),
            "profit": round(float(rrow[3]), 2)
        })
    
    # 7-day trend (bread vs beverages revenue + orders + avg per day)
    trend_dates = []
    trend_bread = []
    trend_beverages = []
    trend_orders = []
    trend_avg = []
    for i in range(6, -1, -1):
        d = (dt.strptime(date, "%Y-%m-%d") - timedelta(days=i)).strftime("%Y-%m-%d")
        trend_dates.append(d[5:])  # MM-DD
        cur.execute("""
            SELECT p.category, COALESCE(SUM(oi.line_total), 0)
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON oi.product_name = p.product_name
            WHERE o.order_date = %s AND o.state IN ('paid','completed')
            GROUP BY p.category
        """, (d,))
        day_cat = {"bakery": 0, "beverage": 0}
        for crow in cur.fetchall():
            day_cat[crow[0]] = round(float(crow[1]), 2)
        trend_bread.append(day_cat["bakery"])
        trend_beverages.append(day_cat["beverage"])
        cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(total_amount),0) FROM orders WHERE order_date = %s AND state IN ('paid','completed')",
            (d,),
        )
        orow = cur.fetchone()
        day_orders = int(orow[0] or 0)
        day_rev = float(orow[1] or 0)
        trend_orders.append(day_orders)
        trend_avg.append(round(day_rev / day_orders, 2) if day_orders else 0)
    
    return {
        "status": "ok",
        "data": {
            "date": date,
            "today_revenue": today_revenue,
            "today_profit": today_profit,
            "today_orders": today_orders,
            "avg_order": avg_order,
            **basket_metrics,
            "today_discount": today_discount,
            "expired_cost": expired_cost,
            "expired_products": expired_products,
            "sold_bread_sku_count": sold_bread_sku_count,
            "non_sellable_return_cost": non_sellable_return_cost,
            "profit_margin": profit_margin,
            "mtd_revenue": mtd_revenue,
            "mtd_profit": mtd_profit,
            "mtd_orders": mtd_orders,
            "mtd_expired_cost": mtd_expired_cost,
            "mtd_non_sellable_return_cost": mtd_non_sellable_return_cost,
            "revenue_change": rev_change,
            "profit_change": prof_change,
            "orders_change": ord_change,
            "avg_change": avg_change,
            "previous_day_available": previous_day_available,
            "previous_day_date": yesterday,
            "recent_baseline": recent_baseline,
            "payment": payment,
            "dine_type": dine_type_breakdown,
            "category": cat_data,
            "trend": {"dates": trend_dates, "bread": trend_bread, "beverages": trend_beverages, "orders": trend_orders, "avg_order": trend_avg},
            "bread_ranking": bread_ranking,
            "beverage_ranking": beverage_ranking,
        }
    }


# GET /s4/revenue/hourly -- Hourly breakdown of bread vs beverages sales
@router.get("/revenue/hourly", dependencies=[Depends(require_manager)])
async def revenue_hourly(date: str = None):
    """Return hourly revenue, profit, order behavior, and category split for a date."""

    db = get_db()
    cur = db.cursor()

    if date is None:
        cur.execute("SELECT MAX(order_date) FROM orders WHERE state IN ('paid','completed')")
        date = str(cur.fetchone()[0])

    cur.execute("""
        SELECT HOUR(order_time) as hr,
               COUNT(*) as orders,
               COALESCE(SUM(total_amount), 0) as revenue,
               COALESCE(SUM(total_profit), 0) as profit
        FROM orders
        WHERE order_date = %s AND state IN ('paid','completed')
        GROUP BY hr
        ORDER BY hr
    """, (date,))
    order_rows = [r for r in cur]

    cur.execute("""
        SELECT HOUR(o.order_time) as hr, p.category, SUM(oi.line_total) as revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_name = p.product_name
        WHERE o.order_date = %s AND o.state IN ('paid','completed')
        GROUP BY hr, p.category
        ORDER BY hr
    """, (date,))
    rows = [r for r in cur]
    return_cost_by_hour = _get_non_sellable_return_cost_by_hour(cur, date)
    expired_cost = _get_expired_cost(cur, date)
    all_hours = sorted(
        set(int(r[0]) for r in rows + order_rows if r[0] is not None)
        | set(return_cost_by_hour)
    )
    if all_hours:
        raw_min = min(all_hours)
        raw_max = max(all_hours)
        min_hr = max(6, raw_min - 1)
        max_hr = min(23, raw_max + 1)
        if min_hr >= max_hr:
            min_hr = 6
            max_hr = 22
    else:
        cur.execute(
            """SELECT COALESCE(MIN(HOUR(order_time)), 8), COALESCE(MAX(HOUR(order_time)), 21) FROM orders WHERE order_date >= DATE_SUB(%s, INTERVAL 30 DAY) AND state IN ('paid','completed')""",
            (date,),
        )
        range_row = cur.fetchone()
        min_hr = max(6, int(range_row[0] or 8) - 1)
        max_hr = min(23, int(range_row[1] or 21) + 1)

    num_hours = max_hr - min_hr + 1
    hours = [f"{h:02d}:00" for h in range(min_hr, max_hr + 1)]
    bread_data = [0.0] * num_hours
    beverage_data = [0.0] * num_hours
    revenue_data = [0.0] * num_hours
    profit_data = [0.0] * num_hours
    return_cost_data = [0.0] * num_hours
    order_data = [0] * num_hours
    avg_order_data = [0.0] * num_hours
    margin_data = [0.0] * num_hours
    expired_cost_data = [0.0] * num_hours

    for row in rows:
        hr = int(row[0])
        cat = row[1]
        rev = round(float(row[2] or 0), 2)
        idx = hr - min_hr
        if 0 <= idx < num_hours:
            if cat == "bakery":
                bread_data[idx] = rev
            else:
                beverage_data[idx] = rev

    for row in order_rows:
        hr = int(row[0])
        idx = hr - min_hr
        if 0 <= idx < num_hours:
            orders = int(row[1] or 0)
            revenue = round(float(row[2] or 0), 2)
            profit = round(float(row[3] or 0), 2)
            order_data[idx] = orders
            revenue_data[idx] = revenue
            profit_data[idx] = profit
            avg_order_data[idx] = round(revenue / orders, 2) if orders else 0.0
            margin_data[idx] = round(profit / revenue * 100, 1) if revenue else 0.0

    for hour, return_cost in return_cost_by_hour.items():
        idx = hour - min_hr
        if 0 <= idx < num_hours:
            return_cost_data[idx] = return_cost
            profit_data[idx] = round(profit_data[idx] - return_cost, 2)
            revenue = revenue_data[idx]
            margin_data[idx] = (
                round(profit_data[idx] / revenue * 100, 1) if revenue else 0.0
            )

    if expired_cost > 0:
        hours.append("Closing adjustment")
        bread_data.append(0.0)
        beverage_data.append(0.0)
        revenue_data.append(0.0)
        profit_data.append(-expired_cost)
        return_cost_data.append(0.0)
        expired_cost_data.append(expired_cost)
        order_data.append(0)
        avg_order_data.append(0.0)
        margin_data.append(0.0)

    return {
        "status": "ok",
        "data": {
            "date": date,
            "hours": hours,
            "bread": bread_data,
            "beverages": beverage_data,
            "revenue": revenue_data,
            "profit": profit_data,
            "non_sellable_return_cost": return_cost_data,
            "expired_cost": expired_cost_data,
            "orders": order_data,
            "avg_order": avg_order_data,
            "margin": margin_data,
        }
    }


# GET /s4/revenue/historical -- Sales query by date range + granularity
@router.get("/revenue/historical", dependencies=[Depends(require_manager)])
async def revenue_historical(start: str = None, end: str = None, granularity: str = "day", category: str = "total"):
    """Return per-product sales per period (for time-slider chart)."""
    from datetime import datetime as dt, timedelta

    db = get_db()
    cur = db.cursor()

    if end is None:
        cur.execute("SELECT MAX(order_date) FROM orders WHERE state IN ('paid','completed')")
        end = str(cur.fetchone()[0])
    if start is None:
        end_dt = dt.strptime(end, "%Y-%m-%d")
        start = (end_dt - timedelta(days=30)).strftime("%Y-%m-%d")

    group_expr, label_expr = _revenue_period_expressions(
        granularity,
        "o.order_date",
    )

    cur.execute(f"""
        SELECT oi.product_name, {label_expr} as period_label, {group_expr} as period_val,
               SUM(oi.line_total) as total_revenue, SUM(oi.line_profit) as total_profit
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_name = p.product_name
        WHERE o.order_date BETWEEN %s AND %s
        AND o.state IN ('paid','completed')
        AND (%s = 'total' OR p.category = CASE WHEN %s = 'bread' THEN 'bakery' WHEN %s = 'beverages' THEN 'beverages' END)
        GROUP BY oi.product_name, period_val, period_label
        ORDER BY period_val, oi.product_name
    """, (start, end, category, category, category))

    products = {}
    period_set = set()
    for row in cur.fetchall():
        pname = row[0]
        period = str(row[1])
        rev = round(float(row[3] or 0), 2)
        prof = round(float(row[4] or 0), 2)
        period_set.add(period)
        if pname not in products:
            products[pname] = {
                "name": pname.replace("_", " ").title(),
                "total_revenue": 0.0,
                "total_profit": 0.0,
                "periods": {}
            }
        products[pname]["total_revenue"] = round(products[pname]["total_revenue"] + rev, 2)
        products[pname]["total_profit"] = round(products[pname]["total_profit"] + prof, 2)
        products[pname]["periods"][period] = {"revenue": rev, "profit": prof}

    adjustment_rows = _get_order_adjustments_by_period(
        cur,
        start,
        end,
        granularity,
        category,
    )
    for period_label, _period_value, raw_revenue, raw_profit in adjustment_rows:
        period = str(period_label)
        revenue_adjustment = round(float(raw_revenue or 0), 2)
        profit_adjustment = round(float(raw_profit or 0), 2)
        if abs(revenue_adjustment) < 0.005 and abs(profit_adjustment) < 0.005:
            continue
        period_set.add(period)
        product = products.setdefault(
            "__order_adjustments__",
            {
                "name": "Order adjustments",
                "total_revenue": 0.0,
                "total_profit": 0.0,
                "periods": {},
            },
        )
        product["total_revenue"] = round(
            product["total_revenue"] + revenue_adjustment,
            2,
        )
        product["total_profit"] = round(
            product["total_profit"] + profit_adjustment,
            2,
        )
        product["periods"][period] = {
            "revenue": revenue_adjustment,
            "profit": profit_adjustment,
        }

    return_rows = _get_non_sellable_return_cost_by_period(
        cur,
        start,
        end,
        granularity,
        category,
    )
    expired_rows = _get_expired_cost_by_period(
        cur,
        start,
        end,
        granularity,
        category,
    )
    for product_name, period_label, _period_value, raw_cost in return_rows + expired_rows:
        period = str(period_label)
        return_cost = round(float(raw_cost or 0), 2)
        period_set.add(period)
        if product_name not in products:
            products[product_name] = {
                "name": product_name.replace("_", " ").title(),
                "total_revenue": 0.0,
                "total_profit": 0.0,
                "periods": {},
            }
        product = products[product_name]
        product["total_profit"] = round(
            product["total_profit"] - return_cost,
            2,
        )
        period_data = product["periods"].setdefault(
            period,
            {"revenue": 0.0, "profit": 0.0},
        )
        period_data["profit"] = round(period_data["profit"] - return_cost, 2)

    all_periods = sorted(period_set)
    product_list = sorted(products.values(), key=lambda x: x["total_revenue"], reverse=True)

    return {
        "status": "ok",
        "data": {
            "start": start,
            "end": end,
            "granularity": granularity,
            "category": category,
            "periods": all_periods,
            "products": product_list,
        }
    }




# ======================================================================
# S4 Raw Material Inventory Wastage API
# ======================================================================

def _get_theoretical(material_name):
    """Shared: return (theo_stock, consumed, restocked, ref_actual, ref_ts)."""
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT check_date, actual_stock, created_at FROM material_wastage_log WHERE material_name = %s ORDER BY id DESC LIMIT 1",
        (material_name,)
    )
    last = cur.fetchone()
    if last:
        ref_actual = float(last[1])
        ref_ts = str(last[2])
    else:
        cur.execute(
            "SELECT stock_quantity FROM raw_materials "
            "WHERE material_name = %s AND track_inventory = 1",
            (material_name,),
        )
        stock_row = cur.fetchone()
        if not stock_row:
            raise HTTPException(409, "Material is not stock-tracked")
        current_stock = float(stock_row[0])
        today_start = datetime.now().strftime("%Y-%m-%d") + " 00:00:00"
        ref_ts = today_start

    cur.execute(
        """
        SELECT COALESCE(SUM(quantity), 0)
        FROM material_transactions
        WHERE material_name = %s
          AND transaction_type = 'outflow'
          AND created_at >= %s
        """,
        (material_name, ref_ts),
    )
    consumed = float(cur.fetchone()[0])

    cur.execute(
        "SELECT COALESCE(SUM(quantity), 0) FROM material_transactions WHERE material_name = %s AND transaction_type IN ('inflow','restock') AND created_at >= %s",
        (material_name, ref_ts)
    )
    restocked = float(cur.fetchone()[0])

    if not last:
        ref_actual = current_stock + consumed - restocked
    theoretical_stock = ref_actual - consumed + restocked
    return theoretical_stock, consumed, restocked, ref_actual, ref_ts


@router.get("/inventory/materials", dependencies=[Depends(require_manager)])
async def get_materials():
    """Get all raw materials with current stock."""
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT material_name, stock_quantity, unit, unit_price "
        "FROM raw_materials WHERE track_inventory = 1 ORDER BY material_name"
    )
    rows = cur.fetchall()
    materials = []
    for r in rows:
        materials.append({
            "material_name": r[0],
            "stock_quantity": float(r[1]),
            "unit": r[2],
            "unit_price": float(r[3]) if r[3] else 0,
        })
    return {"status": "ok", "materials": materials}


@router.get(
    "/inventory/materials/theoretical",
    dependencies=[Depends(require_manager)],
)
async def get_materials_theoretical():
    """Get theoretical stock for each material based on last check."""
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT material_name, stock_quantity, unit, unit_price "
        "FROM raw_materials WHERE track_inventory = 1 ORDER BY material_name"
    )
    rows = cur.fetchall()
    materials = []
    for r in rows:
        mn = r[0]
        theo_stock, consumed, restocked, ref_actual, ref_ts = _get_theoretical(mn)
        cur.execute("SELECT check_date FROM material_wastage_log WHERE material_name = %s ORDER BY id DESC LIMIT 1", (mn,))
        last = cur.fetchone()
        materials.append({
            "material_name": mn,
            "current_stock": float(r[1]),
            "theoretical_stock": round(theo_stock, 3),
            "consumed_since": round(consumed, 3),
            "unit": r[2],
            "unit_price": float(r[3]) if r[3] else 0,
            "last_check_date": str(last[0]) if last else None,
        })
    return {"status": "ok", "materials": materials}


@router.post("/inventory/check", dependencies=[Depends(require_manager)])
async def inventory_check(payload: dict):
    """Submit inventory check. Compare actual vs theoretical stock."""
    check_date = payload.get("check_date", datetime.now().strftime("%Y-%m-%d"))
    counts = payload.get("counts", [])
    if not counts:
        raise HTTPException(400, "No material counts provided")

    db = get_db()
    cur = db.cursor()
    results = []

    for item in counts:
        mn = item.get("material_name", "")
        user_actual = float(item.get("actual_stock", 0))

        theo_stock, theo_consumed, restocked, ref_actual, ref_ts = _get_theoretical(mn)

        actual_consumed = round(ref_actual + restocked - user_actual, 3)
        wastage_qty = round(actual_consumed - theo_consumed, 3)

        if theo_consumed > 0.0001:
            wastage_rate = round(wastage_qty / theo_consumed, 4)
        else:
            wastage_rate = 0.0

        cur.execute(
            "INSERT INTO material_wastage_log (material_name, check_date, theoretical_stock, actual_stock, theoretical_consumed, actual_consumed, wastage_qty, wastage_rate) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (mn, check_date, round(theo_stock, 3), user_actual, round(theo_consumed, 3), actual_consumed, wastage_qty, wastage_rate)
        )
        cur.execute(
            "UPDATE raw_materials SET stock_quantity = %s "
            "WHERE material_name = %s AND track_inventory = 1",
            (user_actual, mn),
        )

        results.append({
            "material_name": mn,
            "theoretical_stock": round(theo_stock, 3),
            "actual_stock": user_actual,
            "theoretical_consumed": round(theo_consumed, 3),
            "actual_consumed": actual_consumed,
            "wastage_qty": wastage_qty,
            "wastage_rate": wastage_rate,
        })

    db.commit()
    return {"status": "ok", "check_date": check_date, "results": results}


@router.get("/inventory/check/history", dependencies=[Depends(require_manager)])
async def inventory_check_history(material_name: str = None, limit: int = 30):
    """Get material wastage log history."""
    db = get_db()
    cur = db.cursor()
    if material_name:
        cur.execute(
            """
            SELECT mw.id, mw.material_name, mw.check_date, mw.theoretical_stock, mw.actual_stock,
                   mw.theoretical_consumed, mw.actual_consumed, mw.wastage_qty, mw.wastage_rate,
                   mw.created_at, rm.unit
            FROM material_wastage_log mw
            JOIN raw_materials rm ON rm.material_name = mw.material_name
            WHERE mw.material_name = %s
            ORDER BY mw.id DESC LIMIT %s
            """,
            (material_name, limit)
        )
    else:
        cur.execute(
            """
            SELECT mw.id, mw.material_name, mw.check_date, mw.theoretical_stock, mw.actual_stock,
                   mw.theoretical_consumed, mw.actual_consumed, mw.wastage_qty, mw.wastage_rate,
                   mw.created_at, rm.unit
            FROM material_wastage_log mw
            JOIN raw_materials rm ON rm.material_name = mw.material_name
            ORDER BY mw.id DESC LIMIT %s
            """,
            (limit,)
        )
    rows = cur.fetchall()
    history = []
    for r in rows:
        history.append({
            "id": r[0],
            "material_name": r[1],
            "check_date": str(r[2]),
            "theoretical_stock": float(r[3]),
            "actual_stock": float(r[4]),
            "theoretical_consumed": float(r[5]),
            "actual_consumed": float(r[6]),
            "wastage_qty": float(r[7]),
            "wastage_rate": float(r[8]),
            "created_at": str(r[9]) if r[9] else "",
            "unit": r[10],
        })
    return {"status": "ok", "history": history}


@router.get("/inventory/dashboard", dependencies=[Depends(require_manager)])
async def inventory_dashboard(date: str = None):
    """Return bread stock + baking materials + coffee materials + BI metrics for dashboard."""
    db = get_db()
    cur = db.cursor()

    # ---- Bread finished goods (date-aware) ----
    from datetime import datetime as dt, timedelta
    if date is None:
        date = dt.now().strftime("%Y-%m-%d")

    # Get current stock as baseline
    cur.execute("""SELECT bi.product_name, bi.freshness_status, SUM(bi.quantity_remaining) as qty FROM batch_inventory bi JOIN products p ON bi.product_name = p.product_name WHERE p.category = %s GROUP BY bi.product_name, bi.freshness_status""", ("bakery",))
    current_map = {}
    for r in cur.fetchall():
        current_map[(r[0], r[1])] = int(r[2] or 0)

    # Get sales AFTER the given date (to add back for earlier dates)
    cur.execute("""SELECT oi.product_name, oi.freshness, SUM(oi.quantity) as sold FROM order_items oi JOIN orders o ON oi.order_id = o.id WHERE o.order_date > %s AND o.order_date <= %s GROUP BY oi.product_name, oi.freshness""", (date, dt.now().strftime("%Y-%m-%d")))
    addback_map = {}
    for r in cur.fetchall():
        addback_map[(r[0], r[1])] = int(r[2] or 0)

    bread_map = {}
    for (pn, status), current in current_map.items():
        addback = addback_map.get((pn, status), 0)
        remaining = current + addback
        if pn not in bread_map:
            bread_map[pn] = {"product_name": pn, "fresh_qty": 0, "day1_qty": 0, "total_qty": 0}
        if status == "Fresh":
            bread_map[pn]["fresh_qty"] += remaining
        else:
            bread_map[pn]["day1_qty"] += remaining
        bread_map[pn]["total_qty"] += remaining

    bread_stock = []
    fresh_total = 0
    day1_total = 0
    for pn in sorted(bread_map.keys()):
        b = bread_map[pn]
        fresh_total += b["fresh_qty"]
        day1_total += b["day1_qty"]
        bread_stock.append(b)

    # ---- Baking materials ----
    cur.execute("""
        SELECT material_name, category, unit, stock_quantity, reorder_point
        FROM raw_materials
        WHERE category IN ('flour','baking','dairy','sugar','packaging')
          AND track_inventory = 1
        ORDER BY category, material_name
    """)
    baking_materials = []
    for r in cur.fetchall():
        mn = r[0]
        stock = float(r[3] or 0)
        reorder = float(r[4] or 0)
        cur.execute("SELECT MAX(created_at) FROM material_transactions WHERE material_name = %s AND transaction_type = 'restock'", (mn,))
        lr = cur.fetchone()[0]
        if lr:
            cur.execute("SELECT COALESCE(SUM(quantity),0) FROM material_transactions WHERE material_name = %s AND transaction_type = 'outflow' AND created_at >= %s", (mn, lr))
            baseline = round(stock + float(cur.fetchone()[0]), 3)
        else:
            baseline = round(stock, 3)
        baking_materials.append({
            "material_name": mn, "category": r[1], "unit": r[2],
            "stock": round(stock, 3), "reorder_point": reorder,
            "baseline": baseline,
        })

    # ---- Coffee materials ----
    cur.execute("""
        SELECT material_name, category, unit, stock_quantity, reorder_point
        FROM raw_materials
        WHERE category IN ('coffee')
          AND track_inventory = 1
        ORDER BY material_name
    """)
    coffee_materials = []
    for r in cur.fetchall():
        mn = r[0]
        stock = float(r[3] or 0)
        reorder = float(r[4] or 0)
        cur.execute("SELECT MAX(created_at) FROM material_transactions WHERE material_name = %s AND transaction_type = 'restock'", (mn,))
        lr = cur.fetchone()[0]
        if lr:
            cur.execute("SELECT COALESCE(SUM(quantity),0) FROM material_transactions WHERE material_name = %s AND transaction_type = 'outflow' AND created_at >= %s", (mn, lr))
            baseline = round(stock + float(cur.fetchone()[0]), 3)
        else:
            baseline = round(stock, 3)
        coffee_materials.append({
            "material_name": mn, "category": r[1], "unit": r[2],
            "stock": round(stock, 3), "reorder_point": reorder,
            "baseline": baseline,
        })

    # ---- 1. Inventory Value ----
    # Bread finished goods value (quantity x price, Day-1 at 80%)
    cur.execute("""
        SELECT COALESCE(SUM(
            CASE WHEN bi.freshness_status = 'Day-1'
                THEN bi.quantity_remaining * p.unit_price * 0.8
                ELSE bi.quantity_remaining * p.unit_price
            END
        ), 0)
        FROM batch_inventory bi
        JOIN products p ON bi.product_name = p.product_name
    """)
    bread_value = round(float(cur.fetchone()[0]), 2)
    # Raw materials value (stock x unit_price)
    cur.execute(
        "SELECT COALESCE(SUM(stock_quantity * unit_price), 0) "
        "FROM raw_materials WHERE track_inventory = 1"
    )
    material_value = round(float(cur.fetchone()[0]), 2)
    inventory_value = round(bread_value + material_value, 2)

    # ---- 2. Fresh vs Day-1 pie (date-aware) ----
    from datetime import datetime as dt
    if date is None:
        date = dt.now().strftime("%Y-%m-%d")
    pie_data = [{"name": "Fresh", "value": fresh_total}, {"name": "Day-1", "value": day1_total}]

    # ---- 3. Stock Days Remaining ----
    cur.execute("""
        SELECT material_name, stock_quantity, unit
        FROM raw_materials
        WHERE track_inventory = 1
        ORDER BY material_name
    """)
    stock_days = []
    for r in cur.fetchall():
        mn = r[0]
        stock = float(r[1] or 0)
        # Get daily average consumption from outflow transactions (last 30 days or all available)
        cur.execute("""
            SELECT COALESCE(SUM(quantity), 0),
                   DATEDIFF(MAX(created_at), MIN(created_at)) + 1
            FROM material_transactions
            WHERE material_name = %s AND transaction_type = 'outflow'
        """, (mn,))
        crow = cur.fetchone()
        total_outflow = float(crow[0] or 0)
        date_span = int(crow[1] or 0)
        if total_outflow > 0 and date_span > 0:
            daily_avg = round(total_outflow / date_span, 3)
            days_remaining = round(stock / daily_avg, 1) if daily_avg > 0 else None
        else:
            daily_avg = None
            days_remaining = None
        stock_days.append({
            "material_name": mn,
            "stock": round(stock, 3),
            "daily_avg": daily_avg,
            "days_remaining": days_remaining,
            "unit": r[2],
        })

    # ---- 4. Material Consumption Top 5 ----
    cur.execute("""
        SELECT material_name, SUM(quantity) as total
        FROM material_transactions
        WHERE transaction_type = 'outflow'
        GROUP BY material_name
        ORDER BY total DESC
        LIMIT 5
    """)
    consumption_top5 = []
    for r in cur.fetchall():
        consumption_top5.append({
            "material_name": r[0],
            "consumed_qty": round(float(r[1]), 3),
        })

    return {
        "status": "ok",
        "bread_stock": bread_stock,
        "baking_materials": baking_materials,
        "coffee_materials": coffee_materials,
        "inventory_value": inventory_value,
        "fresh_day1_pie": pie_data,
        "fresh_total": fresh_total,
        "day1_total": day1_total,
        "stock_days": stock_days,
        "consumption_top5": consumption_top5,
    }


@router.get(
    "/inventory/stock-days-history",
    dependencies=[Depends(require_manager)],
)
async def stock_days_history(date: str = None):
    """Return historical stock days remaining for a given date.
    Computes stock position at end of the target date and daily average
    consumption up to that date."""
    from datetime import datetime as dt, timedelta
    
    db = get_db()
    cur = db.cursor()
    
    if date is None:
        date = dt.now().strftime("%Y-%m-%d")
    
    target_end = date + " 23:59:59"
    
    stock_days = []
    
    # Get all materials
    cur.execute(
        "SELECT material_name, stock_quantity, unit FROM raw_materials "
        "WHERE track_inventory = 1 ORDER BY material_name"
    )
    materials = cur.fetchall()
    
    for r in materials:
        mn = r[0]
        current_stock = float(r[1] or 0)
        unit = r[2]
        
        # Stock at target date = current_stock + outflows_after_date - inflows_after_date
        cur.execute("""
            SELECT 
                COALESCE(SUM(CASE WHEN transaction_type = 'outflow' THEN quantity ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN transaction_type = 'inflow' THEN quantity ELSE 0 END), 0)
            FROM material_transactions
            WHERE material_name = %s AND created_at > %s
        """, (mn, target_end))
        adj_row = cur.fetchone()
        outflow_after = float(adj_row[0] or 0)
        inflow_after = float(adj_row[1] or 0)
        stock_at_date = round(current_stock + outflow_after - inflow_after, 3)
        
        # Daily average consumption up to target date
        cur.execute("""
            SELECT COALESCE(SUM(quantity), 0),
                   DATEDIFF(%s, MIN(created_at)) + 1
            FROM material_transactions
            WHERE material_name = %s AND transaction_type = 'outflow'
              AND created_at <= %s
        """, (date, mn, target_end))
        crow = cur.fetchone()
        total_outflow = float(crow[0] or 0)
        date_span = int(crow[1] or 0)
        
        if total_outflow > 0 and date_span > 0:
            daily_avg = round(total_outflow / date_span, 3)
            days_remaining = round(stock_at_date / daily_avg, 1) if daily_avg > 0 else None
        else:
            daily_avg = None
            days_remaining = None
        
        stock_days.append({
            "material_name": mn,
            "stock": stock_at_date,
            "daily_avg": daily_avg,
            "days_remaining": days_remaining,
            "unit": unit,
        })
    
    cur.close()
    db.close()
    
    return {
        "status": "ok",
        "date": date,
        "stock_days": stock_days,
    }

@router.get(
    "/inventory/wastage/summary",
    dependencies=[Depends(require_manager)],
)
async def wastage_summary(date: str = ""):
    """Get latest wastage rates per material up to the selected date."""
    db = get_db()
    cur = db.cursor()
    if date:
        cur.execute("""
            SELECT m1.material_name, m1.theoretical_consumed, m1.wastage_qty, m1.wastage_rate, m1.check_date, rm.unit
            FROM material_wastage_log m1
            JOIN raw_materials rm ON rm.material_name = m1.material_name
                AND rm.track_inventory = 1
            INNER JOIN (
                SELECT mw.material_name, MAX(mw.id) as max_id
                FROM material_wastage_log mw
                WHERE mw.check_date = (
                    SELECT MAX(mw2.check_date)
                    FROM material_wastage_log mw2
                    WHERE mw2.material_name = mw.material_name
                      AND mw2.check_date <= %s
                )
                GROUP BY mw.material_name
            ) m2 ON m1.id = m2.max_id
        """, (date,))
    else:
        cur.execute("""
            SELECT m1.material_name, m1.theoretical_consumed, m1.wastage_qty, m1.wastage_rate, m1.check_date, rm.unit
            FROM material_wastage_log m1
            JOIN raw_materials rm ON rm.material_name = m1.material_name
                AND rm.track_inventory = 1
            INNER JOIN (
                SELECT material_name, MAX(id) as max_id
                FROM material_wastage_log
                GROUP BY material_name
            ) m2 ON m1.id = m2.max_id
        """)
    rows = cur.fetchall()
    summary = []
    for r in rows:
        summary.append({
            "material_name": r[0],
            "theoretical_consumed": float(r[1]),
            "wastage_qty": float(r[2]),
            "wastage_rate": float(r[3]),
            "check_date": str(r[4]),
            "unit": r[5],
        })
    return {"status": "ok", "date": date, "summary": summary}


@router.get("/inventory/restock/history", dependencies=[Depends(require_manager)])
async def inventory_restock_history(
    date: str = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    selected_date = date or datetime.now().strftime("%Y-%m-%d")
    try:
        start = datetime.strptime(selected_date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(400, "Date must use YYYY-MM-DD format") from exc
    end = start + timedelta(days=1)

    db = get_db()
    cur = db.cursor(dictionary=True)
    try:
        cur.execute(
            """
            SELECT id, material_name, quantity, unit, reference, created_at
            FROM material_transactions
            WHERE transaction_type = 'restock'
              AND created_at >= %s
              AND created_at < %s
            ORDER BY created_at, id
            LIMIT %s
            """,
            (
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"),
                limit,
            ),
        )
        records = cur.fetchall()
        latest_record_date = None
        latest_record_count = 0
        if not records:
            cur.execute(
                """
                SELECT DATE(created_at) AS latest_date, COUNT(*) AS record_count
                FROM material_transactions
                WHERE transaction_type = 'restock'
                  AND created_at < %s
                GROUP BY DATE(created_at)
                ORDER BY latest_date DESC
                LIMIT 1
                """,
                (end.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            latest = cur.fetchone()
            if latest:
                latest_record_date = str(latest.get("latest_date") or "") or None
                latest_record_count = int(latest.get("record_count") or 0)
    finally:
        cur.close()
        db.close()

    for record in records:
        created_at = record.get("created_at")
        if isinstance(created_at, datetime):
            record["created_at"] = created_at.strftime("%Y-%m-%d %H:%M:%S")
        record["quantity"] = float(record.get("quantity") or 0)
    return {
        "status": "ok",
        "date": selected_date,
        "count": len(records),
        "records": records,
        "latest_record_date": latest_record_date,
        "latest_record_count": latest_record_count,
    }


@router.post("/inventory/restock", dependencies=[Depends(require_manager)])
async def inventory_restock(payload: dict):
    """Restock raw materials. Adds quantity to stock_quantity and records transaction."""
    material_name = payload.get("material_name", "")
    raw_quantity = payload.get("quantity")
    if (
        not isinstance(material_name, str)
        or not material_name.strip()
        or isinstance(raw_quantity, bool)
        or not isinstance(raw_quantity, (int, float, Decimal))
    ):
        raise HTTPException(400, "Invalid material or quantity")
    add_quantity = Decimal(str(raw_quantity))
    if not add_quantity.is_finite() or add_quantity <= 0:
        raise HTTPException(400, "Invalid material or quantity")
    material_name = material_name.strip()
    add_qty = float(add_quantity)

    db = get_db(autocommit=False)
    cur = db.cursor()
    try:
        cur.execute(
            "SELECT stock_quantity, unit, track_inventory FROM raw_materials "
            "WHERE material_name = %s FOR UPDATE",
            (material_name,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, f"Material '{material_name}' not found")
        if not bool(row[2]):
            raise HTTPException(409, "Material is not stock-tracked")

        current = float(row[0])
        unit = row[1]
        new_stock = round(current + add_qty, 6)
        cur.execute(
            "UPDATE raw_materials SET stock_quantity = stock_quantity + %s "
            "WHERE material_name = %s",
            (add_qty, material_name),
        )
        if cur.rowcount != 1:
            raise HTTPException(409, f"Material stock changed: {material_name}")
        cur.execute(
            "INSERT INTO material_transactions "
            "(material_name, transaction_type, quantity, unit, reference) "
            "VALUES (%s,%s,%s,%s,%s)",
            (material_name, "restock", add_qty, unit, "manual_restock"),
        )
        db.commit()
        return {
            "status": "ok",
            "material_name": material_name,
            "previous_stock": round(current, 3),
            "added": add_qty,
            "new_stock": round(new_stock, 3),
            "unit": unit,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        cur.close()
        db.close()



