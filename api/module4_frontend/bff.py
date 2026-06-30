from fastapi import APIRouter, HTTPException, Depends, Request
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings as cfg
from db.mysql_client import get_db, q
from models.schemas import (
    LoginRequest, LoginResponse, ComboScore, UserRole,
    DeductRequest, DeductResponse,
)

router = APIRouter(prefix="/s4", tags=["Module 4 - BFF"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

COFFEE_BREAD_PAIRS = {
    "Latte": ["Croissant","Danish"],
    "Americano": ["Muffin","Donut"],
    "Cappuccino": ["Cinnamon Roll","Sourdough"],
    "Cold Brew": ["Bagel","Croissant"],
    "Espresso": ["Baguette"],
    "Flat White": ["Croissant","Muffin"],
    "Mocha": ["Donut","Cinnamon Roll"],
}


# ======================================================================
# Auth helpers
# ======================================================================
@router.post("/login")
async def login(req: LoginRequest):
    db = get_db()
    r = q(db, "users").select("*").eq("username", req.username).execute()
    if not r.data:
        raise HTTPException(401, "Invalid credentials")
    user = r.data[0]
    stored_hash = user.get("password_hash", "")
    if stored_hash == "hash123" or stored_hash == "":
        if req.password != "hash123":
            raise HTTPException(401, "Invalid credentials")
    else:
        if not pwd_context.verify(req.password, stored_hash):
            raise HTTPException(401, "Invalid credentials")
    token = jwt.encode(
        {
            "sub": user["username"],
            "role": user["role"],
            "exp": datetime.utcnow() + timedelta(minutes=cfg.JWT_EXPIRE_MINUTES),
        },
        cfg.JWT_SECRET,
        algorithm=cfg.JWT_ALGORITHM,
    )
    return LoginResponse(
        access_token=token, username=user["username"], role=user["role"]
    )


async def get_current_user(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        raise HTTPException(401, "Missing token")
    try:
        return jwt.decode(token, cfg.JWT_SECRET, algorithms=[cfg.JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(401, "Invalid token")


async def require_manager(user=Depends(get_current_user)):
    if user.get("role") != "manager":
        raise HTTPException(403, "Manager only")
    return user


# ======================================================================
# POST /s4/combo -- Product pairing recommendations
# ======================================================================
@router.post("/combo")
async def get_combo(order: dict):
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
    BAKERY_PRODUCTS = {'donut','croissant','bread_coconut','bread_roll','chiffon','croissant_chocolate'}
    
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
    
    COFFEE_DRINKS = [
        {"name": "Latte", "key": "latte", "price": 8.50},
        {"name": "Americano", "key": "americano", "price": 6.50},
        {"name": "Cappuccino", "key": "cappuccino", "price": 9.00},
        {"name": "Cold Brew", "key": "cold_brew", "price": 10.00},
        {"name": "Iced Americano", "key": "iced_americano", "price": 7.20},
        {"name": "Mocha", "key": "mocha", "price": 10.50},
    ]
    
    # Cart context: which breads does the customer already have?
    order_items = order.get("items", [])
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
    strict_mode = bool(cart_breads and cart_coffee_keys)

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
                context_score = 1.0 if ck not in cart_coffee_keys else 0.3
                cart_boost = 0.15 if pn in cart_breads else 0.0
                total = (W_FLAVOR*flavor_score + W_DISCOUNT*discount_score + W_FRESH*freshness_score + W_INV*inv_score + W_CONTEXT*context_score + cart_boost)
                bundle_price = (get_product_prices().get(pn, 5.0)*(1-discount)) + coffee["price"]
                regular_price = get_product_prices().get(pn, 5.0) + coffee["price"]
                savings = regular_price - bundle_price
                all_scores.append({
                    "product_name": pn, "coffee_name": coffee["name"], "coffee_key": ck,
                    "products": f"{pn.replace('_',' ').title()} + {coffee['name']}",
                    "direction": "bread_to_coffee",
                    "total_score": round(total, 3), "total_price": round(bundle_price, 2),
                    "savings": round(savings, 2), "freshness_status": freshness,
                    "stock_qty": inv_data["total_qty"],
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
                context_score = 1.0 if pn not in cart_breads else 0.3
                coffee_boost = 0.15 if ck in cart_coffee_keys else 0.0
                total = (W_FLAVOR*flavor_score + W_DISCOUNT*discount_score + W_FRESH*freshness_score + W_INV*inv_score + W_CONTEXT*context_score + coffee_boost)
                bundle_price = (get_product_prices().get(pn, 5.0)*(1-discount)) + coffee["price"]
                regular_price = get_product_prices().get(pn, 5.0) + coffee["price"]
                savings = regular_price - bundle_price
                all_scores.append({
                    "product_name": pn, "coffee_name": coffee["name"], "coffee_key": ck,
                    "products": f"{pn.replace('_',' ').title()} + {coffee['name']}",
                    "direction": "coffee_to_bread",
                    "total_score": round(total, 3), "total_price": round(bundle_price, 2),
                    "savings": round(savings, 2), "freshness_status": freshness,
                    "stock_qty": inv_data["total_qty"],
                })

    # Sort by score descending
    # Sort by savings when cart has only coffee (different bread discounts matter), otherwise by total_score
    if cart_coffee_keys and not cart_breads:
        all_scores.sort(key=lambda x: x["savings"], reverse=True)
    else:
        all_scores.sort(key=lambda x: x["total_score"], reverse=True)

    # Pick top-3: prefer diverse breads, but if fewer than 3 unique breads
    # available (e.g. cart only has chiffon), fill with next-best coffees
    top3 = []
    seen_products = set()

    # Pass 1: grab one per unique bread
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
    _product_costs_cache = None
    return dict(_DEFAULT_COSTS)

# Use get_product_prices() directly; this module-level reference is kept for backward compat
# but will only be populated after first successful DB read
PRODUCT_PRICES = {}





# ======================================================================

# ======================================================================
# GET /s4/products -- Return product prices from DB
# ======================================================================
@router.get("/products")
async def list_products():
    """Return all product prices from the database."""
    try:
        db = get_db()
        r = q(db, "products").select("*").eq("category", "bakery").execute()
        if r.data:
            products = []
            for row in r.data:
                products.append({
                    "product_name": row["product_name"],
                    "unit_price": float(row.get("selling_price", row.get("unit_price", 0))),
                    "cost_price": float(row.get("cost_price", 0)),
                })
            return {"status": "ok", "products": products}
    except Exception:
        pass
    # Fallback: return only bakery from cached prices
    bakery = {"donut","croissant","bread_coconut","bread_roll","chiffon","croissant_chocolate"}
    prices = get_product_prices()
    costs = get_product_costs()
    products = []
    for name, price in prices.items():
        if name in bakery:
            products.append({
                "product_name": name,
                "unit_price": float(price) if price else 0,
                "cost_price": float(costs.get(name, 0)),
            })
    return {"status": "ok", "products": products}

# POST /s4/checkout/complete -- Complete payment + deduct inventory
# ======================================================================
@router.post("/checkout/complete")
async def checkout_complete(payload: dict):
    """Process checkout: deduct inventory via FIFO, apply freshness discounts, generate receipt."""
    items = payload.get("items", [])
    if not items:
        raise HTTPException(400, "No items in cart")

    db = get_db()
    from api.module1_yolo import deduct_inventory
    from models.schemas import DeductRequest

    # Split items: bakery (deduct from inventory) vs coffee (no inventory limit)
    bakery_items = []
    coffee_items = []
    BAKERY_KEYS = {"apple_pie","bagel","baguette","bread_coconut","bread_roll","brioche","brownie","chiffon","chocolate_cake","chocopie","cookie","cornbread","cream_horn","croissant","croissant_chocolate","donut","eggtart","flatbread","macaron","mantequilla","melon_bread","muffin","pancake","pandesal","pizza_bread","pullman","soboru_bread","sourdough","stickbread","tostada"}
    COFFEE_KEYS = {"latte","americano","cappuccino","mocha","espresso","flat_white","caramel_macchiato","cold_brew","hot_chocolate","matcha_latte","milk_tea","chai_latte","earl_grey","english_breakfast","lemonade"}
    unknown_items = []
    for item in items:
        pn = item.get("product_name", "")
        if pn in BAKERY_KEYS:
            bakery_items.append(item)
        elif pn in COFFEE_KEYS:
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
    # Deduct bakery items via FIFO
    result = None
    if bakery_items:
        req = DeductRequest(items=bakery_items)
        result = await deduct_inventory(req)
    
    # Record coffee items as direct outflow transactions (no inventory limit)
    coffee_deducted = []
    for item in coffee_items:
        pn = item.get("product_name", "")
        qty = item.get("quantity", 1)
        price = get_product_prices().get(pn, 8.0)  # coffee price from products or default
        q(db, "inventory_transactions").insert({
            "transaction_type": "outflow",
            "batch_id": None,  # coffee has no batch inventory
            "product_name": pn,
            "quantity": qty,
            "unit_price": price,
            "discount_applied": 0,
            "freshness_status": "Fresh",
            "beverage_size": item.get("size", None),
            "beverage_temp": item.get("temperature", None),
            "beverage_sweetness": item.get("sugar", None),
            "beverage_ice": item.get("ice_level", None),
        }).execute()
        coffee_deducted.append({
            "product_name": pn,
            "batch_id": None,
            "quantity_deducted": qty,
            "remaining_after": 0,
        })
    
    # Merge results
    deducted = (result.deducted if result else []) + coffee_deducted
    all_errors = (result.errors if result else [])
    status = result.status if result else "ok"

    # If deduction has errors, do NOT create an order - return errors to frontend
    if all_errors:
        return {
            "status": status,
            "deducted": deducted,
            "errors": all_errors,
            "receipt": None,
            "message": f"{len(deducted)} items deducted, {len(all_errors)} items failed",
        }

    # ---- Build receipt ----
    prices = get_product_prices()
    costs = get_product_costs()
    from api.freshness_service import get_discount_rate
    
    receipt_items = []
    subtotal = 0.0
    discount_total = 0.0
    
    for item in items:
        pn = item.get("product_name", "")
        qty = item.get("quantity", 1)
        freshness = item.get("freshness", "Fresh")
        unit_price = prices.get(pn, 5.0)
        size = item.get("size", "Regular")
        if size == "Large" and pn in COFFEE_KEYS:
            unit_price += 3.0
        discount_rate = get_discount_rate(freshness) if freshness == "Day-1" else 0.0
        line_total = unit_price * qty
        line_discount = line_total * discount_rate
        line_final = line_total - line_discount
        
        receipt_items.append({
            "product_name": pn,
            "quantity": qty,
            "unit_price": round(unit_price, 2),
            "discount_pct": int(discount_rate * 100),
            "discount_amount": round(line_discount, 2),
            "line_total": round(line_final, 2),
        })
        subtotal += line_total
        discount_total += line_discount
    
    total = subtotal - discount_total
    savings = discount_total
    
    # Generate receipt ID
        
    try:
            # ---- Record to orders / order_items / payments tables ----
        now = datetime.now()
        payment_method = payload.get("payment_method", "cash")
        cash_received = payload.get("cash_received", None)
    
        # Packaging fee for takeaway
        dine_type = payload.get("dine_type", "dine_in")
        packaging_fee = 0.30 if dine_type == "takeaway" else 0.0
        if packaging_fee > 0:
            total += packaging_fee
    
    # Build product cost lookup
        all_product_names = [it.get("product_name","") for it in items]
        placeholders = ",".join(["%s"] * len(all_product_names))
        cur = db.cursor()
        cur.execute(f"SELECT product_name, material_cost, wastage_pct FROM products WHERE product_name IN ({placeholders})", all_product_names)
        rows = cur.fetchall(); cost_map = {r[0]: float(r[1]) for r in rows}; wastage_map = {r[0]: float(r[2]) if r[2] else 0.03 for r in rows}
    
        # Calculate order totals
        order_subtotal = subtotal
        order_discount = discount_total
        order_total = total
        order_profit = 0.0
        order_cost = 0.0
        order_item_count = 0
    
        for item in items:
            pn = item.get("product_name","")
            qty = item.get("quantity", 1)
            uprice = prices.get(pn, 5.0)
            mat_cost = cost_map.get(pn, uprice * 0.30)
            wastage_pct = wastage_map.get(pn, 0.03)
            actual_cost = mat_cost * (1 + wastage_pct)
            freshness = item.get("freshness", "Fresh")
            disc_rate = get_discount_rate(freshness) if freshness == "Day-1" else 0.0
            line_total = uprice * qty
            line_disc = line_total * disc_rate
            line_final = line_total - line_disc
            line_profit_raw = line_final - (actual_cost * qty)
            order_cost += actual_cost * qty
            order_profit += line_profit_raw
            order_item_count += qty
    
        # Recalculate profit using final total (includes Top-3 dynamic discount)
        if order_total > 0:
            discount_ratio = order_total / order_subtotal if order_subtotal > 0 else 1.0
        else:
            discount_ratio = 1.0
        order_profit = order_total - (order_cost * discount_ratio)

        receipt_id = f"RCP-{now.strftime('%Y%m%d%H%M%S')}-{now.microsecond // 1000:03d}"

        # INSERT orders
        cur.execute(
            "INSERT INTO orders (ticket_id, order_date, order_time, subtotal, discount_total, total_amount, total_profit, item_count, state, dine_type) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (receipt_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), round(order_subtotal,2), round(order_discount,2), round(order_total,2), round(order_profit,2), order_item_count, "paid", dine_type)
        )
        order_id = cur.lastrowid
    
        # INSERT order_items
        for item in items:
            pn = item.get("product_name","")
            qty = item.get("quantity", 1)
            uprice = prices.get(pn, 5.0)
            mat_cost = cost_map.get(pn, uprice * 0.30)
            wastage_pct = wastage_map.get(pn, 0.03)
            actual_cost = mat_cost * (1 + wastage_pct)
            freshness = item.get("freshness", "Fresh")
            disc_rate = get_discount_rate(freshness) if freshness == "Day-1" else 0.0
            line_total = uprice * qty
            line_disc = line_total * disc_rate
            line_final = line_total - line_disc
            line_profit = line_final - (actual_cost * qty)
        
            coffee_temp = item.get("temperature", None)
            coffee_ice = item.get("ice_level", None)
            coffee_sugar = item.get("sugar", None)
        
            cur.execute(
                "INSERT INTO order_items (order_id, product_name, quantity, unit_price, discount_rate, line_total, line_profit, freshness, coffee_temp, coffee_ice, coffee_sugar) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (order_id, pn, qty, uprice, disc_rate, round(line_final,2), round(line_profit,2), freshness, coffee_temp, coffee_ice, coffee_sugar)
            )

            # Deduct raw materials
            cur.execute(
                "SELECT material_name, quantity_per_unit FROM product_recipes WHERE product_name = %s",
                (pn,)
            )
            for mat_row in cur.fetchall():
                mat_name = mat_row[0]
                used_qty = round(float(mat_row[1]) * qty, 6)
                actual_used_qty = round(used_qty * (1 + wastage_pct), 6)
                cur.execute(
                    "UPDATE raw_materials SET stock_quantity = stock_quantity - %s WHERE material_name = %s",
                    (actual_used_qty, mat_name)
                )
                cur.execute(
                    "INSERT INTO material_transactions (material_name, transaction_type, quantity, unit, reference) VALUES (%s,%s,%s,%s,%s)",
                    (mat_name, 'outflow', actual_used_qty, 'kg', receipt_id)
                )

        # Deduct packaging materials for takeaway
        if packaging_fee > 0:
            cur.execute(
                "UPDATE raw_materials SET stock_quantity = stock_quantity - 1 WHERE material_name = %s",
                ("Packaging Box",)
            )
            cur.execute(
                "INSERT INTO material_transactions (material_name, transaction_type, quantity, unit, reference) VALUES (%s,%s,%s,%s,%s)",
                ("Packaging Box", "outflow", 1, "pcs", receipt_id)
            )
            cur.execute(
                "UPDATE raw_materials SET stock_quantity = stock_quantity - 1 WHERE material_name = %s",
                ("Packaging Bag",)
            )
            cur.execute(
                "INSERT INTO material_transactions (material_name, transaction_type, quantity, unit, reference) VALUES (%s,%s,%s,%s,%s)",
                ("Packaging Bag", "outflow", 1, "pcs", receipt_id)
            )
        # Deduct cup per drink based on size
        for item in items:
            pn = item.get("product_name", "")
            if pn in COFFEE_KEYS:
                qty = item.get("quantity", 1)
                drink_size = item.get("size", "regular")
                cup_name = "Cup Large" if drink_size == "large" else "Cup Regular"
                cur.execute(
                    "UPDATE raw_materials SET stock_quantity = stock_quantity - %s WHERE material_name = %s",
                    (qty, cup_name)
                )
                cur.execute(
                    "INSERT INTO material_transactions (material_name, transaction_type, quantity, unit, reference) VALUES (%s,%s,%s,%s,%s)",
                    (cup_name, "outflow", qty, "pcs", receipt_id)
                )
    
        # INSERT payments
        cur.execute(
            "INSERT INTO payments (order_id, amount, payment_method, payment_date) VALUES (%s,%s,%s,%s)",
            (order_id, round(order_total,2), payment_method, now.strftime("%Y-%m-%d"))
        )
        db.commit()
        

        # Add packaging fee to receipt if applicable
        if packaging_fee > 0:
            receipt_items.append({
                "product_name": "Packaging (Takeaway)",
                "quantity": 1,
                "unit_price": round(packaging_fee, 2),
                "discount_pct": 0,
                "discount_amount": 0,
                "line_total": round(packaging_fee, 2),
            })
        
        receipt = {
            "receipt_id": receipt_id,
            "date": now.strftime("%Y-%m-%d %H:%M"),
            "items": receipt_items,
            "subtotal": round(subtotal, 2),
            "discount_total": round(discount_total, 2),
            "total": round(total, 2),
            "savings": round(savings, 2),
            "order_id": order_id,
        }

        return {
            "status": status,
            "deducted": deducted,
            "errors": all_errors,
            "receipt": receipt,
            "message": f"{len(deducted)} items deducted" + (f", {len(all_errors)} items failed" if all_errors else ""),
        }

    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "deducted": deducted,
            "errors": [f"Database write failed: {str(e)}"],
            "receipt": None,
            "message": f"Checkout failed: {str(e)}",
        }
# GET /s4/revenue/daily -- Revenue dashboard data from MySQL
# ======================================================================
# ======================================================================
# GET /s4/orders/today -- List today's paid orders for refund
# ======================================================================
@router.get("/orders/today")
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
@router.get("/orders/receipt")
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
async def refund_order(payload: dict):
    """Refund an order: reverse inventory deductions, restock materials, mark refunded."""
    ticket_id = payload.get("ticket_id", "")
    if not ticket_id:
        raise HTTPException(400, "ticket_id required")

    db = get_db()
    cur = db.cursor()

    # Find the order
    cur.execute("SELECT id, state, dine_type FROM orders WHERE ticket_id = %s", (ticket_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(404, f"Order {ticket_id} not found")
    order_id, state, dine_type = row
    if state == "refunded":
        raise HTTPException(400, "Order already refunded")
    if state != "paid":
        raise HTTPException(400, f"Cannot refund order in state: {state}")

    # Get order items
    cur.execute("SELECT product_name, quantity, freshness, coffee_size FROM order_items WHERE order_id = %s", (order_id,))
    items = cur.fetchall()

    COFFEE_KEYS = {"latte","americano","cappuccino","mocha","espresso","flat_white","caramel_macchiato","cold_brew","hot_chocolate","matcha_latte","milk_tea","chai_latte","earl_grey","english_breakfast","lemonade"}

    for item in items:
        pn, qty, freshness, coffee_size = item
        if pn in COFFEE_KEYS:
            # Restock cup
            cup_name = "Cup Large" if (coffee_size or "").lower() == "large" else "Cup Regular"
            cur.execute(
                "UPDATE raw_materials SET stock_quantity = stock_quantity + %s WHERE material_name = %s",
                (qty, cup_name)
            )
            cur.execute(
                "INSERT INTO material_transactions (material_name, transaction_type, quantity, unit, reference) VALUES (%s,%s,%s,%s,%s)",
                (cup_name, "refund", qty, "pcs", ticket_id)
            )
        else:
            # Bakery item: add back via inventory_transactions inflow
            cur.execute(
                "INSERT INTO inventory_transactions (transaction_type, product_name, quantity, freshness_status, receipt_id) VALUES (%s,%s,%s,%s,%s)",
                ("inflow", pn, qty, freshness or "Fresh", ticket_id)
            )

    # Restock packaging if takeaway
    if dine_type == "takeaway":
        for pkg_name in ["Packaging Box", "Packaging Bag"]:
            cur.execute(
                "UPDATE raw_materials SET stock_quantity = stock_quantity + 1 WHERE material_name = %s",
                (pkg_name,)
            )
            cur.execute(
                "INSERT INTO material_transactions (material_name, transaction_type, quantity, unit, reference) VALUES (%s,%s,%s,%s,%s)",
                (pkg_name, "refund", 1, "pcs", ticket_id)
            )

    # Mark order refunded
    cur.execute("UPDATE orders SET state = 'refunded' WHERE id = %s", (order_id,))
    db.commit()
    cur.close()

    return {"status": "ok", "message": f"Order {ticket_id} refunded", "items_restored": len(items)}

# ======================================================================
@router.get("/revenue/daily")
async def revenue_daily(date: str = None):
    """Return revenue dashboard data from MySQL orders/order_items/products tables."""
    from datetime import datetime as dt, timedelta
    
    db = get_db()
    cur = db.cursor()
    
    # Default to latest order date
    if date is None:
        cur.execute("SELECT MAX(order_date) FROM orders")
        date = str(cur.fetchone()[0])
    
    # Today KPIs
    cur.execute("""
        SELECT COUNT(*) as orders, SUM(total_amount) as revenue, SUM(total_profit) as profit, COALESCE(SUM(discount_total),0) as discount
        FROM orders WHERE order_date = %s
    """, (date,))
    row = cur.fetchone()
    if not row or not row[0]:
        return {"status": "ok", "data": None, "message": f"No sales data for {date}"}
    
    today_orders = int(row[0])
    today_revenue = round(float(row[1] or 0), 2)
    today_profit = round(float(row[2] or 0), 2)
    avg_order = round(today_revenue / today_orders, 2) if today_orders else 0
    today_discount = round(float(row[3] or 0), 2)
    
    # Profit margin
    profit_margin = round(today_profit / today_revenue * 100, 1) if today_revenue else 0
    
    # MTD (Month-to-Date cumulative)
    month_start = date[:8] + "01"
    cur.execute("""
        SELECT COALESCE(SUM(total_amount),0), COALESCE(SUM(total_profit),0), COUNT(*)
        FROM orders WHERE order_date >= %s AND order_date <= %s
    """, (month_start, date))
    mtd_row = cur.fetchone()
    mtd_revenue = round(float(mtd_row[0] or 0), 2)
    mtd_profit = round(float(mtd_row[1] or 0), 2)
    mtd_orders = int(mtd_row[2] or 0)
    
    # Yesterday comparison
    yesterday = (dt.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    cur.execute("""
        SELECT COUNT(*) as orders, SUM(total_amount) as revenue, SUM(total_profit) as profit, COALESCE(SUM(discount_total),0) as discount
        FROM orders WHERE order_date = %s
    """, (yesterday,))
    yrow = cur.fetchone()
    if yrow and yrow[0]:
        y_orders = int(yrow[0])
        y_revenue = round(float(yrow[1] or 0), 2)
        y_profit = round(float(yrow[2] or 0), 2)
        y_avg = round(y_revenue / y_orders, 2) if y_orders else 0
        rev_change = round((today_revenue - y_revenue) / y_revenue * 100, 1) if y_revenue else 0
        prof_change = round((today_profit - y_profit) / y_profit * 100, 1) if y_profit else 0
        ord_change = round((today_orders - y_orders) / y_orders * 100, 1) if y_orders else 0
        avg_change = round((avg_order - y_avg) / y_avg * 100, 1) if y_avg else 0
    else:
        rev_change = prof_change = ord_change = avg_change = 0
    
        # Payment breakdown (real data from payments table)
    cur.execute("""
        SELECT p.payment_method, COUNT(*) as cnt
        FROM payments p JOIN orders o ON p.order_id = o.id
        WHERE o.order_date = %s AND o.state IN ('paid','draft')
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
    
    # Category breakdown (bread vs beverages)
    cur.execute("""
        SELECT p.category, SUM(oi.line_total) as revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_name = p.product_name
        WHERE o.order_date = %s
        GROUP BY p.category
    """, (date,))
    cat_data = {"Bread": 0, "Beverages": 0}
    for crow in cur.fetchall():
        cat_key = "Bread" if crow[0] == "bakery" else "Coffee"
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
        WHERE o.order_date = %s AND oi.product_name NOT IN ({beverage_placeholders})
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
        WHERE o.order_date = %s AND oi.product_name IN ({beverage_placeholders})
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
            WHERE o.order_date = %s
            GROUP BY p.category
        """, (d,))
        day_cat = {"bakery": 0, "beverages": 0}
        for crow in cur.fetchall():
            day_cat[crow[0]] = round(float(crow[1]), 2)
        trend_bread.append(day_cat["bakery"])
        trend_beverages.append(day_cat["beverages"])
        cur.execute("SELECT COUNT(*), COALESCE(SUM(total_amount),0) FROM orders WHERE order_date = %s", (d,))
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
            "today_discount": today_discount,
            "profit_margin": profit_margin,
            "mtd_revenue": mtd_revenue,
            "mtd_profit": mtd_profit,
            "mtd_orders": mtd_orders,
            "revenue_change": rev_change,
            "profit_change": prof_change,
            "orders_change": ord_change,
            "avg_change": avg_change,
            "payment": payment,
            "category": cat_data,
            "trend": {"dates": trend_dates, "bread": trend_bread, "beverages": trend_beverages, "orders": trend_orders, "avg_order": trend_avg},
            "bread_ranking": bread_ranking,
            "beverage_ranking": beverage_ranking,
        }
    }


# GET /s4/revenue/hourly -- Hourly breakdown of bread vs beverages sales
@router.get("/revenue/hourly")
async def revenue_hourly(date: str = None):
    """Return hourly sales breakdown (bread vs beverages) for a given date."""
    from datetime import datetime as dt

    db = get_db()
    cur = db.cursor()

    if date is None:
        cur.execute("SELECT MAX(order_date) FROM orders")
        date = str(cur.fetchone()[0])

    cur.execute("""
        SELECT HOUR(o.order_time) as hr, p.category, SUM(oi.line_total) as revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_name = p.product_name
        WHERE o.order_date = %s
        GROUP BY hr, p.category
        ORDER BY hr
    """, (date,))
    rows = [r for r in cur]
    if rows:
        all_hours = sorted(set(int(r[0]) for r in rows))
        raw_min = min(all_hours)
        raw_max = max(all_hours)
        min_hr = max(6, raw_min - 1)
        max_hr = min(23, raw_max + 1)
        if min_hr >= max_hr:
            min_hr = 6
            max_hr = 22
    else:
        cur.execute("""SELECT COALESCE(MIN(HOUR(order_time)), 8), COALESCE(MAX(HOUR(order_time)), 21) FROM orders WHERE order_date >= DATE_SUB(%s, INTERVAL 30 DAY)""", (date,))
        range_row = cur.fetchone()
        min_hr = max(6, int(range_row[0] or 8) - 1)
        max_hr = min(23, int(range_row[1] or 21) + 1)

    num_hours = max_hr - min_hr + 1
    hours = [f"{h:02d}:00" for h in range(min_hr, max_hr + 1)]
    bread_data = [0.0] * num_hours
    beverage_data = [0.0] * num_hours

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

    return {
        "status": "ok",
        "data": {
            "date": date,
            "hours": hours,
            "bread": bread_data,
            "beverages": beverage_data,
        }
    }


# GET /s4/revenue/historical -- Sales query by date range + granularity
@router.get("/revenue/historical")
async def revenue_historical(start: str = None, end: str = None, granularity: str = "day", category: str = "total"):
    """Return per-product sales per period (for time-slider chart)."""
    from datetime import datetime as dt, timedelta

    db = get_db()
    cur = db.cursor()

    if end is None:
        cur.execute("SELECT MAX(order_date) FROM orders")
        end = str(cur.fetchone()[0])
    if start is None:
        end_dt = dt.strptime(end, "%Y-%m-%d")
        start = (end_dt - timedelta(days=30)).strftime("%Y-%m-%d")

    if granularity == "week":
        group_expr = "YEARWEEK(o.order_date, 1)"
        label_expr = "CONCAT(YEAR(o.order_date), '-W', LPAD(WEEK(o.order_date, 1), 2, '0'))"
    elif granularity == "month":
        group_expr = "DATE_FORMAT(o.order_date, '%Y-%m')"
        label_expr = group_expr
    elif granularity == "year":
        group_expr = "YEAR(o.order_date)"
        label_expr = "YEAR(o.order_date)"
    else:
        group_expr = "o.order_date"
        label_expr = "o.order_date"

    cur.execute(f"""
        SELECT oi.product_name, {label_expr} as period_label, {group_expr} as period_val,
               SUM(oi.line_total) as total_revenue, SUM(oi.line_profit) as total_profit
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_name = p.product_name
        WHERE o.order_date BETWEEN %s AND %s
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
        cur.execute("SELECT stock_quantity FROM raw_materials WHERE material_name = %s", (material_name,))
        stock = float(cur.fetchone()[0])
        today_start = datetime.now().strftime("%Y-%m-%d") + " 00:00:00"
        cur.execute(
            "SELECT COALESCE(SUM(pr.quantity_per_unit * oi.quantity), 0) FROM order_items oi JOIN orders o ON oi.order_id = o.id JOIN product_recipes pr ON oi.product_name = pr.product_name AND pr.material_name = %s WHERE CONCAT(o.order_date, ' ', o.order_time) >= %s",
            (material_name, today_start)
        )
        consumed_today = float(cur.fetchone()[0])
        ref_actual = stock + consumed_today
        cur.execute(
            "INSERT INTO material_wastage_log (material_name, check_date, theoretical_stock, actual_stock, theoretical_consumed, actual_consumed, wastage_qty, wastage_rate) VALUES (%s,%s,%s,%s,0,0,0,0)",
            (material_name, datetime.now().strftime("%Y-%m-%d"), stock, stock)
        )
        db.commit()
        ref_ts = today_start

    cur.execute("""
        SELECT COALESCE(SUM(pr.quantity_per_unit * oi.quantity), 0)
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN product_recipes pr ON oi.product_name = pr.product_name AND pr.material_name = %s
        WHERE CONCAT(o.order_date, ' ', o.order_time) >= %s
    """, (material_name, ref_ts))
    consumed = float(cur.fetchone()[0])

    cur.execute(
        "SELECT COALESCE(SUM(quantity), 0) FROM material_transactions WHERE material_name = %s AND transaction_type IN ('inflow','restock') AND created_at >= %s",
        (material_name, ref_ts)
    )
    restocked = float(cur.fetchone()[0])

    theoretical_stock = ref_actual - consumed + restocked
    return theoretical_stock, consumed, restocked, ref_actual, ref_ts


@router.get("/inventory/materials")
async def get_materials():
    """Get all raw materials with current stock."""
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT material_name, stock_quantity, unit, unit_price FROM raw_materials ORDER BY material_name")
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


@router.get("/inventory/materials/theoretical")
async def get_materials_theoretical():
    """Get theoretical stock for each material based on last check."""
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT material_name, stock_quantity, unit, unit_price FROM raw_materials ORDER BY material_name")
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


@router.post("/inventory/check")
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
        cur.execute("UPDATE raw_materials SET stock_quantity = %s WHERE material_name = %s", (user_actual, mn))

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


@router.get("/inventory/check/history")
async def inventory_check_history(material_name: str = None, limit: int = 30):
    """Get material wastage log history."""
    db = get_db()
    cur = db.cursor()
    if material_name:
        cur.execute(
            "SELECT id, material_name, check_date, theoretical_stock, actual_stock, theoretical_consumed, actual_consumed, wastage_qty, wastage_rate, created_at FROM material_wastage_log WHERE material_name = %s ORDER BY id DESC LIMIT %s",
            (material_name, limit)
        )
    else:
        cur.execute(
            "SELECT id, material_name, check_date, theoretical_stock, actual_stock, theoretical_consumed, actual_consumed, wastage_qty, wastage_rate, created_at FROM material_wastage_log ORDER BY id DESC LIMIT %s",
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
        })
    return {"status": "ok", "history": history}


@router.get("/inventory/dashboard")
async def inventory_dashboard(date: str = None):
    """Return bread stock + baking materials + coffee materials + BI metrics for dashboard."""
    db = get_db()
    cur = db.cursor()

    # ---- Bread finished goods (date-aware) ----
    from datetime import datetime as dt, timedelta
    if date is None:
        date = dt.now().strftime("%Y-%m-%d")

    # Get current stock as baseline
    cur.execute("""SELECT product_name, freshness_status, SUM(quantity_remaining) as qty FROM batch_inventory GROUP BY product_name, freshness_status""")
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
    cur.execute("SELECT COALESCE(SUM(stock_quantity * unit_price), 0) FROM raw_materials")
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

@router.get("/inventory/wastage/summary")
async def wastage_summary():
    """Get latest wastage rates per material."""
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT m1.material_name, m1.theoretical_consumed, m1.wastage_qty, m1.wastage_rate, m1.check_date
        FROM material_wastage_log m1
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
        })
    return {"status": "ok", "summary": summary}
@router.post("/inventory/restock")
async def inventory_restock(payload: dict):
    """Restock raw materials. Adds quantity to stock_quantity and records transaction."""
    material_name = payload.get("material_name", "")
    add_qty = float(payload.get("quantity", 0))
    if not material_name or add_qty <= 0:
        raise HTTPException(400, "Invalid material or quantity")

    db = get_db()
    cur = db.cursor()

    # Get current stock
    cur.execute("SELECT stock_quantity, unit FROM raw_materials WHERE material_name = %s", (material_name,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(404, f"Material '{material_name}' not found")

    current = float(row[0])
    unit = row[1]
    new_stock = round(current + add_qty, 6)

    cur.execute("UPDATE raw_materials SET stock_quantity = %s WHERE material_name = %s", (new_stock, material_name))
    cur.execute(
        "INSERT INTO material_transactions (material_name, transaction_type, quantity, unit, reference) VALUES (%s,%s,%s,%s,%s)",
        (material_name, "restock", add_qty, unit, "manual_restock")
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



