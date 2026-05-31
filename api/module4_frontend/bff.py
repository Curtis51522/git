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
        inventory[pn]["total_qty"] += b.get("quantity", 0)
        inventory[pn]["batches"].append(b)
        # Track "worst" freshness (for discount scoring)
        f = b.get("freshness_status", "Fresh")
        f_rank = {"Fresh": 0, "Day-1": 1, "Expired": 2}
        if f_rank.get(f, 0) > f_rank.get(inventory[pn]["min_freshness"], 0):
            inventory[pn]["min_freshness"] = f
    
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

    # Determine which breads to score:
    # If cart has breads -> recommend coffee for THOSE breads (cart-driven)
    # If cart empty or only coffee -> score all inventory (fallback)
    if cart_breads:
        target_breads = cart_breads
    else:
        target_breads = set(inventory.keys())

    all_scores = []
    max_inv = max(inv["total_qty"] for inv in inventory.values()) if inventory else 1

    for pn, inv_data in inventory.items():
        if pn not in target_breads:
            continue  # skip breads not in cart

        pairings = PAIRING_MATRIX.get(pn, {})
        freshness = inv_data.get("min_freshness", "Fresh")
        discount = get_discount_rate(freshness)
        inv_pressure = inv_data["total_qty"] / max(max_inv, 1)
        in_cart_bonus = 1.0 if pn in cart_breads else 0.7

        for coffee in COFFEE_DRINKS:
            ck = coffee["key"]

            # 1. Flavor pairing score
            flavor_score = pairings.get(ck, 0.3)

            # 2. Discount score (higher discount = better deal)
            discount_score = discount * 3.33

            # 3. Freshness score (older = more urgent)
            f_map = {"Fresh": 0.3, "Day-1": 0.8, "Expired": 1.0}
            freshness_score = f_map.get(freshness, 0.5)

            # 4. Inventory pressure
            inv_score = min(inv_pressure, 1.0)

            # 5. Order context: boost if coffee NOT already in cart
            context_score = 1.0 if ck not in cart_coffee_keys else 0.3

            # 6. Cart-relevance bonus: bread already in cart gets higher weight
            #    This ensures cart-driven recommendations rank above fallback ones
            cart_boost = 0.15 if pn in cart_breads else 0.0

            total = (
                W_FLAVOR * flavor_score +
                W_DISCOUNT * discount_score +
                W_FRESH * freshness_score +
                W_INV * inv_score +
                W_CONTEXT * context_score +
                cart_boost
            )

            bundle_price = (get_product_prices().get(pn, 5.0) * (1 - discount)) + coffee["price"]
            regular_price = get_product_prices().get(pn, 5.0) + coffee["price"]
            savings = regular_price - bundle_price

            all_scores.append({
                "product_name": pn,
                "coffee_name": coffee["name"],
                "coffee_key": ck,
                "products": f"{pn.replace('_',' ').title()} + {coffee['name']}",
                "flavor_pairing": round(flavor_score, 2),
                "discount_value": round(discount_score, 2),
                "freshness": round(freshness_score, 2),
                "inventory_pressure": round(inv_score, 2),
                "order_context_match": round(context_score, 2),
                "total_score": round(total, 3),
                "total_price": round(bundle_price, 2),
                "savings": round(savings, 2),
                "tray_color": {"Fresh": "green", "Day-1": "yellow", "Expired": "black"}.get(freshness, "green"),
                "freshness_status": freshness,
                "stock_qty": inv_data["total_qty"],
            })

    # Sort by score descending
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
    
    return {"status": "ok", "recommendations": top3, "weights": {
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
    BAKERY_KEYS = {"donut","croissant","bread_coconut","bread_roll","chiffon","croissant_chocolate"}
    for item in items:
        pn = item.get("product_name", "")
        if pn in BAKERY_KEYS:
            bakery_items.append(item)
        else:
            coffee_items.append(item)
    
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
    from datetime import datetime
    now = datetime.now()
    receipt_id = f"RCP-{now.strftime('%Y%m%d%H%M%S')}-{now.microsecond // 1000:03d}"
    
    # Store receipt in DB
    import json
    try:
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO receipts(receipt_id, items, subtotal, discount_total, total, savings) VALUES(%s,%s,%s,%s,%s,%s)",
            (receipt_id, json.dumps(receipt_items), round(subtotal,2), round(discount_total,2), round(total,2), round(savings,2))
        )
        db.commit()
    except Exception:
        pass  # receipt storage is non-critical
    
    receipt = {
        "receipt_id": receipt_id,
        "date": now.strftime("%Y-%m-%d %H:%M"),
        "items": receipt_items,
        "subtotal": round(subtotal, 2),
        "discount_total": round(discount_total, 2),
        "total": round(total, 2),
        "savings": round(savings, 2),
    }
    
    return {
        "status": status,
        "deducted": deducted,
        "errors": all_errors,
        "receipt": receipt,
        "message": f"{len(deducted)} items deducted" + 
                   (f", {len(all_errors)} items failed" if all_errors else ""),
    }