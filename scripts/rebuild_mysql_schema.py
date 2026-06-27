#!/usr/bin/env python
"""Rebuild MySQL tables per schema_v2 + Odoo reference, then import bakary_sales_raw.csv."""
import pandas as pd
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.mysql_client import get_db
from s3_scheduling.scheduler import PRODUCT_RECIPES, _DEFAULT_RECIPE

INGREDIENT_PRICES = {
    "flour_g": 0.008, "butter_g": 0.060, "sugar_g": 0.006,
    "egg_whole_g": 0.020, "egg_yolk_g": 0.030, "egg_white_g": 0.015,
    "milk_ml": 0.012, "chocolate_g": 0.080,
}

DRINK_COSTS = {
    "latte": 2.50, "americano": 1.80, "cappuccino": 2.80,
    "mocha": 3.20, "espresso": 1.50, "flat_white": 2.60,
    "caramel_macchiato": 3.00, "cold_brew": 3.00,
    "hot_chocolate": 2.00, "matcha_latte": 3.50,
    "milk_tea": 1.80, "chai_latte": 2.20,
    "earl_grey": 0.80, "english_breakfast": 0.80, "lemonade": 0.60,
}

COFFEE = {"latte","americano","cappuccino","mocha","espresso","flat_white",
          "caramel_macchiato","cold_brew","hot_chocolate","matcha_latte",
          "milk_tea","chai_latte","earl_grey","english_breakfast","lemonade"}

def calc_bread_cost(pname):
    recipe = PRODUCT_RECIPES.get(pname, _DEFAULT_RECIPE)
    return round(sum(recipe.get(k,0)*v for k,v in INGREDIENT_PRICES.items()), 2)

def main():
    db = get_db()
    cur = db.cursor()

    df = pd.read_csv("data/bakery_sales_raw.csv")
    print(f"Loaded {len(df)} rows, {df['ticket_id'].nunique()} tickets")

    # 1. Drop old tables
    print("Dropping old tables...")
    for t in ["order_items", "payments", "orders", "products"]:
        cur.execute(f"DROP TABLE IF EXISTS {t}")

    # 2. Create products
    print("Creating products...")
    cur.execute("""
        CREATE TABLE products (
            id INT AUTO_INCREMENT PRIMARY KEY,
            product_name VARCHAR(50) UNIQUE NOT NULL,
            category VARCHAR(20) NOT NULL CHECK (category IN ('bakery','coffee')),
            unit_price DECIMAL(8,2) NOT NULL,
            material_cost DECIMAL(8,2) DEFAULT 0,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    avg_prices = df.groupby("product_name")["unit_price_cny"].mean().to_dict()
    for pname in sorted(avg_prices.keys()):
        cat = "coffee" if pname in COFFEE else "bakery"
        price = round(avg_prices[pname], 2)
        cost = DRINK_COSTS.get(pname, 1.50) if cat == "coffee" else calc_bread_cost(pname)
        cur.execute(
            "INSERT INTO products (product_name, category, unit_price, material_cost) VALUES (%s,%s,%s,%s)",
            (pname, cat, price, cost)
        )
    print(f"  {len(avg_prices)} products")

    # Build cost lookup
    cur.execute("SELECT product_name, material_cost FROM products")
    cost_map = {r[0]: float(r[1]) for r in cur.fetchall()}

    # 3. Create orders (schema_v2 + state)
    print("Creating orders table...")
    cur.execute("""
        CREATE TABLE orders (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ticket_id VARCHAR(20) NOT NULL,
            order_date DATE NOT NULL,
            order_time TIME,
            cashier_id VARCHAR(10) DEFAULT NULL,
            subtotal DECIMAL(10,2) DEFAULT 0,
            discount_total DECIMAL(10,2) DEFAULT 0,
            total_amount DECIMAL(10,2) DEFAULT 0,
            total_profit DECIMAL(10,2) DEFAULT 0,
            item_count INT DEFAULT 0,
            state VARCHAR(20) DEFAULT 'paid' CHECK (state IN ('draft','paid','cancelled','refunded')),
            INDEX idx_order_date (order_date),
            INDEX idx_ticket (ticket_id)
        )
    """)

    # 4. Create payments (independent table, Odoo-style)
    print("Creating payments table...")
    cur.execute("""
        CREATE TABLE payments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            order_id INT NOT NULL,
            amount DECIMAL(10,2) NOT NULL,
            payment_method VARCHAR(10) NOT NULL CHECK (payment_method IN ('cash','card','qr')),
            payment_date DATE,
            transaction_id VARCHAR(50),
            is_change BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
            INDEX idx_payment_order (order_id)
        )
    """)

    # 5. Create order_items (schema_v2)
    print("Creating order_items table...")
    cur.execute("""
        CREATE TABLE order_items (
            id INT AUTO_INCREMENT PRIMARY KEY,
            order_id INT NOT NULL,
            product_name VARCHAR(50),
            quantity INT DEFAULT 1,
            unit_price DECIMAL(8,2),
            discount_rate DECIMAL(5,3) DEFAULT 0,
            line_total DECIMAL(10,2),
            line_profit DECIMAL(10,2) DEFAULT 0,
            freshness VARCHAR(10) DEFAULT 'Fresh' CHECK (freshness IN ('Fresh','Day-1')),
            coffee_temp VARCHAR(10) DEFAULT NULL,
            coffee_ice VARCHAR(10) DEFAULT NULL,
            coffee_sugar VARCHAR(10) DEFAULT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
            INDEX idx_item_order (order_id)
        )
    """)

    # 6. Import data
    print("Importing orders and items...")
    PAYMENT_METHODS = ["cash", "card", "qr"]
    PAYMENT_WEIGHTS = [0.45, 0.20, 0.35]
    
    tickets = df.groupby(["date", "ticket_id"]).agg(
        order_time=("time", "first"),
        items=("product_name", list),
        qtys=("quantity", list),
        prices=("unit_price_cny", list)
    ).reset_index()

    ocount = icount = 0
    for i, row in tickets.iterrows():
        odate = str(row["date"])[:10]
        otime = str(row["order_time"])
        tid = str(int(row["ticket_id"]))

        subtotal = total = profit = nitems = 0
        for j in range(len(row["items"])):
            pname = row["items"][j]
            qty = int(row["qtys"][j])
            price = float(row["prices"][j])
            cost = cost_map.get(pname, price * 0.30)
            line_total = qty * price
            line_profit = line_total - (qty * cost)
            subtotal += line_total
            profit += line_profit
            nitems += qty
        total = subtotal  # no discount in historical data

        cur.execute(
            """INSERT INTO orders (ticket_id, order_date, order_time, subtotal, discount_total,
               total_amount, total_profit, item_count, state)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (tid, odate, otime, round(subtotal,2), 0.00, round(total,2), round(profit,2), nitems, 'paid')
        )
        oid = cur.lastrowid

        # Insert payment (one per order)
        pmethod = random.choices(PAYMENT_METHODS, PAYMENT_WEIGHTS)[0]
        cur.execute(
            "INSERT INTO payments (order_id, amount, payment_method, payment_date) VALUES (%s,%s,%s,%s)",
            (oid, round(total,2), pmethod, odate)
        )

        # Insert order items
        for j in range(len(row["items"])):
            pname = row["items"][j]
            qty = int(row["qtys"][j])
            price = float(row["prices"][j])
            cost = cost_map.get(pname, price * 0.30)
            line_total = qty * price
            line_profit = line_total - (qty * cost)
            cur.execute(
                """INSERT INTO order_items (order_id, product_name, quantity, unit_price,
                   discount_rate, line_total, line_profit, freshness)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (oid, pname, qty, price, 0.0, round(line_total,2), round(line_profit,2), 'Fresh')
            )
            icount += 1

        ocount += 1
        if ocount % 20000 == 0:
            print(f"  {ocount}/{len(tickets)} orders...")
            db.commit()

    db.commit()
    print(f"\nDone: {ocount} orders, {icount} items imported")

    # Verify
    cur.execute("SELECT COUNT(*) FROM orders")
    print(f"  orders: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM order_items")
    print(f"  items:  {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM payments")
    print(f"  payments: {cur.fetchone()[0]}")
    cur.execute("SELECT MIN(order_date), MAX(order_date) FROM orders")
    dr = cur.fetchone()
    print(f"  Date range: {dr[0]} ~ {dr[1]}")

if __name__ == "__main__":
    main()
